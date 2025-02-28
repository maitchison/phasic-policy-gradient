"""
Mostly copied from ppo.py but with some extra options added that are relevant to phasic
"""
import random

import torch as th
from . import vtrace
from mpi4py import MPI
from .tree_util import tree_map
from . import torch_util as tu
from .log_save_helper import LogSaveHelper
from .minibatch_optimize import minibatch_optimize
from .roller import Roller
from .reward_normalizer import RewardNormalizer
from . import ppg

import math
from . import logger

INPUT_KEYS = {"ob", "ac", "first", "logp", "vtarg", "adv", "state_in"}
MICRO_BATCH_SIZE = 1024

def compute_gae(
    *,
    vpred: "(th.Tensor[1, float]) value predictions",
    reward: "(th.Tensor[1, float]) rewards",
    first: "(th.Tensor[1, bool]) mark beginning of episodes",
    γ: "(float)",
    λ: "(float)"
):
    orig_device = vpred.device
    assert orig_device == reward.device == first.device
    vpred, reward, first = (x.cpu() for x in (vpred, reward, first))
    first = first.to(dtype=th.float32)
    assert first.dim() == 2
    nenv, nstep = reward.shape
    assert vpred.shape == first.shape == (nenv, nstep + 1)
    adv = th.zeros(nenv, nstep, dtype=th.float32)
    lastgaelam = 0
    for t in reversed(range(nstep)):
        notlast = 1.0 - first[:, t + 1]
        nextvalue = vpred[:, t + 1]
        # notlast: whether next timestep is from the same episode
        delta = reward[:, t] + notlast * γ * nextvalue - vpred[:, t]
        adv[:, t] = lastgaelam = delta + notlast * γ * λ * lastgaelam
    vtarg = vpred[:, :-1] + adv
    return adv.to(device=orig_device), vtarg.to(device=orig_device)

def log_vf_stats(comm, **kwargs):
    logger.logkv(
        "VFStats/EV", tu.explained_variance(kwargs["vpred"], kwargs["vtarg"], comm)
    )
    for key in ["vpred", "vtarg", "adv"]:
        logger.logkv_mean(f"VFStats/{key.capitalize()}Mean", kwargs[key].mean())
        logger.logkv_mean(f"VFStats/{key.capitalize()}Std", kwargs[key].std())

def compute_advantage(model, seg, γ, λ, comm=None):
    comm = comm or MPI.COMM_WORLD
    finalob, finalfirst = seg["finalob"], seg["finalfirst"]
    vpredfinal = model.v(finalob, finalfirst, seg["finalstate"])
    reward = seg["reward"]
    logger.logkv("Misc/FrameRewMean", reward.mean())
    adv, vtarg = compute_gae(
        γ=γ,
        λ=λ,
        reward=reward,
        vpred=th.cat([seg["vpred"], vpredfinal[:, None]], dim=1),
        first=th.cat([seg["first"], finalfirst[:, None]], dim=1),
    )
    log_vf_stats(comm, adv=adv, vtarg=vtarg, vpred=seg["vpred"])
    seg["vtarg"] = vtarg
    adv_mean, adv_var = tu.mpi_moments(comm, adv)
    seg["adv"] = (adv - adv_mean) / (math.sqrt(adv_var) + 1e-8)

def compute_losses(
    model,
    ob,
    ac,
    first,
    logp,
    vtarg,
    adv,
    state_in,
    clip_param,
    vfcoef,
    entcoef,
    kl_penalty,
    output:str="all"
):
    losses = {}
    diags = {}

    pd, vpred, aux, _state_out = model(ob=ob, first=first, state_in=state_in, output=output)

    if output in ['pi', 'all']:
        newlogp = tu.sum_nonbatch(pd.log_prob(ac))
        # prob ratio for KL / clipping based on a (possibly) recomputed logp
        logratio = newlogp - logp
        ratio = th.exp(logratio)

        if clip_param > 0:
            pg_losses = -adv * ratio
            pg_losses2 = -adv * th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            pg_losses = th.max(pg_losses, pg_losses2)
        else:
            pg_losses = -adv * th.exp(newlogp - logp)

        diags["entropy"] = entropy = tu.sum_nonbatch(pd.entropy()).mean()
        diags["negent"] = -entropy * entcoef
        diags["pg"] = pg_losses.mean()
        diags["pi_kl"] = kl_penalty * 0.5 * (logratio ** 2).mean()

        losses["pi"] = diags["negent"] + diags["pg"] + diags["pi_kl"]

        with th.no_grad():
            diags["clipfrac"] = (th.abs(ratio - 1) > clip_param).float().mean()
            diags["approxkl"] = 0.5 * (logratio ** 2).mean()

    if output in ['vf', 'all']:
        losses["vf"] = vfcoef * ((vpred - vtarg) ** 2).mean()


    return losses, diags

def learn(
    *,
    venv: "(VecEnv) vectorized environment",
    model: "(ppo.PpoModel)",
    interacts_total: "(float) total timesteps of interaction" = float("inf"),
    nstep: "(int) number of serial timesteps" = 256,
    γ: "(float) discount" = 0.99,
    λ: "(float) GAE parameter" = 0.95,
    clip_param: "(float) PPO parameter for clipping prob ratio" = 0.2,
    vfcoef: "(float) value function coefficient" = 0.5,
    entcoef: "(float) entropy coefficient" = 0.01,
    nminibatch: "(int) number of minibatches to break epoch of data into" = 8, # stub, see if we get amp improvement?
    n_epoch_vf: "(int) number of epochs to use when training the value function" = 1,
    n_epoch_pi: "(int) number of epochs to use when training the policy" = 1,
    lr: "(float) Adam learning rate" = 5e-4,
    default_loss_weights: "(dict) default_loss_weights" = {},
    store_segs: "(bool) whether or not to store segments in a buffer" = True,
    verbose: "(bool) print per-epoch loss stats" = True,
    log_save_opts: "(dict) passed into LogSaveHelper" = {},
    rnorm: "(bool) reward normalization" = True,
    kl_penalty: "(int) weight of the KL penalty, which can be used in place of clipping" = 0,
    grad_weight: "(float) relative weight of this worker's gradients" = 1,
    comm: "(MPI.Comm) MPI communicator" = None,
    v_mixing: False,
    callbacks: "(seq of function(dict)->bool) to run each update" = (),
    learn_state: "dict with optional keys {'opts', 'roller', 'lsh', 'reward_normalizer', 'curr_interact_count', 'seg_buf'}" = None,
):
    if comm is None:
        comm = MPI.COMM_WORLD

    learn_state = learn_state or {}
    ic_per_step = venv.num * comm.size * nstep

    all_params = list(model.parameters())

    params = {}

    opt_keys = ["pi", "vf"]
    # for the moment use all params...
    params['pi'] = model.params_pi
    params['vf'] = model.params_vf

    opts = learn_state.get("opts") or {k: th.optim.Adam(params[k], lr=lr) for k in opt_keys}

    tu.sync_params(all_params)

    if rnorm:
        reward_normalizer = learn_state.get("reward_normalizer") or RewardNormalizer(venv.num)
    else:
        reward_normalizer = None

    def get_weight(k):
        return default_loss_weights[k] if k in default_loss_weights else 1.0

    def train_with_losses_and_opt(loss_keys, opt, output:str, **arrays):

        # arrays obs is [B, T, H, W, C]

        opt.zero_grad()

        b, t, *state_shape = arrays["ob"].shape
        batch_size = b*t
        n_micro_batches = math.ceil(batch_size / MICRO_BATCH_SIZE)

        for micro_batch in range(n_micro_batches):

            # upload data to correct device
            uploaded_arrays = tree_map(lambda x: tu.split_and_upload(x, micro_batch, n_micro_batches), arrays)

            losses, diags = compute_losses(
                model,
                entcoef=entcoef,
                kl_penalty=kl_penalty,
                clip_param=clip_param,
                vfcoef=vfcoef,
                output=output,
                **uploaded_arrays,
            )
            loss = sum([losses[k] * get_weight(k) for k in loss_keys])
            # because we mean over the batches, we need to divide by micro batch count so that gradient is not scaled.
            loss = loss / n_micro_batches
            loss.backward()

        # this doesn't really work when we have optimizers covering only part of the model...
        # tu.warn_no_gradient(model, "PPO")

        tu.sync_grads(all_params, grad_weight=grad_weight)
        diags = {k: v.detach() for (k, v) in diags.items()}

        # clip grads after we have accumulated them
        grad = th.nn.utils.clip_grad_norm_(tu.get_opt_params(opt), 20)
        logger.logkv_mean(f"grad_{'-'.join(loss_keys)}", grad)

        opt.step()
        diags.update({f"loss_{k}": v.detach() for (k, v) in losses.items()})
        return diags

    def train_pi(**arrays):
        return train_with_losses_and_opt(["pi"], opts["pi"], 'pi', **arrays)

    def train_vf(**arrays):
        return train_with_losses_and_opt(["vf"], opts["vf"], 'vf', **arrays)

    def train_mixture(**arrays):
        """
        Train value function on mixture of off and on policy data.

        Input is mini_batch arrays in [B,T, * ] format on device CPU

        """

        B, T, *state_shape = arrays["ob"].shape
        segs = seg_buf
        ppo_hps = {
            'γ': γ,
            'λ': λ,
        }

        # 1. sample from our replay buffer
        # this will be with replacement, but that is just fine as the buffer is quite large.
        off_policy_mb = next(ppg.make_minibatches(segs, mbsize=B*3, force_no_time_shuffle=True))

        # 2. generate updated value estimates
        ppg.compute_presleep_outputs(
            model=model, segs=[off_policy_mb], mbsize=4, pdkey="oldpd", vpredkey="oldvpred", ppo_hps=ppo_hps,
            include_vtrace=True
        )
        off_policy_mb['vtarg'] = off_policy_mb['vtarg_vtrace']
        off_policy_mb = {k: v for k, v in off_policy_mb.items() if k in arrays}

        # 3. combine everything into one big batch.
        combined_arrays = tu.tree_cat([arrays, off_policy_mb])

        # 4. train on this
        return train_with_losses_and_opt(["vf"], opts["vf"], 'vf', **combined_arrays)

    def train_pi_and_vf(**arrays):
        return train_with_losses_and_opt(["pi", "vf"], opts["pi"], 'all', **arrays)

    roller = learn_state.get("roller") or Roller(
        act_fn=model.act,
        venv=venv,
        initial_state=model.initial_state(venv.num),
        keep_buf=100,
        keep_non_rolling=log_save_opts.get("log_new_eps", False),
    )

    lsh = learn_state.get("lsh") or LogSaveHelper(
        ic_per_step=ic_per_step,
        model=model,
        learn_state={
            'opts': opts,
            'reward_normalizer': reward_normalizer
        },
        comm=comm,
        **log_save_opts
    )

    callback_exit = False  # Does callback say to exit loop?

    curr_interact_count = learn_state.get("curr_interact_count") or 0
    curr_iteration = 0
    seg_buf = learn_state.get("seg_buf") or []

    # cull unwanted data from pre-existing buffer (otherwise we won't be able to concatinate them)
    required_keys = ('ac', 'adv', 'finalfirst', 'finalob', 'finalstate', 'first', 'logp', 'ob', 'reward', 'vpred', 'vtarg')
    for i in range(len(seg_buf)):
        seg_buf[i] = {k: v for k, v in seg_buf.items() if k in required_keys}

    while curr_interact_count < interacts_total and not callback_exit:
        seg = roller.multi_step(nstep)
        lsh.gather_roller_stats(roller)
        if rnorm:
            seg["reward"] = reward_normalizer(seg["reward"], seg["first"])
        compute_advantage(model, seg, γ, λ, comm=comm)

        if store_segs:
            seg_to_store = tree_map(lambda x: x.cpu(), seg)
            if curr_iteration < len(seg_buf):
                # just reuse buffer
                seg_buf[curr_iteration] = seg_to_store
                print(f"Expanded buffer to size {len(seg_buf)}")
            else:
                seg_buf.append(seg_to_store)

        with logger.profile_kv("optimization"):

            # note: we should probably update policy first, then value.. need to think about this a bit more though
            # also I think maybe we could vtrace the on-policy data, as it's now slightly off-policy.
            # this is esentially the AGAE idea...

            minibatch_optimize(
                train_mixture if v_mixing else train_vf,
                {k: seg[k] for k in INPUT_KEYS},
                nminibatch=nminibatch,
                comm=comm,
                nepoch=n_epoch_vf,
                verbose=verbose,
            )

            epoch_stats = minibatch_optimize(
                train_pi,
                {k: seg[k] for k in INPUT_KEYS},
                nminibatch=nminibatch,
                comm=comm,
                nepoch=n_epoch_pi,
                verbose=verbose,
            )
            for (k, v) in epoch_stats[-1].items():
                logger.logkv("Opt/" + k, v)

        lsh()

        curr_interact_count += ic_per_step
        curr_iteration += 1

        for callback in callbacks:
            callback_exit = callback_exit or bool(callback(locals()))

    return dict(
        opts=opts,
        roller=roller,
        lsh=lsh,
        reward_normalizer=reward_normalizer,
        curr_interact_count=curr_interact_count,
        seg_buf=seg_buf,
    )