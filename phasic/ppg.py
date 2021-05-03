import torch

from . import ppo
from . import logger
from . import minibatch_optimize as mbo
from . import vtrace

from collections import defaultdict

import torch as th
import itertools
from . import torch_util as tu
from torch import distributions as td
from .distr_builder import distr_builder
from mpi4py import MPI
from .tree_util import tree_map, tree_reduce
import operator

USE_VTRACE = False
USE_AMP = False

def sum_nonbatch(logprob_tree):
    """
    sums over nonbatch dimensions and over all leaves of the tree
    use with nested action spaces, which require Product distributions
    """
    return tree_reduce(operator.add, tree_map(tu.sum_nonbatch, logprob_tree))


class PpoModel(th.nn.Module):
    def forward(self, ob, first, state_in) -> "pd, vpred, aux, state_out":
        raise NotImplementedError

    @tu.no_grad
    def act(self, ob, first, state_in):
        pd, vpred, _, state_out = self(
            ob=tree_map(lambda x: x[:, None], ob),
            first=first[:, None],
            state_in=state_in,
        )
        ac = pd.sample()
        logp = sum_nonbatch(pd.log_prob(ac))
        return (
            tree_map(lambda x: x[:, 0], ac),
            state_out,
            dict(vpred=vpred[:, 0], logp=logp[:, 0]),
        )

    @tu.no_grad
    def v(self, ob, first, state_in):
        _pd, vpred, _, _state_out = self(
            ob=tree_map(lambda x: x[:, None], ob),
            first=first[:, None],
            state_in=state_in,
        )
        return vpred[:, 0]

class PhasicModel(PpoModel):
    def forward(self, ob, first, state_in) -> "pd, vpred, aux, state_out":
        raise NotImplementedError

    def compute_aux_loss(self, aux, mb):
        raise NotImplementedError

    def initial_state(self, batchsize):
        raise NotImplementedError

    def aux_keys(self) -> "list of keys needed in mb dict for compute_aux_loss":
        raise NotImplementedError

    def set_aux_phase(self, is_aux_phase: bool):
        "sometimes you want to modify the model, e.g. add a stop gradient"


class PhasicValueModel(PhasicModel):
    def __init__(
        self,
        obtype,
        actype,
        enc_fn,
        vtarget_mode="rollout", # [rollout|vtrace]
        arch="dual",  # shared, detach, dual
    ):
        super().__init__()

        detach_value_head = False
        vf_keys = None
        pi_key = "pi"

        self.scaler = th.cuda.amp.GradScaler()

        if arch == "shared":
            true_vf_key = "pi"
        elif arch == "detach":
            true_vf_key = "pi"
            detach_value_head = True
        elif arch == "dual":
            true_vf_key = "vf"
        else:
            assert False

        vf_keys = vf_keys or [true_vf_key]
        self.pi_enc = enc_fn(obtype)
        self.pi_key = pi_key
        self.true_vf_key = true_vf_key
        self.vf_keys = vf_keys
        self.enc_keys = list(set([pi_key] + vf_keys))
        self.detach_value_head = detach_value_head
        pi_outsize, self.make_distr = distr_builder(actype)

        self.vtarget_mode = vtarget_mode

        for k in self.enc_keys:
            self.set_encoder(k, enc_fn(obtype))

        for k in self.vf_keys:
            lastsize = self.get_encoder(k).codetype.size
            self.set_vhead(k, tu.NormedLinear(lastsize, 1, scale=0.1))

        lastsize = self.get_encoder(self.pi_key).codetype.size
        self.pi_head = tu.NormedLinear(lastsize, pi_outsize, scale=0.1)
        self.aux_vf_head = tu.NormedLinear(lastsize, 1, scale=0.1)

    @property
    def params_pi(self):
        return [*self.pi_enc.parameters(), *self.pi_head.parameters(), *self.aux_vf_head.parameters()]

    @property
    def params_vf(self):
        return [*self.vf_enc.parameters(), *self.vf_vhead.parameters()]

    def compute_aux_loss(self, aux, seg, output: str = 'both'):

        aux_scale = 1.0
        true_scale = 1.0

        if self.vtarget_mode == "rollout":
            vtarg_aux = seg["vtarg"]
            vtarg_tru = seg["vtarg"]
        elif self.vtarget_mode == "vtrace":
            assert "vtarg_vtrace" in seg, "v-trace missing targets, make sure you calculate them when enabling v-trace."
            vtarg_aux = seg["vtarg_vtrace"]
            vtarg_tru = seg["vtarg_vtrace"]
        else:
            raise ValueError(f"invalid vtarget_mode {self.vtarget_mode}")

        result = {}

        if output in ["pi", "both"]:
            result["vf_aux"] = aux_scale * 0.5 * ((aux["vpredaux"] - vtarg_aux) ** 2).mean()

        if output in ["vf", "both"]:
            result["vf_true"] = true_scale * 0.5 * ((aux["vpredtrue"] - vtarg_tru) ** 2).mean()

        return result


    def reshape_x(self, x):
        b, t = x.shape[:2]
        x = x.reshape(b, t, -1)

        return x

    def get_encoder(self, key):
        return getattr(self, key + "_enc")

    def set_encoder(self, key, enc):
        setattr(self, key + "_enc", enc)

    def get_vhead(self, key):
        return getattr(self, key + "_vhead")

    def set_vhead(self, key, layer):
        setattr(self, key + "_vhead", layer)

    def forward(self, ob, first, state_in, output: str = 'all'):
        """
        Output is pi|fv|all
        """
        state_out = {}
        x_out = {}

        encoder_keys = [output] if output != 'all' else self.enc_keys

        for k in encoder_keys:
            x_out[k], state_out[k] = self.get_encoder(k)(ob, first, state_in[k])
            x_out[k] = self.reshape_x(x_out[k])

        if output in ['pi', 'all']:
            pi_x = x_out[self.pi_key]
            pivec = self.pi_head(pi_x)
            pd = self.make_distr(pivec)
        else:
            pi_x = None
            pd = None

        aux = {}
        if output in ['vf', 'all']:
            for k in self.vf_keys:
                if self.detach_value_head:
                    x_out[k] = x_out[k].detach()
                aux[k] = self.get_vhead(k)(x_out[k])[..., 0]
            vfvec = aux[self.true_vf_key]
            aux['vpredtrue'] = vfvec
        else:
            vfvec = None

        if pi_x is not None:
            aux["vpredaux"] = self.aux_vf_head(pi_x)[..., 0]

        return pd, vfvec, aux, state_out

    def initial_state(self, batchsize):
        return {k: self.get_encoder(k).initial_state(batchsize) for k in self.enc_keys}

    def aux_keys(self):
        return ["vtarg"]

def make_minibatches(segs, mbsize, force_no_time_shuffle=False):
    """
    Yield one epoch of minibatch over the dataset described by segs
    Each minibatch mixes data between different segs

    Data in input segemnts should be of shape [n, t]
    Output will be [mbsize, t]
    """
    b, t, *state_shape = segs[0]["ob"].shape
    s = len(segs)

    envs_segs = th.tensor(list(itertools.product(range(b), range(s))))

    # filter state_in as they have shape (256,0)
    segs = [{k: v for k, v in seg.items() if k != "state_in"} for seg in segs]
    fake_state = {
        'pi': th.zeros([mbsize, 0]),
        'vf': th.zeros([mbsize, 0])
    }

    if mbo.MB_SHUFFLE_TIME and not force_no_time_shuffle:
        # not be best work, but it'll do the trick
        reshaped_segs = tree_map(lambda x: mbo.merge_down(x, b, t), segs)
        for mbinds in th.randperm(t * b * s).split(mbsize*t):
            result = tree_map(lambda x: mbo.expand_up(x, mbsize, t),
            tu.tree_stack(
                [tu.tree_slice(reshaped_segs[ind % s], ind // s) for ind in mbinds]
            ))
            result["state_in"] = fake_state
            yield result
    else:
        for perminds in th.randperm(len(envs_segs)).split(mbsize):
            esinds = envs_segs[perminds]
            result = tu.tree_stack(
                [tu.tree_slice(segs[segind], envind) for (envind, segind) in esinds]
            )
            result["state_in"] = fake_state
            yield result


def aux_train(*, model, segs, opt, mbsize, name2coef, module:str):
    """
    Train on auxiliary loss + policy KL + vf distance

    module is pi|fv
    """

    def clip_and_sync():
        tu.sync_grads(model.parameters())
        grad = th.nn.utils.clip_grad_norm_(tu.get_opt_params(opt), 20)
        logger.logkv_mean(f"grad_aux", grad)

    optional_keys = {"vtarg_vtrace", "oldvpred"}

    needed_keys = {"ob", "first", "state_in", "oldpd"}.union(model.aux_keys())
    needed_keys = needed_keys.union(set(k for k in optional_keys if k in segs[0]))

    segs = [{k: seg[k] for k in needed_keys} for seg in segs]
    for mb in make_minibatches(segs, mbsize):

        opt.zero_grad()

        with th.cuda.amp.autocast(enabled=USE_AMP):

            mb = tree_map(lambda x: x.to(tu.dev()), mb)

            pd, _, aux, _state_out = model(mb["ob"], mb["first"], mb["state_in"], output=module)
            name2loss = {}

            if module == "pi":
                # this is only needed if we are training pi module.
                name2loss["pol_distance"] = td.kl_divergence(mb["oldpd"], pd).mean()

            name2loss.update(model.compute_aux_loss(aux, mb, module))

            loss = 0
            for name in name2loss.keys():
                unscaled_loss = name2loss[name]
                scaled_loss = unscaled_loss * name2coef.get(name, 1)
                logger.logkv_mean("unscaled/" + name, unscaled_loss)
                logger.logkv_mean("scaled/" + name, scaled_loss)
                loss += scaled_loss

        if USE_AMP:
            clip_and_sync()
            model.scaler.scale(loss).backward()
            model.scaler.step(opt)
            model.scaler.update()
        else:
            loss.backward()
            clip_and_sync()
            opt.step()


def compute_vtrace_targets(seg, counter, ppo_hps, verbose=True):
    """
    Compute v-trace targets for given segment
        seg["logp"] the logprobs of the actions taken during rollout
        seg["oldpd"] the probability distribution calculated during presleep phase
        seg["oldvpred"] the (uncorrected) value estimates calculated during presleep phase
        seg["ac"] actions sampled during rollout
        seg["reward"] rewards generated during rollout
        seg["first"] indicates if state was first in an episode

        seg["finaloldvpred"] estimate of value for final_obs

        ppo_hps: used to make sure gamma, and lambda match
    """

    # data in segments is formatted as (B, T, *)

    # shift finals back one to get terminal states.
    dones = th.cat([seg['first'][:, 1:], seg['finalfirst'][:, None]], dim=1).float()

    def transpose_bt(x):
        return th.transpose(x, 0, 1)

    # this function expects data in t,b format, so we need to swap everything around
    vs, weighted_adv, cs = vtrace.importance_sampling_v_trace(
        behaviour_log_prob=transpose_bt(seg['logp']),
        target_log_prob=transpose_bt(seg['oldpd'].log_prob(seg['ac'])),
        rewards=transpose_bt(seg['reward']),
        dones=transpose_bt(dones),
        target_value_estimates=transpose_bt(seg["oldvpred"]),
        target_value_final_estimate=seg["finaloldvpred"],
        gamma=ppo_hps["γ"],
        lamb=ppo_hps["λ"],
    )

    seg['vtarg_vtrace'] = transpose_bt(vs)

    # show some debug information to make sure everything is ok
    # v_roll_delta is how far v-trace value estimates are away from what PPG would normally train on (which are the rollout values)
    # v_pred_delta is how far v-trace value estimates are away from predictions made at the beginning of the aux_phase

    if verbose:
        v_roll_delta = th.mean((seg['vtarg'] - seg['vtarg_vtrace'])**2)
        v_pred_delta = th.mean((seg['oldvpred'] - seg['vtarg_vtrace']) ** 2)
        v_diff_max = th.max(th.abs(seg['vtarg'] - seg['vtarg_vtrace']))

        if type(counter) == 0:
            counter = counter.zfill(2)

        print(f"" + \
              f"Segment: {counter} cs:{cs.mean():.2f} +- {cs.std():.3f}," +
              f"v_pred_delta:{v_pred_delta:.3f} v_roll_delta:{v_roll_delta:.3f}, " +
              f"v_diff_max: {v_diff_max:.2f} " +
              f"v_diff_max: {v_diff_max:.2f} " +
              f"Dones: {dones.sum()} " +
              f"Rewards {seg['reward'].sum()}"
              )
        logger.logkv_mean(f"vtrace/cs_mu_{counter}", cs.mean())
        logger.logkv_mean(f"vtrace/cs_std_{counter}", cs.std())
        logger.logkv_mean(f"vtrace/v_roll_delta_{counter}", v_roll_delta)
        logger.logkv_mean(f"vtrace/v_pred_delta_{counter}", v_pred_delta)
        logger.logkv_mean(f"vtrace/v_pred_max_{counter}", v_diff_max)


def compute_presleep_outputs(
    *, model, segs, mbsize, pdkey="oldpd", vpredkey="oldvpred", ppo_hps, include_vtrace=False
):
    def forward(ob, first, state_in):
        pd, vpred, _aux, _state_out = model.forward(ob.to(tu.dev()), first, state_in)
        # I bring these back to the cpu so everything in seg is on the same device.
        # (also saves a bit of GPU ram if we have lots of segments)
        pd.logits = pd.logits.cpu()
        vpred = vpred.cpu()
        return pd, vpred

    for i, seg in enumerate(segs):
        seg[pdkey], seg[vpredkey] = tu.minibatched_call(
            forward, mbsize, ob=seg["ob"], first=seg["first"], state_in=seg["state_in"]
        )
        if include_vtrace:
            # generate the final value estimate
            with th.no_grad():
                pd, vpred = forward(seg["finalob"][:, None], seg["finalfirst"][:, None], tree_map(lambda x: x[:, None], seg["finalstate"]))
            seg["finaloldvpred"] = vpred[:, 0]
            # generate vtrace estimates
            compute_vtrace_targets(seg=seg, counter=i if len(segs) > 1 else 'xx', ppo_hps=ppo_hps)



def learn(
    *,
    model,
    venv,
    ppo_hps,
    aux_lr,
    aux_mbsize,
    n_aux_epoch_pi=6,
    n_aux_epoch_vf=6,
    v_mixing=False,
    n_pi=32,
    kl_ewma_decay=None,
    interacts_total=float("inf"),
    name2coef=None,
    comm=None,
):
    """
    Run PPO for X iterations
    Then minimize aux loss + KL + value distance for X passes over data
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    ppo_state = None

    opt_vf = th.optim.Adam(model.params_vf, lr=aux_lr)
    opt_pi = th.optim.Adam(model.params_pi, lr=aux_lr)

    name2coef = name2coef or {}

    use_aux_phase = n_aux_epoch_pi != 0 or n_aux_epoch_vf != 0

    while True:
        store_segs = n_pi != 0 and use_aux_phase

        # Policy phase
        ppo_state = ppo.learn(
            venv=venv,
            model=model,
            learn_state=ppo_state,
            callbacks=[
                lambda _l: n_pi > 0 and _l["curr_iteration"] >= n_pi,
            ],
            interacts_total=interacts_total,
            store_segs=store_segs,
            v_mixing=v_mixing,
            comm=comm,
            **ppo_hps,
        )

        if ppo_state["curr_interact_count"] >= interacts_total:
            break

        if use_aux_phase:
            segs = ppo_state["seg_buf"]
            compute_presleep_outputs(model=model, segs=segs, mbsize=aux_mbsize, ppo_hps=ppo_hps, include_vtrace=USE_VTRACE)

            for i in range(n_aux_epoch_pi):
                logger.log(f"Aux epoch pi_{i}")
                aux_train(
                    model=model,
                    segs=segs,
                    opt=opt_pi,
                    mbsize=aux_mbsize,
                    name2coef=name2coef,
                    module='pi',
                )
                logger.dumpkvs()

            for i in range(n_aux_epoch_vf):
                logger.log(f"Aux epoch vf_{i}")
                aux_train(
                    model=model,
                    segs=segs,
                    opt=opt_vf,
                    mbsize=aux_mbsize,
                    name2coef=name2coef,
                    module='vf',
                )
                logger.dumpkvs()
