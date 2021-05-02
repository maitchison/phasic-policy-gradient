import argparse

DEFAULT_LOG_DIR = "./"

def train_fn(env_name="coinrun",
             distribution_mode="hard",
             arch="dual",  # 'shared', 'detach', or 'dual'
             # 'shared' = shared policy and value networks
             # 'dual' = separate policy and value networks
             # 'detach' = shared policy and value networks, but with the value function gradient detached during the policy phase to avoid interference
             interacts_total=100_000_000,
             num_envs=256,
             n_epoch_pi=1,
             n_epoch_vf=1,
             gamma=.999,
             aux_lr=5e-4,
             lr=5e-4,
             nminibatch=8,
             aux_mbsize=4,
             clip_param=.2,
             kl_penalty=0.0,
             n_aux_epoch_pi=6,
             n_aux_epoch_vf=6,
             v_mixing=False,
             n_pi=32,
             beta_clone=1.0,
             vf_true_weight=1.0,
             vtarget_mode="rollout",
             log_dir=DEFAULT_LOG_DIR,
             comm=None):

    from phasic import ppg
    from phasic import torch_util as tu
    from phasic.impala_cnn import ImpalaEncoder
    from phasic import logger
    from phasic.envs import get_venv
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD
    tu.setup_dist(comm=comm)
    tu.register_distributions_for_tree_util()

    print(f"World size is {comm.size}")

    # we take num_envs as number of environments total, not per worker.
    assert num_envs % comm.size == 0
    num_envs //= comm.size

    if log_dir is not None:
        format_strs = ['csv', 'stdout'] if comm.Get_rank() == 0 else []
        logger.configure(comm=comm, dir=log_dir, format_strs=format_strs)

    venv = get_venv(num_envs=num_envs, env_name=env_name, distribution_mode=distribution_mode)

    enc_fn = lambda obtype: ImpalaEncoder(
        obtype.shape,
        outsize=256,
        chans=(16, 32, 32),
    )
    model = ppg.PhasicValueModel(venv.ob_space, venv.ac_space, enc_fn, arch=arch, vtarget_mode=vtarget_mode)

    model.to(tu.dev())
    logger.log(tu.format_model(model))
    tu.sync_params(model.parameters())

    # note: for the moment vf_true and vf_aux share the same weight hyper parameter
    name2coef = {"pol_distance": beta_clone, "vf_true": vf_true_weight, "vf_aux": vf_true_weight}

    ppg.learn(
        venv=venv,
        model=model,
        interacts_total=interacts_total,
        ppo_hps=dict(
            lr=lr,
            γ=gamma,
            λ=0.95,
            nminibatch=nminibatch,
            n_epoch_vf=n_epoch_vf,
            n_epoch_pi=n_epoch_pi,
            clip_param=clip_param,
            kl_penalty=kl_penalty,
            log_save_opts={"save_mode": "last"}
        ),
        aux_lr=aux_lr,
        aux_mbsize=aux_mbsize,
        n_aux_epoch_pi=n_aux_epoch_pi,
        n_aux_epoch_vf=n_aux_epoch_vf,
        v_mixing=v_mixing,
        n_pi=n_pi,
        name2coef=name2coef,
        comm=comm,
    )

def main():

    parser = argparse.ArgumentParser(description='Process PPG training arguments.')
    parser.add_argument('run', type=str, default='experiment')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--num_envs', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n_epoch_pi', type=int, default=1)
    parser.add_argument('--n_epoch_vf', type=int, default=1)
    parser.add_argument('--n_aux_epoch_pi', type=int, default=6)
    parser.add_argument('--n_aux_epoch_vf', type=int, default=6)
    parser.add_argument('--n_pi', type=int, default=32)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--micro_batch_size', type=int, default=1024)
    parser.add_argument('--kl_penalty', type=float, default=0.0)
    parser.add_argument('--aux_lr', type=float, default=5e-4)
    parser.add_argument('--amp', type=bool, default=False, help="Enabled Automatic Mixed Precision on aux_phase.")
    parser.add_argument('--v_mixing', type=bool, default=False, help="Enables off-policy mixing during ppo training phase.")
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--vtarget_mode', type=str, default='rollout', help="[rollout|vtrace]")
    parser.add_argument('--shuffle_time', type=bool, default=False)
    parser.add_argument('--arch', type=str, default='dual') # 'shared', 'detach', or 'dual'

    args = parser.parse_args()

    import os

    # these must be executed before torch import
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if args.device == 'auto':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    print(f"Set devices to: {os.environ['CUDA_VISIBLE_DEVICES']}")
    os.environ["RCALL_LOGDIR"] = "./"

    # handle micro_batch_size
    import phasic.ppo
    phasic.ppo.MICRO_BATCH_SIZE = args.micro_batch_size

    # handle shuffle
    import phasic.minibatch_optimize
    phasic.minibatch_optimize.MB_SHUFFLE_TIME = args.shuffle_time

    # handle vtrace / amp
    import phasic.ppg
    phasic.ppg.USE_VTRACE = "vtrace" in args.vtarget_mode
    phasic.ppg.USE_AMP = args.amp

    # handle MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # train
    train_fn(
        env_name=args.env_name,
        num_envs=args.num_envs,
        n_epoch_pi=args.n_epoch_pi,
        n_epoch_vf=args.n_epoch_vf,
        n_aux_epoch_pi=args.n_aux_epoch_pi,
        n_aux_epoch_vf=args.n_aux_epoch_vf,
        interacts_total=int(args.epochs*1e6),
        n_pi=args.n_pi,
        arch=args.arch,
        v_mixing=args.v_mixing,
        log_dir=args.run,
        vtarget_mode=args.vtarget_mode,
        aux_lr=args.aux_lr,
        comm=comm)

if __name__ == '__main__':
    main()
