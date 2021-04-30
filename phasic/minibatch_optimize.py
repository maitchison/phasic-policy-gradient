import torch
import torch as th
from phasic.tree_util import tree_map
from phasic import torch_util as tu, logger

MB_SHUFFLE_TIME = False

def _fmt_row(width, row, header=False):
    out = " | ".join(_fmt_item(x, width) for x in row)
    if header:
        out = out + "\n" + "-" * len(out)
    return out


def _fmt_item(x, l):
    if th.is_tensor(x):
        assert x.dim() == 0
        x = float(x)
    if isinstance(x, float):
        v = abs(x)
        if (v < 1e-4 or v > 1e4) and v > 0:
            rep = "%7.2e" % x
        else:
            rep = "%7.5f" % x
    else:
        rep = str(x)
    return " " * (l - len(rep)) + rep


class LossDictPrinter:
    """
    Helps with incrementally printing out stats row by row in a formatted table
    """

    def __init__(self):
        self.printed_header = False

    def print_row(self, d):
        if not self.printed_header:
            logger.log(_fmt_row(12, d.keys()))
            self.printed_header = True
        logger.log(_fmt_row(12, d.values()))

def minibatch_optimize(
    train_fn: "function (dict) -> dict called on each minibatch that returns training stats",
    tensordict: "Dict[str->th.Tensor]",
    *,
    nepoch: "(int) number of epochs over dataset",
    nminibatch: "(int) number of minibatch per epoch",
    comm: "(MPI.Comm) MPI communicator",
    verbose: "(bool) print detailed stats" = False,
    epoch_fn: "function () -> dict to be called each epoch" = None,
    shuffle_time=False,
):
    ntrain = tu.batch_len(tensordict)
    if nminibatch > ntrain:
        logger.log(f"Warning: nminibatch > ntrain!! ({nminibatch} > {ntrain})")
        nminibatch = ntrain
    ldp = LossDictPrinter()
    epoch_dicts = []

    for _ in range(nepoch):
        # must keep minibatches on CPU as we want to upload them microbatch at a time
        mb_dicts = [
            train_fn(**mb) for mb in minibatch_gen(tensordict, nminibatch=nminibatch, device="cpu")
        ]
        local_dict = {k: float(v) for (k, v) in dict_mean(mb_dicts).items()}
        if epoch_fn is not None:
            local_dict.update(dict_mean(epoch_fn()))
        global_dict = dict_mean(comm.allgather(local_dict))
        epoch_dicts.append(global_dict)
        if verbose:
            ldp.print_row(global_dict)

    return epoch_dicts


def dict_mean(ds):
    return {k: sum(d[k] for d in ds) / len(ds) for k in ds[0].keys()}


def to_th_device(x:torch.Tensor):
    return to_device(tu.dev())


def to_device(x:torch.Tensor, device):
    assert th.is_tensor(x), "to_th_device should only be applied to torch tensors"
    dtype = th.float32 if x.dtype == th.float64 else None
    return x.to(device, dtype=dtype)


def minibatch_gen(data, *, batch_size=None, nminibatch=None, forever=False, device=None):
    """
    Generator that produces shuffled minibatches delivered to specified device. If device is None then
    tu.dev() is used as the default.

    Inputs are expected to be of shape [b, t, *]

    If MB_SHUFFLE_TIME is enabled then time dimension (N) is shuffled aswell, but still returns
    minibatches of shape [b//nminibatch, t, *]

    """
    assert (batch_size is None) != (
        nminibatch is None
    ), "only one of batch_size or nminibatch should be specified"

    b, t, *state_shape = data['ob'].shape

    if nminibatch is None:
        nminibatch = max(b // batch_size, 1)

    mb_size = b // nminibatch

    # filter state_in as they have shape (256,0)
    if "state_in" in data and data["state_in"]["pi"].shape[1] == 0:
        data = {
            k: v for k, v in data.items() if k != "state_in"
        }
        fake_state = {
            'pi': th.zeros([mb_size, 0]),
            'vf': th.zeros([mb_size, 0])
        }
    else:
        fake_state = None


    while True:
        if MB_SHUFFLE_TIME:
            reshaped_data = tree_map(lambda x: merge_down(x, b, t), data)
            for mbinds in th.chunk(th.randperm(t*b), nminibatch):
                result = tree_map(
                    lambda x: to_device(expand_up(x, mb_size, t), device or tu.dev()),
                    tu.tree_slice(reshaped_data, mbinds)
                )
                if fake_state is not None:
                    result['state_in'] = fake_state
                yield result
        else:
            for mbinds in th.chunk(th.randperm(b), nminibatch):
                result = tree_map(lambda x: to_device(x, device or tu.dev()), tu.tree_slice(data, mbinds))
                if fake_state is not None:
                    result['state_in'] = fake_state
                yield result

        if not forever:
            return

def merge_down(x, b, t):
    _b, _t, *shape = x.shape
    assert b == _b and t == _t, f"Expected shape to be ({t}, {b}, *) but found {x.shape}"
    return x.reshape(b*t, *shape)

def expand_up(x, b, t):
    bt, *shape = x.shape
    assert bt == b*t, f"Expected shape to be ({b*t}, *) but found {x.shape}"
    return x.reshape(b, t, *shape)
