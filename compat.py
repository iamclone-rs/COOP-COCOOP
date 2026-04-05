import inspect

import torch
from torch.optim import lr_scheduler


def patch_torch_lr_scheduler_verbose():
    """Allow older callers to pass ``verbose`` on newer PyTorch releases."""

    for name in ("LRScheduler", "_LRScheduler"):
        scheduler_cls = getattr(lr_scheduler, name, None)
        if scheduler_cls is None:
            continue

        try:
            signature = inspect.signature(scheduler_cls.__init__)
        except (TypeError, ValueError):
            continue

        if "verbose" in signature.parameters:
            continue

        original_init = scheduler_cls.__init__
        if getattr(original_init, "_coop_verbose_compat", False):
            continue

        def compat_init(self, optimizer, last_epoch=-1, verbose=None, _orig=original_init):
            return _orig(self, optimizer, last_epoch)

        compat_init._coop_verbose_compat = True
        compat_init.__doc__ = original_init.__doc__
        compat_init.__name__ = original_init.__name__
        compat_init.__qualname__ = original_init.__qualname__
        scheduler_cls.__init__ = compat_init


def patch_torch_load_weights_only_default():
    """Restore the pre-2.6 torch.load default for legacy checkpoints."""

    try:
        signature = inspect.signature(torch.load)
    except (TypeError, ValueError):
        return

    if "weights_only" not in signature.parameters:
        return

    original_load = torch.load
    if getattr(original_load, "_coop_weights_only_compat", False):
        return

    def compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    compat_load._coop_weights_only_compat = True
    compat_load.__doc__ = original_load.__doc__
    compat_load.__name__ = original_load.__name__
    compat_load.__qualname__ = original_load.__qualname__
    torch.load = compat_load


patch_torch_lr_scheduler_verbose()
patch_torch_load_weights_only_default()
