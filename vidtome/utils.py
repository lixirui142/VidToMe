import torch
from einops import rearrange

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False

def init_generator(device: torch.device, fallback: torch.Generator=None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback

def join_frame(x, fsize):
    """ Join multi-frame tokens """
    x = rearrange(x, "(B F) N C -> B (F N) C", F=fsize)
    return x

def split_frame(x, fsize):
    """ Split multi-frame tokens """
    x = rearrange(x, "B (F N) C -> (B F) N C", F=fsize)
    return x

def func_warper(funcs):
    """ Warp a function sequence """
    def fn(x, **kwarg):
        for func in funcs:
            x = func(x, **kwarg)
        return x
    return fn

def join_warper(fsize):
    def fn(x):
        x = join_frame(x, fsize)
        return x
    return fn

def split_warper(fsize):
    def fn(x):
        x = split_frame(x, fsize)
        return x
    return fn
