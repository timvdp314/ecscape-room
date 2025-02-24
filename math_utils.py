import numpy as np

def tuple_addition(t1: tuple[any], t2: tuple[any]):
    res = tuple(int(x) for x in (np.array(t1) + np.array(t2)))

    return res

def np_to_tuple(t: np.array):
    return tuple(int(x) for x in t)