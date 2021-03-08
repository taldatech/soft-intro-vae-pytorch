import math
import numpy as np
import os
import pandas as pd
import pickle

from decimal import Decimal
from itertools import accumulate, tee, chain
from typing import List, Tuple, Dict, Optional, Any, Set

from utils.plyfile import load_ply

READERS = {
    '.ply': load_ply,
    '.np': lambda file_path: pickle.load(open(file_path, 'rb')),
}


def load_file(file_path):
    _, ext = os.path.splitext(file_path)
    return READERS[ext](file_path)


def add_float(a, b):
    return float(Decimal(str(a)) + Decimal(str(b)))


def ranges(values: List[float]) -> List[Tuple[float]]:
    lower, upper = tee(accumulate(values, add_float))
    lower = chain([0], lower)

    return zip(lower, upper)


def make_slices(values: List[float], N: int):
    slices = [slice(int(N * s), int(N * e)) for s, e in ranges(values)]
    return slices


def make_splits(
        data: pd.DataFrame,
        splits: Dict[str, float],
        seed: Optional[int] = None):

    # assert correctness
    if not math.isclose(sum(splits.values()), 1.0):
        values = " ".join([f"{k} : {v}" for k, v in splits.items()])
        raise ValueError(f"{values} should sum up to 1")

    # shuffle with random seed
    data = data.iloc[np.random.permutation(len(data))]
    slices = make_slices(list(splits.values()), len(data))

    return {
            name: data[idxs].reset_index(drop=True) for name, idxs in zip(splits.keys(), slices)
        }


def sample_other_than(black_list:  Set[int], x: np.ndarray) -> int:
    res = np.random.randint(0, len(x))
    while res in black_list:
        res = np.random.randint(0, len(x))

    return res


def clip_cloud(p: np.ndarray) -> np.ndarray:
    # create list of extreme points
    black_list = set(np.hstack([
        np.argmax(p, axis=0), np.argmin(p, axis=0)
    ]))

    # swap any other point
    for idx in black_list:
        p[idx] = p[sample_other_than(black_list, p)]

    return p


def find_extrema(xs, n_cols: int=3, clip: bool=True) -> Dict[Any, List[float]]:
    from collections import defaultdict

    mins = defaultdict(lambda: [np.inf for _ in range(n_cols)])
    maxs = defaultdict(lambda: [-np.inf for _ in range(n_cols)])

    for x, c in xs:
        x = clip_cloud(x) if clip else x
        mins[c] = [min(old, new) for old, new in zip(mins[c], np.min(x, axis=0))]
        maxs[c] = [max(old, new) for old, new in zip(maxs[c], np.max(x, axis=0))]

    return mins, maxs


def merge_dicts(
        dict_old: Dict[Any, List[float]],
        dict_new: Dict[Any, List[float]], op=min) -> Dict[Any, List[float]]:
    '''
    Simply takes values on List of floats for given key
    '''
    d_out = {** dict_old}
    for k, v in dict_new.items():
        if k in dict_old:
            d_out[k] = [op(new, old) for old, new in zip(dict_new[k], dict_old[k])]
        else:
            d_out[k] = dict_new[k]

    return d_out


def save_extrema(clazz, root_dir, splits=('train', 'test', 'valid')):
    '''
    Maybe this should be class dependent normalization?
    '''
    min_dict, max_dict = {}, {}
    for split in splits:
        data = clazz(root_dir=root_dir, split=split, remap=False)
        mins, maxs = find_extrema(data)
        min_dict = merge_dicts(min_dict, mins, min)
        max_dict = merge_dicts(max_dict, maxs, max)

    # vectorzie values
    for d in (min_dict, max_dict):
        for k in d:
            d[k] = np.array(d[k])

    with open(os.path.join(root_dir, 'extrema.np'), 'wb') as f:
        pickle.dump((min_dict, max_dict), f)


def remap(old_value: np.ndarray,
          old_min: np.ndarray, old_max: np.ndarray,
          new_min: float = -0.5, new_max: float = 0.5) -> np.ndarray:
    '''
    Remap reange
    '''
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value
