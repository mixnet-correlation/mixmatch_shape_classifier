import os

import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.math import multiply, log, reduce_sum 
from tensorflow.keras.backend import epsilon

import numpy as np
from numpy.lib.stride_tricks import as_strided

from pathlib import Path
from multiprocessing import shared_memory


def windowize(arr, wsize, addn, axis=1):
    step = int(wsize * addn)
    return rolling_window(arr, wsize, step, axis=1)


def cutoff_windows(arr, index=-1, data_channel=0):
    # take indices where the first element is a nan (empty window)
    # (data should not be empty, regardless of acks)
    # if index=-1, cuts off after the last window with no nans
    # if index=0, cuts off after the first with nans (at the end)
    empty = np.isnan(arr[:, :, -1, ...]).any(axis=0)
    empty = empty[:, data_channel, ...]
    if len(empty.shape) > 1:
        empty = empty.any(axis=1)
    return arr[:, ~empty, ...]

def unstack(arr, axis=-1):
    return np.moveaxis(arr, axis, 0)


def stack(arr1, arr2, axis=-1):
    return np.stack((arr1, arr2), axis=axis)


def discard_along_axes(anc, pns):
    return unstack(cutoff_windows(stack(anc, pns)))


def flatten_windows(arr):
    new_shape = (arr.shape[0] * arr.shape[1],) + arr.shape[2:]
    return arr.reshape(new_shape)


def window_view(arr, num_wins):
    new_shape = (arr.shape[0] // num_wins, num_wins,) + arr.shape[1:]
    return arr.reshape(new_shape)


def repeat(arr, size=None):
    if size is None or size == 1:
        return arr
    rep = np.repeat(arr[:, None, ...], repeats=size, axis=1)
    new_shape = (rep.shape[0] * rep.shape[1],) + arr.shape[1:]
    return rep.reshape(new_shape)


def tile(arr, size=None):
    if size is None or size == 1:
        return arr
    repeats = (size,) + (1,) * arr.ndim
    til = np.tile(arr[None, ...], repeats)
    new_shape = (til.shape[0] * til.shape[1],) + arr.shape[1:]
    return til.reshape(new_shape)


def concatenate_pairwise(init, resp):
    return join(repeat(init, resp.shape[0]),
                  tile(resp, init.shape[0]))


def rolling_window(arr, wsize, step, axis=1):
    # Note: skips the last window if size of array is not
    # a multiple of the size of the window.
    assert step > 0
    shape = list(arr.shape)
    overlap = arr.shape[axis]
    if step < wsize:
        overlap += 1 - wsize
    shape[axis] = overlap // step
    shape.insert(axis + 1, wsize)
    
    strides = list(arr.strides)
    strides[axis] = arr.strides[axis] * step
    strides.insert(axis + 1, arr.strides[axis])

    return as_strided(arr, shape, strides)


def pad(array: np.ndarray, target_length: int, value: float = 0, axis: int = 0) -> np.ndarray:
    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=(value,))


def diff(arr, axis=1):
    #np.diff(np.insert(arr, 0, 0.0, axis=axis), axis=axis)
    return np.diff(arr, axis=axis)

def join(a, b, axis=1):
    return np.sort(np.concatenate((a, b), axis=axis), axis=axis, kind='mergesort')


def get_pos_indices(nwins, nflows):
    iflow = np.arange(nflows)
    index = iflow * nwins
    index = np.repeat((1 + nflows) * index, nwins)
    offset = np.tile(np.arange(nwins), nflows)
    return index + offset


def convert_to_tensor(v, add_axis=True):
    if add_axis:
        v = v[..., None]    
    return tf.convert_to_tensor(v)


def get_best_loss(f):
    best_loss = 1.0
    for line in open(f):
        if "Best" in line:
            loss = float(line.strip().split()[-1])
            if loss < best_loss:
                best_loss = loss
    return best_loss


# LOSS
################################################################
def cosine_similarity(u, v):
    assert u.ndim == 2
    return tf.math.reduce_sum(tf.math.multiply(u,v), axis=1) / (tf.math.sqrt(tf.math.reduce_sum(tf.math.multiply(u, u), axis=1)) * tf.math.sqrt(tf.math.reduce_sum(tf.math.multiply(v, v), axis=1)))


def triplet_loss(ap, p, an, n, margin):
    return tf.minimum(tf.maximum( margin + cosine_similarity(an, n) - cosine_similarity(ap, p), 0), margin)


def cross_entropy(y_pred, y_true):
    y_pred, y_true = tf.reshape(y_pred, [-1]), tf.reshape(y_true, [-1])
    pos = multiply(log(y_pred + epsilon()), y_true) / reduce_sum(y_true)
    neg = multiply((1. - y_true), log(1. - y_pred + epsilon())) / reduce_sum(1. - y_true)
    return -reduce_sum(pos + neg)

# OTHERS:
################################################################

def latest_paths(path, n=1):
    return sorted(Path(path).glob('*/'), key=os.path.getmtime)[-n]
    

def update_symlink(frompath, topath):
    (topath).unlink(missing_ok=True)
    (topath).symlink_to(frompath)    


def latest_symlink(path):
    update_symlink(latest_paths(path), path / 'latest')


def create_shared_memory(X, name, dtype):
    size = np.dtype(dtype).itemsize * X.shape[0] * X.shape[1]
    shm = shared_memory.SharedMemory(create=True, size=size, name=name)
    arr = np.ndarray(X.shape, dtype=dtype, buffer=shm.buf)
    arr[:] = X[:]
    return shm


def close_shared_memory(shm):
    shm.close()
    shm.unlink()
