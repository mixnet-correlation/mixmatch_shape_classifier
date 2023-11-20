import numpy as np

from common import *


def delay_sequences(s1, s2, delta=0):
    l1, l2 = s1.shape[0], s2.shape[0]
    if delta >= 0:
        if l1 <= l2:
            if l1 + delta > l2:
                delayed_s1 = s1[:l2 - delta]
                delayed_s2 = s2[delta:]
            else:
                delayed_s1 = s1
                delayed_s2 = s2[delta:l1 + delta]
        else:
            delayed_s1 = s1[:l2 - delta]
            delayed_s2 = s2[delta:]
    else:
        if l1 > l2:
            if l1 + delta <= l2:
                delayed_s1 = s1[-delta:]
                delayed_s2 = s2[:l1 + delta]
            else:
                delayed_s1 = s1[-delta:l2 - delta]
                delayed_s2 = s2
        else:
            delayed_s1 = s1[-delta:]
            delayed_s2 = s2[:l1 + delta]

    return delayed_s1, delayed_s2


def align_pairs_channel(S1, S2, delays):
    m = delays.shape[1]
    #print(f'{n=}, {m=}, {S1.shape=}, {S2.shape=}, {delays.shape=}')
    D1, D2 = np.full(S1.shape, np.nan), np.full(S2.shape, np.nan)
    for i, (s1, s2) in enumerate(zip(S1, S2)):
        d1, d2 = delay_sequences(s1[~np.isnan(s1)],
                                 s2[~np.isnan(s2)],
                                 delays[i // m, i % m])
        D1[i, :len(d1)] = d1
        D2[i, :len(d2)] = d2
    return D1, D2


def align_pairs(S1, S2, delays):
    D1_data, D2_data = align_pairs_channel(S1[..., 0],
                                           S2[..., 0],
                                           delays[..., 0])
    D1_acks, D2_acks = align_pairs_channel(S1[..., 1],
                                           S2[..., 1],
                                           delays[..., 1])
    return stack(D1_data, D1_acks), stack(D2_data, D2_acks)
