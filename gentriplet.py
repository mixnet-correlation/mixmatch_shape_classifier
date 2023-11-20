import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.experimental.numpy import moveaxis

import logging
logger = logging.getLogger('train')
from common import *
from parse import *
from alignment import *


class TripletGenerator():
    def __init__(self, shape_model, drift_model,
            alpha=0.1, window_shift=1.0, window_size=100, batch_size=32):
        self.shape_model    = shape_model
        self.drift_model    = drift_model
        self.alpha          = alpha
        self.window_size    = window_size
        self.window_shift   = window_shift
        self.batch_size     = batch_size
        
        self.nwins          = None
        self.init_flows     = None
        self.resp_flows     = None
    
    def get_windows(self, to_pairs, from_pairs, delays):
        # get shapes
        self.init_flows = delays.shape[0]
        self.resp_flows = delays.shape[1]

        # pariwise combinations between initiator and responder
        anc_pairs = concatenate_pairwise(*to_pairs)
        pns_pairs = concatenate_pairwise(*from_pairs)
        
        # alignment
        anc_pairs, pns_pairs = align_pairs(anc_pairs, pns_pairs, delays)

        # windowize
        anc_wins = windowize(anc_pairs, self.window_size, self.window_shift)
        pns_wins = windowize(pns_pairs, self.window_size, self.window_shift)
        
        # discard empty windows across axes
        anc_wins, pns_wins = discard_along_axes(anc_wins, pns_wins)
        
        # get num flows and num windows
        self.pairs = anc_wins.shape[0]
        self.nwins = anc_wins.shape[1]
        
        assert self.init_flows * self.resp_flows == self.pairs
        
        # flatten the windows dimension
        anc_wins = flatten_windows(anc_wins)
        pns_wins = flatten_windows(pns_wins)            
        
        return anc_wins, pns_wins
    
    def get_tensors(self, anc_wins, pns_wins, channel=0):
        # Calculate similarities
        sims_data = self.calculate_similarities(anc_wins[..., channel],
                                                pns_wins[..., channel])

        # Mine negatives
        data_pos_idx, data_neg_idx = self.calculate_negatives(sims_data)

        # Get the samples by index and channel
        aps = anc_wins[data_pos_idx, ..., channel]
        ps  = pns_wins[data_pos_idx, ..., channel]
        
        ans = anc_wins[data_neg_idx, ..., channel]
        ns  = pns_wins[data_neg_idx, ..., channel]

        return (convert_to_tensor(aps), convert_to_tensor(ps),
                    convert_to_tensor(ans), convert_to_tensor(ns))
    
    def calculate_drift_scores(self, anc_wins, pns_wins, add_axis=True, roll=True):
        # convert to tensor
        anc_tens = convert_to_tensor(anc_wins, add_axis=add_axis)
        pns_tens = convert_to_tensor(pns_wins, add_axis=add_axis)
        
        # calculate score
        scores = self.drift_model((anc_tens, pns_tens))
        scores = tf.reshape(scores, shape=(self.init_flows, self.resp_flows, self.nwins))
        
        # roll axis for a more intuitive order:
        #   initiator-responder-windows
        if roll:
            scores = moveaxis(scores, 2, 0)
        
        return scores

    def calculate_similarities(self, anc_wins, pns_wins, add_axis=True, roll=True):
        # convert to tensor
        anc_tens = convert_to_tensor(anc_wins, add_axis=add_axis)
        pns_tens = convert_to_tensor(pns_wins, add_axis=add_axis)
        
        # calculate similarities
        anc_emb, pns_emb = self.shape_model(anc_tens), self.shape_model(pns_tens)
        sims = cosine_similarity(anc_emb, pns_emb)
        sims = tf.reshape(sims, shape=(self.init_flows, self.resp_flows, self.nwins))
        
        # roll axis for a more intuitive order:
        #   initiator-responder-windows
        if roll:
            sims = moveaxis(sims, 2, 0)
        
        return sims
    
    def calculate_negatives(self, sims=None):
        # TODO: can be parallelized
        pos_all, neg_all = [], []
        for w in range(self.nwins):
            # initiator k to all responder similarities
            for k, row in enumerate(sims[w, :, :]):
                # take indices of examples that satisfy the semi-hard condition
                alpha = tf.constant(self.alpha, dtype=tf.float32)
                neg_idxs = tf.where(row + alpha > row[k])
                neg_idxs = tf.reshape(tf.cast(neg_idxs, dtype=np.int32), shape=[-1])

                # fix indices
                neg_idxs = neg_idxs[neg_idxs != k]
                pos_idxs = k * np.ones(len(neg_idxs), dtype=np.int32)

                # calculate index in window array
                offset = k * self.resp_flows * self.nwins
                neg_all += list(offset + ((neg_idxs * self.nwins) + w))
                pos_all += list(offset + ((pos_idxs * self.nwins) + w))

        return np.array(pos_all, dtype=np.int32), np.array(neg_all, dtype=np.int32)    
