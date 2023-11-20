from pathlib import Path
import numpy as np

from const import *


class DataManager():
    def __init__(self, datapath):
        self.datapath = Path(datapath)
        
    def load(self, dataset):
        data   = np.load(self.datapath / f'{dataset}.npz')
        delays = np.load(self.datapath / f"{dataset}_delay_matrix.npz")
        delays = delays[f'{dataset}_delay_matrix']
        return Dataset(data, delays)

               
class Dataset():
    def __init__(self, data, delays):
        self.data = {}
        for o in ORIGINS:
            for d in DIRECTIONS:
                self.data[f'{o}_{d}'] = data[f'{o}_{d}']
        self.init_size = data['initiator_to_gateway'].shape[0]
        self.resp_size = data['responder_to_gateway'].shape[0]
        self.delays    = delays

    @property
    def initiator(self):
        return (self.data[f'initiator_{d}'] for origin in ORIGINS)
    
    @property
    def responder(self):
        return (self.data[f'responder_{d}'] for d in DIRECTIONS)
    
    @property
    def to_gateway(self):
        return (self.data[f'{o}_to_gateway'] for o in ORIGINS)
    
    @property
    def from_gateway(self):
        return (self.data[f'{o}_from_gateway'] for o in ORIGINS)
        
    def shuffle(self):
        perm = np.arange(self.init_size)
        np.random.shuffle(perm)
        for o in ORIGINS:
            for d in DIRECTIONS:
                self.data[f'{o}_{d}'] = self.data[f'{o}_{d}'][perm]
        self.delays = self.delays[perm, :][:, perm]        
    
    def chunk(self, init_start, init_stop, resp_start=None, resp_stop=None):
        if resp_start == None:
            resp_start = init_start
        if resp_stop == None:
            resp_stop = init_stop
        chunk_data = {} 
        for d in DIRECTIONS:
            chunk_data[f'initiator_{d}'] = self.data[f'initiator_{d}'][init_start:init_stop]
            chunk_data[f'responder_{d}'] = self.data[f'responder_{d}'][resp_start:resp_stop]
        chunk_delays = self.delays[init_start:init_stop, resp_start:resp_stop, :]
        return Dataset(chunk_data, chunk_delays)
        
