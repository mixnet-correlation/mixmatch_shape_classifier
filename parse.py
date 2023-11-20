import sys
import time
import fileinput

import numpy as np
import argparse as ap

from pathlib import Path

from const import *
from common import *


def load_to_array(csvpath, num_pairs=35000, scale=1, dtype=np.float64):
    '''Return an array from a file by padding to max capture length.'''
    pairs = []
    max_length = 0
    for i, line in enumerate(open(csvpath)):
        if i >= num_pairs:
            break
        times = np.array(line.strip().split(), dtype=dtype)
        length = len(times)
        if length > max_length:
            max_length = length
        pairs.append(times)
    arr = np.array([pad(pair, max_length, np.nan) for pair in pairs])
    return arr


def concatenate_parts(datapath, dataset, dtype='data'):
    to_concatenate = [p.stem for p in datapath.glob(f'{dataset}_*_{dtype}.*')]
    for f in np.unique(to_concatenate):
        # check if parts have already been concatenated
        if (datapath / f).is_file():
            continue
        
        # concatenate and print to new file
        with open(datapath / f, 'w') as fout:
            for line in fileinput.input(sorted(datapath.glob(f'{f}.*'))):
                fout.write(line)
            fileinput.close()
            

def create_ack_symlinks(datapath, dataset):
    '''
    For acks:
       - ack initiator_to-gateway = data initiator_from-gateway
       - ack responder_to-gateway = data responder_from-gateway
       
    We create symlinks to save space.
    '''
    for origin in ORIGINS:
        for fpath in datapath.glob(f'{dataset}_{origin}_to_gateway_data*'):
            fname = fpath.name
            new_fname = fname.replace('data', 'ack')
            fname = fname.replace('to_gateway', 'from_gateway')
            # check if symlink exists
            if (datapath / new_fname).is_symlink():
                break
            abspath = (datapath / fname).resolve()
            (datapath / new_fname).symlink_to(abspath)


def ensure_outpath(outpath):
    if outpath is None:
        datadir = ROOTPATH / 'data'
        outpath = (datadir / time.strftime("%Y-%m-%d_%H-%M-%S"))
        outpath.mkdir(parents=True, exist_ok=True)
        update_symlink(outpath, datadir / 'latest')
    return Path(outpath)


def parse_all(datapath, outpath=None, exp_num=None, factor=1):
    # make paths
    datapath = Path(datapath) 
    outpath = ensure_outpath(outpath)
    with open(outpath / 'experiment.info', 'w') as f:
        f.write(f'{exp_num}\n{factor}')
    
    for ds in ['train', 'val', 'test']:
        # concatenate parts
        concatenate_parts(datapath, ds, dtype='data')
        concatenate_parts(datapath, ds, dtype='ack')
        
        # create ack symlinks
        create_ack_symlinks(datapath, ds)
        
        num_pairs = PAIRS[ds] // factor
        
        merged = {}
        max_len = 0
        for o in ORIGINS:
            for d in DIRECTIONS:
                key = f'{ds}_{o}_{d}'
                data = load_to_array(datapath / f'{key}_data', num_pairs)
                acks = load_to_array(datapath / f'{key}_ack', num_pairs)
                
                # pad to the same size
                if acks.shape[1] < data.shape[1]:
                    acks = pad(acks, data.shape[1], np.nan, axis=1)
                    
                elif data.shape[1] < acks.shape[1]:
                    data = pad(data, acks.shape[1], np.nan, axis=1)
                    
                if max_len < data.shape[1]:
                    max_len = data.shape[1]
                
                merged[f'{o}_{d}'] = stack(data, acks)
                assert np.allclose(merged[f'{o}_{d}'][..., 0], data, equal_nan=True)
                
        for k, v in merged.items():
            merged[k] = data = pad(v, max_len, np.nan, axis=1)
            print(f'{ds}, {k}, {merged[k].shape=}')
        
        # copy chunk of delay matrix
        delays = np.load(datapath / f'{ds}_delay_matrix.npz')[f'delay_matrix_{ds}']
        new_delays = {}
        new_delays[f'{ds}_delay_matrix'] = delays[:num_pairs, :num_pairs, :]
        np.savez_compressed(outpath / f'{ds}_delay_matrix.npz', **new_delays)
        np.savez_compressed(f'{outpath / ds}.npz', **merged)


def main():
    parser = config_parser()
    args   = parser.parse_args()
    parse_all(args.datapath, 
              outpath=args.outpath,
              exp_num=str(args.experiment),
              factor=args.factor)


def config_parser():
    parser = ap.ArgumentParser('Parsing csv files and splitting into windows.')
    parser.add_argument('datapath',
                        type=str,
                        help='path to the csv files.')
    parser.add_argument('--experiment',
                        type=int,
                        default=1,
                        choices=EXPERIMENTS,
                        help='Experiment number.')
    parser.add_argument('--outpath',
                        default=None,
                        help='Path where the windowed traces are dumped.')
    parser.add_argument('--factor',
                        type=int,
                        default=1,
                        help='Fraction of data.')
    return parser


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
