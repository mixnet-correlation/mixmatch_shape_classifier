import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import argparse
import random

from tqdm import tqdm
from pathlib import Path
from functools import partial

from scipy import sparse

import numpy as np
import tensorflow as tf

from keras.models import load_model

from sklearn.metrics import roc_curve

from model import *
from const import *
from common import *
from dataman import DataManager
from alignment import *
from gentriplet import *
from multiprocessing import Pool, cpu_count, shared_memory

n_cpus = cpu_count()  


# Remove all nondeterminism
seed = int(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Limit GPU memorgy growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)



def config_parser():    
    p = argparse.ArgumentParser('Evaluate on test set.')
    p.add_argument('datapath',
                   type=str,
                   help='path to data.')
    p.add_argument('respath',
                   type=str,
                   help='path to results.')
    p.add_argument('--window-shift',
                    type=float,
                    default=1.0,
                    help='Window shift percentage.')
    p.add_argument('--window-size',
                    type=int,
                    default=256,
                    help='Window size.')
    p.add_argument('--batch-size',
                    type=int,
                    default=15,
                    help='Batch size.')
    return p


def main(datapath='./data/latest/', respath='./results/latest/',
         window_shift=1.0, window_size=256, batch_size=15):
    # Parse arguments
    datapath    = Path(datapath)
    respath     = Path(respath)

    thresholds = np.linspace(0., 0.999, 100)

    # Load datasets
    respath = Path(respath)
    dm = DataManager(datapath)
    logger.info("Loading test data...")
    test = dm.load('test')

    # Load model
    shape_model = load_model(respath / 'shape_model.tf', compile=False)
    drift_model = load_model(respath / 'drift_model.tf', compile=False)
    
    # replace results file
    results_file = respath / 'step_results.csv'
    if results_file.exists():
        results_file.unlink(missing_ok=True)
    
    tg = TripletGenerator(shape_model, drift_model,
            window_shift=window_shift, window_size=window_size, batch_size=batch_size)
    batch_count = test.init_size // batch_size
    end = batch_count * batch_size
    for start in (pbar := tqdm(range(0, end, batch_size), total=batch_count)):
        stop = start + batch_size
        
        chunk = test.chunk(start, stop, 0, test.resp_size)
        
        # Get windows
        # TODO: use rolling window with stride=1
        with tf.device(f'/cpu:0'):
            anc_wins, pns_wins = tg.get_windows(chunk.to_gateway,
                                                chunk.from_gateway,
                                                chunk.delays)

            # shape scores
            sims_data = tg.calculate_similarities(anc_wins[..., 0], pns_wins[..., 0]).numpy()
            sims_acks = tg.calculate_similarities(anc_wins[..., 1], pns_wins[..., 1]).numpy()        

            # drift scores
            sigs_data = 2 * (1 - tg.calculate_drift_scores(anc_wins[..., 0], pns_wins[..., 0]).numpy()) - 1
            sigs_acks = 2 * (1 - tg.calculate_drift_scores(anc_wins[..., 1], pns_wins[..., 1]).numpy()) - 1

            # Apply thresholds
            assert tg.init_flows == batch_size
            assert tg.resp_flows == test.resp_size
            y_true  = np.eye(tg.init_flows, test.resp_size, k=start, dtype=np.int8)
            for i, th in enumerate(thresholds):
                for win_idx in range(1, tg.nwins + 1):
                    sims_data_w = sims_data[:win_idx].mean(axis=0)
                    sigs_data_w = sigs_data[:win_idx].mean(axis=0)
                    score_data  = (sims_data_w + sigs_data_w) / 2

                    sims_acks_w = sims_acks[:win_idx].mean(axis=0)
                    sigs_acks_w = sigs_acks[:win_idx].mean(axis=0)
                    score_acks  = (sims_acks_w + sigs_acks_w) / 2

                    y_pred_data = (score_data >= th).astype(int)
                    y_pred_acks = (score_acks >= th).astype(int)

                    y_pred = (((score_data + score_acks) / 2) >= th).astype(int)
                    
                    res_data = calculate_metrics(y_pred_data, y_true)
                    res_acks = calculate_metrics(y_pred_acks, y_true)
                    res_both = calculate_metrics(y_pred, y_true)

                    # calculate metrics for shape only
                    shape_score = (sims_data_w + sims_acks_w) / 2
                    res_shape = calculate_metrics((shape_score > th).astype(int), y_true)

                    # calculate metrics for drift only
                    drift_score = (sigs_data_w + sigs_acks_w) / 2
                    res_drift = calculate_metrics((drift_score > th).astype(int), y_true)

                    TPR = float(res_data[0]) / float(tg.init_flows)
                    FPR = float(res_data[1]) / float(sims_data_w.size - tg.init_flows)

                    if win_idx == tg.nwins and i == batch_size:
                        TPR = float(res_data[0]) / float(tg.init_flows)
                        FPR = float(res_data[1]) / float(sims_data_w.size - tg.init_flows)
                        msg = (f'{TPR=:.6f} ({res_data[0]} / {tg.init_flows}), ' \
                               f'{FPR=:.6f} ({res_data[1]} / {sims_data_w.size - tg.init_flows}) ')
                        pbar.set_description(msg)

                    # write results to file
                    with open(results_file, 'a') as fo:
                        print(f'{win_idx},{th},sum,data,{res_data[0]},{res_data[1]}', file=fo)
                        print(f'{win_idx},{th},sum,acks,{res_acks[0]},{res_acks[1]}', file=fo)
                        print(f'{win_idx},{th},sum,both,{res_both[0]},{res_both[1]}', file=fo)
                        print(f'{win_idx},{th},shape,both,{res_shape[0]},{res_shape[1]}', file=fo)
                        print(f'{win_idx},{th},drift,both,{res_drift[0]},{res_drift[1]}', file=fo)
                        fo.flush()


def calculate_metrics(y_pred, y_true):
    tps = ((y_true == 1) & (y_pred == 1)).astype(int).sum()
    fps = ((y_true == 0) & (y_pred == 1)).astype(int).sum()
    return tps, fps


if __name__ == '__main__':
    try:
        p = config_parser()
        args = p.parse_args() 
        sys.exit(main(**vars(args)))
    except KeyboardInterrupt:
        sys.exit(1)
