import sys
import time
import shutil
import argparse
import traceback

import multiprocessing as mp

from pathlib import Path
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials

import logging
import logging.config

from const import *
from common import *
from train import main as train


# configure paths
TUNEDIR = ROOTPATH / 'tuning'/ time.strftime("%Y-%m-%d_%H-%M-%S")
TUNEDIR.mkdir(parents=True, exist_ok=True)

# configure logging
logging.config.fileConfig("./logging.conf")
logger = logging.getLogger("tune")


def train_and_evaluate(args):
    try:  
        # parse dataset and windowize
        datapath = Path(args['datapath']).resolve()
        tunedir = TUNEDIR / time.strftime("%Y-%m-%d_%H-%M-%S")
        tunedir.mkdir(parents=True, exist_ok=True)
        args['respath'] = str(tunedir)
        
        logger.info(f"Starting a new evaluation {datapath}: {args}")
        logger.info("Running training in a subprocess...")
        p = mp.Process(target=train, kwargs=args)
        p.start()

        timeout = 7200
        start = time.time()
        while time.time() - start <= timeout:
            if p.is_alive():
                time.sleep(30)  
            else:
                break
        else:
            logger.error("timed out, killing process")
            p.terminate()
        p.join()
        
        # get last loss
        logger.debug("Training finished... collecting loss results")
        best_val_loss = get_best_loss(tunedir / 'train.log')
        logger.info(f"Finished evaluation with val loss: {best_val_loss}")
        return {'loss': best_val_loss, 'status': STATUS_OK}
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Exception: {e=}")
        return {'loss': 1000, 'status': STATUS_FAIL}


def main():
    # Configure parser
    p = argparse.ArgumentParser('Hyperparameter tuning.')
    p.add_argument('datapath',
                   type=str,
                   help='Path to data directory.')
    args = p.parse_args()
    
    # create dir structure
    space = {
            'datapath':         args.datapath,
            'num_epochs':       20,
            'patience':         5,
            'start_from_epoch': 10,
            'min_delta':        0.0001,
            'window_shift':     hp.choice('window_shift', [0.25, 0.75, 1.0]),
            'window_size':      200,
            'batch_size' :      15,
            'learning_rate':    0.001,
            'alpha':            0.1       
            }

    trials = Trials()
    best = fmin(train_and_evaluate, space, algo=tpe.suggest, max_evals=50, trials=trials)
    logger.info(f'best: {best}')
    shutil.move('debug_tune.log', TUNEDIR / 'debug.log')
    

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
            
