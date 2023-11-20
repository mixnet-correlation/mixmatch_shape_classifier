import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import traceback
import sys
import shutil
import random
import argparse
import logging
import logging.config

import tensorflow.keras.backend as kb

from tqdm import tqdm
from pathlib import Path

from tensorflow.keras import optimizers

from const import *
from model import *
from common import *
from dataman import DataManager
from alignment import *
from gentriplet import *

logging.config.fileConfig("./logging.conf")
logger = logging.getLogger('train')

log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)
fh = logging.FileHandler('tensorflow.log')
fh.setLevel(logging.DEBUG)
log.addHandler(fh)

# Remove all nondeterminism
seed = int(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Limit GPU memorgy growth
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class ModelTrainer():
    def __init__(self, shape_model, drift_model,
            triplet_generator, batch_size=15, learning_rate=0.001):
        self.shape_model = shape_model
        self.drift_model = drift_model
        self.tg = triplet_generator
        self.batch_size = batch_size
        self.opt = optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        
    def train_batch(self, data, backward=True):
        batch_count = data.init_size // self.batch_size
        final_count = batch_count * self.batch_size    
        batch_iterator = enumerate(range(0, final_count, self.batch_size))
        logger.debug(f"Batch size: {self.batch_size}; Total number of batches: {batch_count}")
        for batch_index, start in (pbar := tqdm(batch_iterator, total=batch_count)):
            for channel in CHANNELS:
                end = start + self.batch_size
                
                # Select batch samples
                chunk = data.chunk(start, end)
                
                # Get windows
                anc_wins, pns_wins = self.tg.get_windows(chunk.to_gateway, chunk.from_gateway, chunk.delays) 
                
                with tf.device(f'/{DEVICE}:0'):
                    # Get samples
                    sample = self.tg.get_tensors(anc_wins, pns_wins, channel=channel)
                del chunk
                
                num_triplets = sample[-1].shape[0]
                pos_neg = self.tg.nwins * self.batch_size * (self.batch_size - 1) 
                sh_perc = num_triplets / pos_neg 
                
                if num_triplets == 0:
                    logger.debug("No SH negatives found, skipping the batch...")
                    continue
    
                # Perform forward pass of both payload and ack traffic flows
                with tf.GradientTape() as st, tf.GradientTape() as dt, tf.GradientTape() as ft:
                    aps, ps, ans, ns = sample
                    inflows = tf.concat((aps, ans), axis=0)
                    outflows = tf.concat((ps, ns), axis=0)
                    labels = tf.concat((tf.zeros(aps.shape[0]), tf.ones(ans.shape[0])), axis=0)

                    # feed to shape model and get loss
                    shape_embs = self.shape_model(sample)
                    shape_pred = convert_to_tensor(cosine_similarity(self.shape_model(inflows),
                                                                     self.shape_model(outflows)))
                    shape_loss = tf.math.reduce_sum(triplet_loss(*shape_embs, self.tg.alpha)) / num_triplets
                    del shape_embs                 
                    
                    # feed to drift model and get loss
                    # we feed the sample to the drift model as there there is no reason for why the drift
                    # of semi-hard negatives should be more or less informative about the drift. In addition,
                    # because this is done per-window, we give examples of all window positions to the drift
                    # model.
                    drift_pred = self.drift_model((inflows, outflows))
                    drift_loss = cross_entropy(labels, drift_pred)
                    del sample

                    yield shape_loss, drift_loss
                    msg = (f"Channel {channel}, "
                           f"shape loss: {shape_loss:.6f}, "
                           f"drift loss: {drift_loss:.6f}, "
                           f"SH perc: {100 * sh_perc:.2f}% ({num_triplets}/{pos_neg})")
                    logger.debug(msg)
                    pbar.set_description(msg)                      

                    if backward:
                        if shape_loss > 0:
                            shape_grads = st.gradient(shape_loss, self.shape_model.trainable_weights)
                            self.opt.apply_gradients(zip(shape_grads, self.shape_model.trainable_weights))

                        if drift_loss > 0:
                            drift_grads = dt.gradient(drift_loss, self.drift_model.trainable_weights)
                            self.opt.apply_gradients(zip(drift_grads, self.drift_model.trainable_weights))


def train(datapath='./data/latest/', respath='./results/latest',
          alpha=0.1, window_shift=1.0, window_size=256,
          learning_rate=1e-3, batch_size=15, num_epochs=100,
          start_from_epoch=20, patience=10, min_delta=1e-5):

    # Load datasets
    respath = Path(respath)
    dm = DataManager(datapath)
    
    logger.info("Loading datasets...")
    train = dm.load('train')
    val   = dm.load('val')
    
    # Instantiate model
    drift_model = DriftModel()
    shape_model = DeepCoffeaModel()
    tg          = TripletGenerator(shape_model, drift_model, alpha, window_shift, window_size, batch_size)
    mt          = ModelTrainer(shape_model, drift_model, tg, batch_size, learning_rate)
    
    patience_local = patience
    for epoch in range(num_epochs):
        tr_best_shape_loss, vl_best_shape_loss = 1.0, 1.0
        tr_best_drift_loss, vl_best_drift_loss = 1.0, 1.0

        start_time = time.time()
        logger.info(f"Epoch number {epoch}")

        # training
        train.shuffle()
        logger.info(f"Training...")
        for shape_tr_loss, drift_tr_loss  in mt.train_batch(train):
            if shape_tr_loss < tr_best_shape_loss:
                tr_best_shape_loss = shape_tr_loss
            if drift_tr_loss < tr_best_drift_loss:
                tr_best_drift_loss = drift_tr_loss
                
        # validation
        val.shuffle()
        logger.info(f"Validating...")
        for shape_vl_loss, drift_vl_loss in mt.train_batch(val, backward=False):
            if shape_vl_loss < vl_best_shape_loss:
                vl_best_shape_loss = shape_vl_loss        
                logger.debug(f'Saving shape model weights to {respath}')
                mt.shape_model.save(respath / 'shape_model.tf', save_format='tf')

            if drift_vl_loss < vl_best_drift_loss:
                vl_best_drift_loss = drift_vl_loss        
                logger.debug(f'Saving drift model weights to {respath}')
                mt.drift_model.save(respath / 'drift_model.tf', save_format='tf')

            # Check early stopping
            if np.abs(vl_best_shape_loss - shape_vl_loss) < min_delta: 
                if epoch > start_from_epoch:
                    patience_local -= 1
                    if patience_local == 0:
                        logger.info(f"Val loss did not improve for {patience} epochs: stop!")
                        return 0
            else:
                patience_local = patience
                
        logger.info(f"Best training shape loss: {tr_best_shape_loss:.6f}, "
                    f"Best valdation shape loss: {vl_best_shape_loss:.6f}")
        logger.info(f"Best training drift loss: {tr_best_drift_loss:.6f}, "
                    f"Best valdation drift loss: {vl_best_drift_loss:.6f}")
        logger.info("Time taken: %.2fs" % (time.time() - start_time))

    return 0


def config_parser():
    p = argparse.ArgumentParser('Train embeddings model.')
    p.add_argument('--datapath',
                    type=str,
                    default='./data/latest/',
                    help='path to input data.')
    p.add_argument('--respath',
                    type=str,
                    help='path to results.')
    p.add_argument('--alpha',
                    type=float,
                    default=0.1,
                    help='Loss margin.')
    p.add_argument('--window-shift',
                    type=float,
                    default=1.0,
                    help='Window shift percentage.')
    p.add_argument('--window-size',
                    type=int,
                    default=256,
                    help='Window size.')
    p.add_argument('--learning-rate',
                    type=float,
                    default=1e-3,
                    help='Learning rate.')
    p.add_argument('--batch-size',
                    type=int,
                    default=15,
                    help='Batch size.')
    p.add_argument('--num-epochs',
                    type=int,
                    default=100,
                    help='Number of epochs.')
    return p


def main(*args, **kwargs):
    # set paths
    datapath = Path(kwargs['datapath'])
    if datapath.is_symlink():
        datapath = datapath.readlink()
    if 'respath' not in kwargs or kwargs['respath'] is None:
        kwargs['respath'] = RESPATH / datapath.name
    else:
        if os.path.isdir(kwargs['respath']):
            shutil.rmtree(kwargs['respath'])
    kwargs['respath'] = Path(kwargs['respath'])
    kwargs['respath'].mkdir(parents=True, exist_ok=True)

    # get experiment name
    with open(datapath / 'experiment.info') as f:
        experiment = f.readline().strip()
     
    # set logger
    formatter = logging.Formatter(LOG_FORMAT)
    fileHandler = logging.FileHandler(kwargs['respath'] / 'train.log')
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(f"Started training of Experiment {experiment} with args: {kwargs=}")
    
    # create latest symlink
    (RESPATH / 'latest').unlink(missing_ok=True)
    (RESPATH / 'latest').symlink_to(kwargs['respath'])    

    return sys.exit(train(**kwargs))


if __name__ == '__main__':
    p = config_parser()
    args = p.parse_args()
    try:
        sys.exit(main(None, **vars(args)))
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception:
        print(traceback.format_exc())
