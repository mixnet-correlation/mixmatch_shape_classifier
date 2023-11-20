import json
import contextlib

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def get_dirs(json_file):
    '''Yield directory paths of experiment's captures.'''
    with open(json_file, 'rb') as fi:
        for d in json.load(fi):
            yield Path(json_file).parent / d


@contextlib.contextmanager
def figure_context(*args, **kwargs):
    # contextmanager to close figure
    fig = plt.figure(*args, **kwargs)
    yield fig
    plt.close(fig)


def parse_config(datapath, config):
    config.sections()
    config_file = datapath / 'config.ini'
    config.read(config_file)
    params = config['DEFAULT']
    exp_sec = [s for s in config.sections()
                                if 'EXP' in s][-1]
    params.update(config[exp_sec])
    return params


def read_csv_results(csv_path, batch_size=15, negatives=5250):
    column_names = ['N', 'TH', 'Packet type', 'TPS', 'FPS']
    data = pd.read_csv(csv_path, names=column_names)
    group = data.groupby(['N', 'TH', 'Packet type'])
    data = group.sum()

    data['TPR'] = data.TPS / (batch_size * group.count().TPS)
    data['FPR'] = data.FPS / ((negatives * batch_size - batch_size) * group.count().FPS)
    data = data.reset_index().drop(columns=['TH', 'TPS', 'FPS'])
    data['Packet type'] = data['Packet type'].replace('both', 'acks+data')

    return data


def read_csv_losses(csv_path):
    ##  read csv as dataframe
    loss_df = pd.read_csv(csv_path, names=['train', 'val'])
    loss_df = loss_df.reset_index(names='epoch')
    loss_df = pd.melt(loss_df, id_vars=['epoch'], value_vars=['train', 'val'])

    return loss_df


def plot_loss(loss_df, png_path):
    ## print figure and save it to a file
    with figure_context(figsize=(10, 10)):
        sns.lineplot(data=loss_df, x='epoch', y='value', hue='variable')
        plt.savefig(png_path)


def plot_roc(data, png_path):
    # style        
    with figure_context(figsize=(6, 5)):
        g = sns.lineplot(data=data,
                          x='FPR', y='TPR', hue='N', style='Packet type',
                          legend='full', errorbar=None)

        # line for random classifier
        diag = np.arange(0, 1, 0.001)
        sns.lineplot(x=diag, y=diag, color='gray', linestyle='--')

        plt.setp(g.lines, alpha=.9)
        g.set(xlabel='FPR\n',
              ylabel='TPR',
              xscale='log',
              xlim=(0.00000008, 0.25));

        g.set_title("ROC curve for tuning\n\n\n")

        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig(png_path)





