# The model is inspired in the DF model by Sirinam et al
# https://github.com/triplet-fingerprinting/tf/blob/master/src/model_training/DF_model.py
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten


class DeepCoffeaModel(Model):
    def __init__(self, emb_size=64):
        super().__init__()        
        
        # parameters
        filter_num       = [ 32,  64, 128,  256]
        kernel_size      = [  8,   8,   8,    8]
        conv_stride_size = [  1,   1,   1,    1]
        pool_stride_size = [  4,   4,   4,    4]
        pool_size        = [  8,   8,   8,    8]
        dropout_rates    = [0.1, 0.2, 0.3, None]
        
        self.conv_layers = ()
        self.pool_layers = ()
        self.dropout_layers = ()
        
        for k in range(len(filter_num)):
            self.conv_layers += (Conv1D(filters=filter_num[k],
                                    kernel_size=kernel_size[k],
                                    strides=conv_stride_size[k],
                                    activation='relu',
                                    padding='same',
                                    name=f'block{k}_conv1'),)
            
            self.conv_layers += (Conv1D(filters=filter_num[k],
                                        kernel_size=kernel_size[k],
                                        strides=conv_stride_size[k],
                                        activation='relu',
                                        padding='same',
                                        name=f'block{k}_conv2'),)
            
            self.pool_layers += (MaxPooling1D(pool_size=pool_size[k],
                                              strides=pool_stride_size[k],
                                              padding='same',
                                              name=f'block{k}_pool'),)
            if k < len(filter_num) - 1:
                self.dropout_layers += (Dropout(dropout_rates[k],
                                                name=f'block{k}_dropout'),)

        self.flatten = Flatten()        
        self.dense = Dense(emb_size, name=f'FeaturesVec')

    def call_internal(self, input):
        embedding = input[:, 1:, :] - input[:, 0:-1, :]
        for k in range(len(self.pool_layers)):
            embedding = self.conv_layers[2*k](embedding)
            embedding = self.conv_layers[2*k + 1](embedding)
            embedding = self.pool_layers[k](embedding)
            if k < len(self.dropout_layers):
                embedding = self.dropout_layers[k](embedding)
        embedding = self.flatten(embedding)
        embedding = self.dense(embedding)
        return embedding

    def call(self, input):
        if type(input) == tuple:
            return (self.call_internal(i) for i in input)
        else:
            return self.call_internal(input)

class DriftModel(Model):
    # This drift model exclusively computes a score that only regards drift
    def __init__(self):
        super().__init__()
        self.conv_layers_drift = ()
        self.pool_layers_drift = ()
        self.fcc_drift = ()
        
        self.conv_layers_drift += (Conv1D(4, 8, activation='relu', padding='same'),)
        self.conv_layers_drift += (Conv1D(8, 8, activation='relu', padding='same'),)
        self.conv_layers_drift += (Conv1D(16, 8, activation='relu', padding='same'),)
        
        self.pool_layers_drift += (AveragePooling1D(pool_size = 8, strides = 2),)
        self.pool_layers_drift += (AveragePooling1D(pool_size = 8, strides = 2),)
        self.pool_layers_drift += (AveragePooling1D(pool_size = 8, strides = 2),)
        
        self.flatten_drift = Flatten()
        
        self.fcc_drift += (Dense(16, activation='relu'),)
        self.fcc_drift += (Dense(16, activation='relu'),)
        self.fcc_drift += (Dense(1, activation='sigmoid'),)
        
    def call(self, inputs):
        inflow, outflow = inputs
        drift_score = tf.cast(outflow, dtype=np.float32) - tf.cast(inflow, dtype=np.float32)
        
        for k in range(len(self.conv_layers_drift)):
            drift_score = self.conv_layers_drift[k](drift_score)
            drift_score = self.pool_layers_drift[k](drift_score)
        drift_score = self.flatten_drift(drift_score)
        
        for k in range(len(self.fcc_drift)):
            drift_score = self.fcc_drift[k](drift_score)
            
        return drift_score


class FusionModel(tf.keras.Model):
    # This model combines a drift score and a shape score.
    # The drift score is the cosine similarity of the outputs of a
    # DeepCoffeeModel or MyModel class instances.
    # The shape score is the output of a MyDriftModel_v2 class instance.
    def __init__(self):
        super().__init__()
        self.fcc = ()
        self.fcc += (tf.keras.layers.Dense(16, activation='relu'),)
        self.fcc += (tf.keras.layers.Dense(16, activation='relu'),)
        self.fcc += (tf.keras.layers.Dense(1, activation='sigmoid'),)

    def call(self, inputs):
        shape_score, drift_score = inputs
        output = self.fcc[0](tf.concat((shape_score, drift_score), axis=-1))
        for k in range(1, len(self.fcc)):
            output = self.fcc[k](output)

        return output
