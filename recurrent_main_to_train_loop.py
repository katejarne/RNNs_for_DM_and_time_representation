# recurrent_main_to_train_loop.py
"""
This script trains a Recurrent Neural Network (RNN) on various decision-making (DM) tasks,
including simple and integrative temporal tasks.
The script dynamically selects the appropriate dataset generator based on the task parameters
and initializes the RNN with customizable constraints,
initialization schemes, and dense layer configurations.
The model is trained to predict the expected output of each task, and training loss
is monitored with early stopping and model checkpointing.
The script supports modular adjustments to task parameters, recurrent constraints, and dataset configurations.
"""

import warnings
import matplotlib.pyplot as plt
import os

os.environ['TF_DISABLE_GPU'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.layers import SimpleRNN, Layer, Dense
from keras import optimizers, initializers
from tensorflow.keras.constraints import Constraint

# Select generator according to de task parametrization

#################################################################################################################
# Parameters:                                                                                                   #
# task: "Simple DM", "Simple DM Long-short", "Simple DM 4 times","Simple DM 8 times", "Integral DM",            #
# "Integral DM signal keep", "Integral DM Cue","Multi Ampli"                                                    #
# mem_gap: depend on task (time between stimulus ends and answers start)                                        #
# Init_rrn: pepe, "Orthogonal", keras initializer                                                               #
# use_bias: False, True
# epochs = 20 for Simple dm-like tasks
# recurrent_constraint: could be "None" or any particular network connectivity constraint "CustomConstraint"    #
# Network dense layer could be change for "CustomDense" to train only some weights and not all                  #
#################################################################################################################

task = "Simple DM"  # "Integral DM" #"interval compare"#"Simple DM"  #"Simple DM 8 time encoded"# "Simple DM 8 times"
# "Simple DM 8 time encoded" #"Integral DM Cue" #"Integral DM signal keep"  # "Multi Ampli"#"Integral DM"
bias = False
recurrent_constraint = None
Init_rrn = "Orthogonal"
epochs = 20

# Pure Temporal tasks

if task in ["Simple DM", "Simple DM Long-short", "Simple DM 4 times","Simple DM 8 times",
            "Simple DM 8 time encoded"]:
    # parameters
    mem_gap = 0
    Init_rnn = "Orthogonal"
    if task in ["Simple DM", "Simple DM Long-short"]:
        # be aware of internal conf for data_set_generator to create one data set or the other
        from data_set_generators.generate_DM_delayed_response_sample import *
    if task == "Simple DM 4 times":
        from data_set_generators.generate_DM_delayed_response_sample_mult_times_4 import *
    if task == "Simple DM 8 times":
        from data_set_generators.generate_DM_delayed_response_sample_mult_times_8 import *
    if task == "Simple DM 8 time encoded":
        from data_set_generators.generate_DM_delayed_response_sample_mult_times_8_intervals import *

# Integration tasks (temporal + integration +cue)
if task in ["Integral DM", "Integral DM signal keep", "Integral DM Cue"]:
    # parameters
    Init_rnn = "Orthogonal"
    print("ACA")

    if task == "Integral DM":
        from data_set_generators.generate_perceptual_dm import *
        mem_gap = 200
    if task == "Integral DM signal keep":
        from data_set_generators.generate_perceptual_dm_sig_not_end import *
        mem_gap = 50
    if task == "Integral DM Cue":
        from data_set_generators.generate_perceptual_dm_sig_not_end_cue_mod import *
        mem_gap = 50
# Multi amplitude task
if task == "Multi Ampli":
    from data_set_generators.generate_DM_delayed_response_sample_mult_amplitude_8 import *
    mem_gap = 100

if task == "interval compare":
    from data_set_generators.generate_interval_comparison import *
    mem_gap = 20

task_sample_sizes = {"Simple DM": 15050, "Simple DM Long-short": 15050, "Simple DM 4 times": 15050,
                     "Simple DM 8 times": 15050,
                     "Integral DM": 6*15050, "Integral DM signal keep": 6*15050,  "Integral DM Cue": 15050,
                     "Multi Ampli": 6*15050, "Simple DM 8 time encoded": 15050, "interval compare":15050 }

sample_size = task_sample_sizes.get(task)

if sample_size is None:
    raise ValueError(f"Unknown task: {task}")


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.000001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        if current < self.value:
            if self.verbose > 0:
                print(" Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


class CustomDense(Layer):
    def __init__(self, units, trainable_indices=None, use_bias=True, **kwargs):
        self.units = units
        self.trainable_indices = trainable_indices
        self.use_bias = use_bias
        super(CustomDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", (input_shape[-1], self.units), initializer='zeros', trainable=True)
        self.mask = np.zeros((input_shape[-1], self.units), dtype=np.float32)
        if self.trainable_indices is not None and len(self.trainable_indices) > 0:
            self.mask[self.trainable_indices] = 1.0
        self.mask = self.add_weight("mask", (input_shape[-1], self.units),
                                    initializer=tf.initializers.Constant(self.mask), trainable=False)
        if self.use_bias:
            self.bias = self.add_weight("bias", (self.units,), initializer="zeros", trainable=True)
        super(CustomDense, self).build(input_shape)

    def call(self, inputs):
        masked_kernel = self.kernel * self.mask
        output = tf.matmul(inputs, masked_kernel)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        return output

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({
            'units': self.units,
            'trainable_indices': self.trainable_indices,
            'use_bias': self.use_bias
        })
        return config


class CustomConstraint(Constraint):
    def __call__(self, w):
        num_inputs = tf.shape(w)[-1] // 2
        num_filas = tf.shape(w)[0] // 2
        input1, input2 = w[:num_filas, :num_inputs], w[num_filas:, num_inputs:]
        input1 = input1 * tf.cast(tf.equal(input1, 0.), tf.float32)
        input2 = input2 * tf.cast(tf.equal(input2, 0.), tf.float32)
        last_rows_1 = w[num_filas:, :num_inputs]
        last_rows_2 = w[:num_filas, num_inputs:]
        full_1 = tf.concat([input1, last_rows_1], 0)
        full_2 = tf.concat([last_rows_2, input2], 0)
        result = tf.concat([full_1, full_2], axis=-1)
        return result


x_train, y_train, seq_dur = generate_trials(sample_size, mem_gap)
# x_train, y_train, seq_dur, Ts = generate_trials(sample_size, mem_gap)

# Low rank init


class LowRankInitializer(tf.keras.initializers.Initializer):
    def __init__(self, rank, scale=1.05):
        self.rank = rank
        self.scale = scale  # Factor 1.05 del paper

    def __call__(self, shape, dtype=None):
        units = shape[0]  # N_rec
        # Estimate σ para A y B
        sigma = (self.scale**2 / (self.rank * units))  # ** 0.25
        sigma = tf.cast(sigma, dtype=dtype)
        # A and B matrices
        A = tf.random.normal((units, self.rank), stddev=sigma, dtype=dtype)
        B = tf.random.normal((self.rank, units), stddev=sigma, dtype=dtype)
        # W low rank
        W = tf.matmul(A, B)
        return W

    def get_config(self):
        return {'rank': self.rank, 'scale': self.scale}


class LowRankRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, rank, scale=1.05, activation='tanh', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.rank = rank
        self.scale = scale
        self.activation = tf.keras.activations.get(activation)
        self.state_size = units
        self.output_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Init A and B
        sigma = (self.scale**2 / (self.rank * self.units))**0.5
        initializer = tf.keras.initializers.RandomNormal(stddev=sigma)
        self.A = self.add_weight(shape=(self.units, self.rank), initializer=initializer, name='A')
        self.B = self.add_weight(shape=(self.rank, self.units), initializer=initializer, name='B')
        self.W_xh = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_xh')
        self.bias = self.add_weight(shape=(self.units,), initializer='zeros', name='bias')
        self.built = True

    def call(self, inputs, states):

        h_prev = states[0] if states else self.get_initial_state(batch_size=tf.shape(inputs)[0], dtype=inputs.dtype)

        #  W = A * B
        W_rec = tf.matmul(self.A, self.B)
        h = self.activation(tf.matmul(inputs, self.W_xh) + tf.matmul(h_prev, W_rec) + self.bias)
        return h, [h]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # If dtype is not specified, use the dtype of the weights (or float32 by default)
        if dtype is None:
            dtype = self.A.dtype  # Usa el dtype de la matriz A (ya inicializada)

        # If batch_size is None, try to get it from inputs
        if batch_size is None and inputs is not None:
            batch_size = tf.shape(inputs)[0]

        # initial state (zeros)
        return [tf.zeros(shape=(batch_size, self.units), dtype=dtype)]

# Parameters

rank = 4  # low rank implementation
recurrent_constraint = None
epochs = 20


def DM_fun(t, N_rec, base, base_plot):
    lista_distancia = []
    R = 50
    # Rand normal inicialization with adjustable sigma
    RandNor_sigma_adjuts = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0*np.sqrt(float(1)/float(N_rec)),
                                                              seed=None)
    # Init low-rank
    recurrent_initializer = LowRankInitializer(rank=rank, scale=1.05)

    # trainable_indices = np.random.choice(N_rec, R, replace=False)
    # seed(None)

    print("Starting Training:")
    model = Sequential()
    model.add(SimpleRNN(units=N_rec, return_sequences=True, kernel_constraint=None,
                        kernel_initializer='glorot_uniform', recurrent_initializer=RandNor_sigma_adjuts,
                        recurrent_constraint=recurrent_constraint, use_bias=bias))
    model.add(Dense(units=1))
    # model.add(Dense(units=1, kernel_initializer=initializers.RandomUniform(minval=-1, maxval=1)))  # Uniform distr
    # model.save(base + '/' +'initial.hdf5')  # Use .keras format

    # low rank tries
    # model = tf.keras.Sequential([
    # tf.keras.layers.Input(shape=(None, 1)),  # Secuencia de longitud variable
    # tf.keras.layers.RNN(LowRankRNNCell(units=N_rec, rank=4), return_sequences=True)])

    ADAM = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0)

    model.compile(loss='mse', optimizer=ADAM)
    # model.summary()
    model.build(input_shape=(None, x_train.shape[1], x_train.shape[2]))  # Define la forma de entrada
    model.save(base + '/' + str(N_rec)+'_initial.hdf5')
    filepath_k = base + '/simple_DM_weights-{epoch:02d}.keras'  # Use .keras format
    callbacks = [EarlyStoppingByLossVal(monitor='loss', value=0.00005, verbose=1),
                 ModelCheckpoint(filepath_k, monitor='val_loss', save_best_only=False, verbose=1)]

    history = model.fit(x_train[50:sample_size, :, :], y_train[50:sample_size, :, :],
                        epochs=epochs, batch_size=64, callbacks=callbacks)

    model.save(base + '/' + str(N_rec)+'_final.hdf5')
    x_pred = x_train[0:50, :, :]
    y_pred = model.predict(x_pred)

    print("X train shape:\n", x_train.shape)
    print("X prediction shape\n", x_pred.shape)
    print("Y train shape\n", y_train.shape)

    fig = plt.figure(figsize=(6, 8))
    fig.suptitle("\"DM\" Data Set Trained Output \n (amplitude in arb. units time in ms)", fontsize=20)
    for ii in np.arange(10):
        plt.subplot(5, 2, ii + 1)
        plt.plot(x_train[ii, :, 0], color='g', label="Input")
        if x_train.shape[2] == 2:
            plt.plot(x_train[ii, :, 1], color='pink',label="Cue")
        plt.plot(y_train[ii, :, 0], color='gray', linewidth=2, label="Expected Output")
        plt.plot(y_pred[ii, :, 0], color='r', label="Predicted Output")
        plt.ylim([-2.5, 2.5])
        plt.legend(fontsize=5, loc=3)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        a = y_train[ii, :, 0]
        b = y_pred[ii, :, 0]
        a_min_b = np.linalg.norm(a - b)
        lista_distancia.append(a_min_b)
    figname = base_plot + "/data_set_sample_trained.png"
    plt.savefig(figname, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.grid(True)
    plt.plot(history.history['loss'])
    plt.title('Model loss during training')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig_name = base_plot + "/model_loss_" + str(N_rec) + ".png"
    plt.savefig(fig_name, dpi=200)
    print(model.summary())

    return lista_distancia
