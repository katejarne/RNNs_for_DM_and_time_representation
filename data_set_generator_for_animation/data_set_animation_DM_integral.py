"""
MIT License - C. Jarne V. 1.0 - 2024
Dataset generator to create task animations for
Perceptual Decision-Making with Integral Coding (Windowed  Integration  Decision-Making)
"""
import numpy as np
from scipy import signal
import scipy
from numpy.random import seed


def generate_single_trial(mem_gap=200, first_in=50, stim_dur=100, ensure_negative_integral=False):
    #seed(2)
    seed(3)
    stim_noise = 1
    out_gap = 250 - 20
    rec_noise = 0

    xor_seed_A = np.array([[0], [1]])
    seq_dur = first_in + stim_dur + mem_gap + (out_gap - mem_gap)
    win = signal.hann(10)

    out_t = mem_gap + first_in + stim_dur
    x_train_ = np.zeros((seq_dur, 1))
    x_train = np.zeros((seq_dur, 1))
    y_train = 0.045 * np.ones((seq_dur, 1))

    trial_type = np.random.randint(2)
    if ensure_negative_integral:
        x_train_[first_in:first_in + stim_dur, 0] = -(xor_seed_A[trial_type, 0] + stim_noise * np.random.randn(stim_dur))
    else:
        x_train_[first_in:first_in + stim_dur, 0] = xor_seed_A[trial_type, 0] + stim_noise * np.random.randn(stim_dur)

    x_train[first_in:first_in + stim_dur, 0] = signal.convolve(x_train_[first_in:first_in + stim_dur, 0], win, mode='same') / sum(win)

    y_train[out_t:, 0] = np.sign(scipy.integrate.simps(x_train[first_in:first_in + stim_dur, 0], x=None, dx=1, axis=-1, even='avg'))

    return x_train, y_train, seq_dur, first_in + stim_dur, out_t
