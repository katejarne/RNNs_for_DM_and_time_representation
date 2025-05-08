"""
MIT License - C. Jarne V. 1.0 - 2024
Dataset generator to create task animations for
Delayed Binary Decision-Making Task
"""
import numpy as np
from scipy import signal
import scipy


def generate_single_trial(pulse_height, pulse_sign, mem_gap=20, first_in=50, stim_dur=20):
    xor_seed_A = np.array([[0], [1]])
    seq_dur = first_in + stim_dur + mem_gap + (250 - 20 - mem_gap)
    win = scipy.signal.windows.hann(10)

    out_t = mem_gap + first_in + stim_dur
    x_train_ = np.zeros((seq_dur, 1))
    x_train = np.zeros((seq_dur, 1))
    y_train = np.zeros((seq_dur, 1))

    trial_type = 1  # Use a fixed trial type for simplicity
    stimulus = xor_seed_A[trial_type, 0] * pulse_height * pulse_sign
    convolved_stimulus = signal.convolve(stimulus * np.ones(stim_dur), win, mode='same') / sum(win)
    x_train[first_in:first_in + stim_dur, 0] = convolved_stimulus

    if pulse_height == 1 and pulse_sign == 1:
        y_train[out_t + 50:, 0] = xor_seed_A[trial_type, 0]
    elif pulse_height == 1 and pulse_sign == -1:
        y_train[out_t + 50:, 0] = -1 * xor_seed_A[trial_type, 0]
    elif pulse_height == 2 and pulse_sign == 1:
        y_train[out_t + 100:, 0] = xor_seed_A[trial_type, 0]
    elif pulse_height == 2 and pulse_sign == -1:
        y_train[out_t + 100:, 0] = -1 * xor_seed_A[trial_type, 0]

    return x_train, y_train, seq_dur, first_in + stim_dur, out_t
