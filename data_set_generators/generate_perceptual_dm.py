"""
Perceptual Decision-Making with Integral Coding (Windowed Integration Decision-Making)
MIT License - C. Jarne V. 1.0 - 2024

This script generates training data for perceptual decision tasks where outputs depend
on the integral of sustained input stimuli. The response polarity encodes the sign
of the integrated input signal after a fixed processing delay.

Key Features:
- Integral-based decisions: Output = sign(∫input dt)
- Sustained stimuli: Input duration = mem_gap parameter
- Temporal structure: Fixed 10ms processing delay + 200ms stimulus duration
- Input conditioning: Hann-window shaped noise bursts
- Binary outputs: ±1 responses maintained until trial end

Parameters:
- size: Number of trials
- mem_gap: Stimulus duration (ms) & output delay controller
- stim_noise: Input amplitude scaling factor

Technical Implementation:
1. Input Generation:
   - Gaussian noise bursts (σ=1) shaped by 10-sample Hann window
   - Fixed 50ms initial delay
   - Stimulus duration controlled by mem_gap (200ms default)

2. Output Logic:
   - Numerical integration using Simpson's rule
   - Polarity determination: sign(integral_result)
   - Fixed 10ms processing delay post-stimulus
   - Sustained output until trial end

3. Temporal Structure:
   - Initial delay: 50ms
   - Stimulus phase: mem_gap duration (200ms)
   - Response phase: 210ms+
   - Total trial length: 440ms

4. Reproducibility: Seeded randomization (NumPy seed=2)

Outputs:
- x_train: Integrated inputs [samples × 440ms × 1 channel]
- y_train: Binary decisions [samples × 440ms × 1 channel] (±1)
- seq_dur: Fixed sequence duration
- PNG plot: 10 samples showing input-output relationships

Design Notes:
- Integration window: Full stimulus duration (mem_gap)
- Noise characteristics: Raw Gaussian noise before Hann smoothing
- Temporal precision: Fixed 100ms plotting ticks for visual alignment
- Output stability: Sustained signal after decision point
- Experimental control: Zero variable delay (var_delay_length=0)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
from numpy.random import seed
from scipy import signal

start_time = time.time()


def generate_trials(size, mem_gap):
    seed(2)
    first_in = 50   # time to start the first stimulus
    stim_dur = mem_gap  # 100  #stimulus duration
    stim_noise = 1  # 0.03 #noise
    var_delay_length = 0    # change for a variable length stimulus
    out_gap = 250-20  # how much lenth add to the sequence duration
    sample_size = size  # sample size
    
    xor_seed_A = np.array([[1], [1]]) # np.array([[0], [1]]) if you want zero input
    seq_dur = first_in+stim_dur+mem_gap+var_delay_length+(out_gap-mem_gap)
    win = scipy.signal.windows.hann(10)
 
    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1

    out_t = 10 + first_in+stim_dur
    trial_types = np.random.randint(2, size=sample_size)
    x_train_ = np.zeros((sample_size, seq_dur, 1))
    x_train = np.zeros((sample_size, seq_dur, 1))
    y_train = 0 * np.ones((sample_size, seq_dur, 1))

    for ii in np.arange(sample_size):

        x_train_[ii, first_in:first_in + stim_dur, 0] = \
            xor_seed_A[trial_types[ii], 0] * stim_noise * np.random.randn(stim_dur)
        x_train[ii, first_in:first_in + stim_dur, 0] = \
            signal.convolve(x_train_[ii, first_in:first_in + stim_dur, 0], win, mode='same') / sum(win)
        y_train[ii, out_t + var_delay[ii]+10:, 0] =\
            np.sign(scipy.integrate.simps(x_train[ii, first_in:first_in + stim_dur, 0],
                                          x=None, dx=1, axis=-1, even='avg'))
       
    x_train = x_train  # + stim_noise * np.random.randn(sample_size, seq_dur, 1)
    y_train = y_train  # + stim_noise * np.random.randn(sample_size, seq_dur, 1)
   
    print("--- %s seconds to generate learning dataset---" % (time.time() - start_time))
    return x_train, y_train, seq_dur


# Data set generation
sample_size = 10
x_train, y_train, seq_dur = generate_trials(sample_size, 200)

fig = plt.figure(figsize=(6, 8))
fig.suptitle("Data Set Training Samples\n (amplitude in arb. units time in ms)", fontsize=20)
for ii in np.arange(10):
    plt.subplot(5, 2, ii + 1)    
    plt.plot(x_train[ii, :, 0], color='g',label="Input")
    plt.plot(y_train[ii, :, 0], color='gray', linewidth=2, label="Expected Output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize=5, loc=3)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xticks(np.arange(0, len(x_train[0])+50, 100), fontsize=8)

fig_name = "dataset_perceptual_dm_int.png"
plt.savefig(fig_name, dpi=200)
plt.show()
