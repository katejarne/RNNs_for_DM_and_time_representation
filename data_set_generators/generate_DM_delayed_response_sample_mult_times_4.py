"""
Multi-interval Amplitude-based Decision-Making Task Generator with Latency Coding
MIT License - C. Jarne V. 1.0 - 2024

This script creates training data for time-coded decision tasks where response latency
depends on input pulse height. Four discrete input amplitudes (1-4) map to fixed
response delays (25-100ms), with preserved polarity (±1) and temporal noise.

Key Features:
- Amplitude-temporal coding: 4 pulse heights → 4 response latencies (25,50,75,100ms)
- Polarity conservation: ± inputs → ± outputs with matching timing
- Temporal jitter: Randomized stimulus onset (0-70ms variability)
- Input processing: Hann-window smoothed pulses + Gaussian noise (σ=0.05)
- Adaptive outputs: Response duration scales with delay (longer latency = shorter response)

Parameters:
- size: Number of trials
- mem_gap: Fixed processing delay (ms)
- stim_dur: Stimulus duration (20ms fixed)
- stim_noise: Input noise amplitude
- var_delay_length: Maximum stimulus jitter (70ms)

Technical Implementation:
1. Input Generation:
   - 4-level pulse heights (1-4) with ±1 polarity
   - Hann window smoothing (10-sample)
   - Stimulus jitter (0-70ms) + additive noise

2. Output Generation:
   - Height-to-latency mapping: {1:25ms, 2:50ms, 3:75ms, 4:100ms}
   - Sustained binary response (±1) until trial end
   - Fixed 50ms processing delay post-stimulus

3. Temporal Structure:
   - Initial fixation: 50ms
   - Stimulus phase: 20ms ± jitter
   - Response phase: 180-255ms (inversely related to latency)

4. Reproducibility: Seeded randomization (NumPy seed=2)

Outputs:
- x_train: Noisy input pulses [samples × 370ms × 1 channel]
- y_train: Latency-coded outputs [samples × 370ms × 1 channel]
- sequence_duration: Total trial length (370ms)
- PNG plot: 10-sample visualization showing time-amplitude relationships
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.random import seed
start_time = time.time()


def generate_trials(size, mem_gap):
    seed(2)  # seed for reproducibility
    first_in = 50
    stim_dur = 20
    stim_noise = 0.05
    var_delay_length = 70
    out_gap = 250 - 20
    xor_seed_A = np.array([[0], [1]])
    sequence_duration = first_in + stim_dur + mem_gap + var_delay_length + (out_gap - mem_gap)
    win = signal.windows.hann(10)

    var_delay = np.zeros(size, dtype=int) if var_delay_length == 0 \
        else np.random.randint(var_delay_length, size=size) + 1
    out_t = mem_gap + first_in + stim_dur

    trial_types = np.random.randint(2, size=size)
    pulse_heights = np.random.choice([1, 2, 3, 4], size=size)
    pulse_signes = np.random.choice([-1, 1], size=size)
    height_to_time = {1: 25, 2: 50, 3: 75, 4: 100}

    x_train_ = np.zeros((size, sequence_duration, 1))
    x_train = np.zeros((size, sequence_duration, 1))
    y_train = np.zeros((size, sequence_duration, 1))

    for sample in range(size):
        pulse_height = pulse_heights[sample]
        pulse_signe = pulse_signes[sample]
        response_time = height_to_time[pulse_height]

        # Input generation
        x_train_[sample, first_in + var_delay[sample]:first_in + stim_dur + var_delay[sample], 0] \
            = xor_seed_A[trial_types[sample], 0] * pulse_height * pulse_signe
        x_train[sample, first_in + var_delay[sample]:first_in + stim_dur + var_delay[sample], 0] \
            = signal.convolve(x_train_[sample, first_in + var_delay[sample]:first_in + var_delay[sample] + stim_dur, 0],
                              win, mode='same') / sum(win)

        # Output generation
        if pulse_height > 0:  # if stimulus is present
            y_train[sample, out_t + response_time + var_delay[sample]:, 0] = \
                xor_seed_A[trial_types[sample], 0] * pulse_signe
        # else output remains 0

    # Adding input noise
    x_train = x_train + stim_noise * np.random.randn(size, sequence_duration, 1)

    print("--- %s seconds to generate learning dataset---" % (time.time() - start_time))

    return x_train, y_train, sequence_duration


# Generate dataset
x_train, y_train, seq_dur = generate_trials(10, 0)

fig = plt.figure(figsize=(6, 8))
fig.suptitle("10 Training Samples\n (amplitude in arb. units, time in ms)", fontsize=20)
for ii in range(10):
    plt.subplot(5, 2, ii + 1)
    plt.plot(x_train[ii, :, 0], color='g', label="Input")
    plt.plot(y_train[ii, :, 0], color='gray', linewidth=2)
    plt.ylim([-4, 4])
    plt.legend(fontsize=5, loc=3)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xticks(np.arange(0, seq_dur + 50, 100), fontsize=8)

fig_name = "dataset_with_4_response_times.png"
plt.savefig(fig_name, dpi=200)
plt.show()
