"""
Synthetic Dataset Generator for Delayed Binary Decision-Making Task
MIT License - C. Jarne V. 1.0 - 2024

This script generates training data for basic delayed decision-making model.
It creates binary input-output pairs where outputs replicate input polarity
after a fixed delay, with randomized stimulus timing and additive noise.

Key Features:
- Binary decisions: Detect input pulse presence (0/1)
- Polarity conservation: ± inputs → ± outputs
- Temporal variability: Random stimulus delays (0-70ms jitter)
- Input processing: Hann window smoothing + Gaussian noise
- Response timing: Fixed 50ms post-stimulus response latency

Parameters:
- size: Number of trials (samples)
- mem_gap: Memory maintenance period (ms)
- stim_dur: Stimulus pulse duration (20ms fixed)
- stim_noise: Input noise level (σ=0.05)
- var_delay_length: Maximum stimulus delay jitter (70ms)

Technical Implementation:
1. Input Generation:
   - Square pulses smoothed with 10-point Hann window
   - ±1 polarity randomization per trial
   - Stimulus jitter: ±35ms temporal variability
   - Additive white noise (μ=0, σ=0.05)

2. Output Generation:
   - Fixed 50ms response delay post-stimulus
   - Output amplitude = Input polarity (±1)
   - Response duration: 230ms sustained output

3. Temporal Structure:
   - Initial fixation: 50ms
   - Stimulus phase: 20ms ± jitter
   - Memory period: mem_gap ms
   - Response phase: 230ms

4. Reproducibility: Fixed random seed (NumPy seed=2)

Outputs:
- x_train: Input sequences [samples × time × channels]
- y_train: Target outputs [samples × time × channels]
- sequence_duration: Total trial length (370ms)
- PNG plot: 10-sample visualization with input/output traces
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.random import seed
from scipy import signal

start_time = time.time()


def generate_trials(size, mem_gap):
    seed(2)  # seed(None)
    first_in = 50
    stim_dur = 20
    stim_noise = 0.05
    var_delay_length = 70
    out_gap = 250 - 20
    xor_seed_A = np.array([[0], [1]])
    sequence_duration = first_in + stim_dur + mem_gap + var_delay_length + (out_gap - mem_gap)
    win = scipy.signal.windows.hann(10)

    var_delay = np.zeros(size, dtype=int) if var_delay_length == 0 else np.random.randint(var_delay_length, size=size) + 1
    out_t = mem_gap + first_in + stim_dur

    trial_types = np.random.randint(2, size=size)
    pulse_heights = np.ones(size)  # if simple DM #long short: np.random.choice([2, 1]
    pulse_signes = np.random.choice([-1, 1], size=size)

    x_train_ = np.zeros((size, sequence_duration, 1))
    x_train = np.zeros((size, sequence_duration, 1))
    y_train = np.zeros((size, sequence_duration, 1))

    for sample in range(size):
        pulse_height = pulse_heights[sample]
        pulse_signe = pulse_signes[sample]
        x_train_[sample, first_in+var_delay[sample]:first_in + stim_dur+var_delay[sample], 0] \
            = xor_seed_A[trial_types[sample], 0] * pulse_height * pulse_signe
        x_train[sample, first_in+var_delay[sample]:first_in + stim_dur+var_delay[sample], 0] \
            = signal.convolve(x_train_[sample, first_in+var_delay[sample]:first_in+var_delay[sample]
                                                                          + stim_dur, 0], win, mode='same') / sum(win)

    x_train = x_train + stim_noise* np.random.randn(size, sequence_duration, 1)

    for sample in range(size):
        if pulse_heights[sample] == 1:
            y_train[sample, out_t + var_delay[sample] + 50:, 0] = \
                xor_seed_A[trial_types[sample], 0] * pulse_signes[sample]
        else:
            y_train[sample, out_t + 100 + var_delay[sample]:, 0] = \
                xor_seed_A[trial_types[sample], 0] * pulse_signes[sample]

    print("--- %s seconds to generate learning dataset---" % (time.time() - start_time))

    return x_train, y_train, sequence_duration


x_train, y_train, seq_dur = generate_trials(10, 0)


fig = plt.figure(figsize=(6, 8))
fig.suptitle("10 Training Samples\n (amplitude in arb. units, time in ms)", fontsize=20)
for ii in range(10):
    plt.subplot(5, 2, ii + 1)
    plt.plot(x_train[ii, :, 0], color='g', label="Input")
    plt.plot(y_train[ii, :, 0], color='gray', linewidth=2, label="Expected Output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize=5, loc=3)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xticks(np.arange(0, seq_dur+50, 100), fontsize=8)

fig_name = "Data_Set_simple_dm.png"
plt.savefig(fig_name, dpi=200)
plt.show()
