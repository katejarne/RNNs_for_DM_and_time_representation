"""
Synthetic Dataset Generator for Amplitude-Coded Decision-Making Tasks
MIT License - C. Jarne V. 1.0 - 2024

This script generates training samples for amplitude-based decision-making models.
It creates input pulses with 8 discrete heights (1-8) and ± polarities, mapped to
fixed output amplitudes (0.25-2.0). The input pulses are smoothed with a Hann window,
embedded in variable delays, and corrupted by adjustable Gaussian noise.

Key Features:
- 8 input pulse heights → 8 output amplitudes (nonlinear encoding)
- Adjustable parameters: sequence duration, stimulus timing, noise level, delays
- Polarity inversion preservation (± inputs → ± outputs)
- Built-in visualization of 10 training samples

Parameters:
- size: Number of training samples
- mem_gap: Memory period between stimulus and response (ms)
- stim_dur: Stimulus pulse duration (ms)
- stim_noise: Input noise amplitude (Gaussian std)
- var_delay_length: Max random delay variability (ms)

Technical Implementation:
1. Input Processing: Pulse smoothing (Hann window convolution) + additive noise
2. Output Generation: Fixed-delay responses with amplitude coding (0.25-2.0 scale)
3. Temporal Structure:
   - Initial fixation period (50ms)
   - Stimulus + variable delay (70ms max)
   - Response period (230ms)
4. Reproducibility: Fixed random seed (NumPy)

Outputs:
- x_train: Input sequences (noisy smoothed pulses)
- y_train: Target outputs (delayed amplitude-coded responses)
- sequence_duration: Total trial length (ms)
- PNG plot: Visualizes 10 samples with input/output traces
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.random import seed

start_time = time.time()


def generate_trials(size, mem_gap):
    seed(1)  # Seed for reproducibility
    first_in = 50
    stim_dur = 20
    stim_noise = 0.05
    var_delay_length = 70
    out_gap = 250 - 20
    xor_seed_A = np.array([[0], [1]])
    sequence_duration = first_in + stim_dur + mem_gap + var_delay_length + (out_gap - mem_gap)
    win = signal.windows.hann(10)

    var_delay = np.zeros(size, dtype=int) if var_delay_length == 0 else np.random.randint(var_delay_length, size=size) + 1
    out_t = mem_gap + first_in + stim_dur

    trial_types = np.random.randint(2, size=size)
    pulse_heights = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=size)  # Alturas del pulso (1 a 8)
    pulse_signes = np.random.choice([-1, 1], size=size)

    height_to_amplitude = {1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.25, 6: 1.5, 7: 1.75, 8: 2.0}

    x_train_ = np.zeros((size, sequence_duration, 1))
    x_train = np.zeros((size, sequence_duration, 1))
    y_train = np.zeros((size, sequence_duration, 1))

    for sample in range(size):
        pulse_height = pulse_heights[sample]
        pulse_signe = pulse_signes[sample]
        output_amplitude = height_to_amplitude[pulse_height]

        # Input generation
        x_train_[sample, first_in + var_delay[sample]:first_in + stim_dur + var_delay[sample], 0] \
            = xor_seed_A[trial_types[sample], 0] * pulse_height * pulse_signe
        x_train[sample, first_in + var_delay[sample]:first_in + stim_dur + var_delay[sample], 0] \
            = signal.convolve(x_train_[sample, first_in + var_delay[sample]:first_in + var_delay[sample] + stim_dur, 0],
                              win, mode='same') / sum(win)

        # Output generation
        y_train[sample, out_t + 50 + var_delay[sample]:, 0] = \
            xor_seed_A[trial_types[sample], 0] * pulse_signe * output_amplitude

    # Add noise to the input
    x_train = x_train + stim_noise * np.random.randn(size, sequence_duration, 1)

    print("--- %s seconds to generate learning dataset---" % (time.time() - start_time))

    return x_train, y_train, sequence_duration


# Dataset example
x_train, y_train, seq_dur = generate_trials(10, 0)

# Plot results
fig = plt.figure(figsize=(6, 8))
fig.suptitle("10 Training Samples\n (amplitude in arb. units, time in ms)", fontsize=20)
for ii in range(10):
    plt.subplot(5, 2, ii + 1)
    plt.plot(x_train[ii, :, 0], color='g', label="Input")
    plt.plot(y_train[ii, :, 0], color='gray', linewidth=2, label=f"Output (Amplitude={y_train[ii, 50, 0]:.2f})")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize=5, loc=3)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xticks(np.arange(0, seq_dur + 50, 100), fontsize=8)

fig_name = "dataset_with_8_output_amplitudes.png"
plt.savefig(fig_name, dpi=200)
plt.show()
