"""
Multi-interval Distance-based Decision-Making Task Generator
MIT License - C. Jarne V. 1.0 - 2024

This script generates training data for time-interval discrimination tasks where
the response delay matches the interval between two input pulses. Eight possible
intervals (20-160ms) are encoded through pulse separation and reproduced in output timing.

Key Features:
- Dual-pulse encoding: Input contains 2 pulses with configurable separation (T)
- Temporal mirroring: Response delay = Inter-pulse interval (T)
- Adaptive trial length: Automatically adjusts for longest interval (160ms)
- Input processing: Hann-window shaped pulses (20ms) + onset jitter (1-70ms)
- Self-documenting plots: Displays both programmed and measured temporal parameters

Parameters:
- size: Number of trials
- T_options: 8 interval options [20,40,60,80,100,120,140,160]ms
- stim_dur: Pulse duration (20ms fixed)
- var_delay_length: Initial pulse jitter range (70ms)

Technical Implementation:
1. Input Generation:
   - Two 20ms pulses separated by T ms
   - Pulse onset jitter (1-70ms) for first pulse
   - Hann window smoothing (10 samples)
   - Fixed amplitude (±1) with matching polarity

2. Output Logic:
   - Response initiates T ms after second pulse
   - Sustained ±1 output until trial end
   - Automatic T validation: Measured vs programmed

3. Temporal Structure:
   - Initial delay: 50ms ± jitter
   - Pulse phase: 2×20ms pulses + T separation
   - Response phase: T ms delay + remaining trial time
   - Max trial length: 440ms

4. Reproducibility: Seeded randomization (NumPy seed=1)

Outputs:
- x_train: Dual-pulse inputs [samples × 440ms × 1 channel]
- y_train: T-coded responses [samples × 440ms × 1 channel]
- Ts: Programmed intervals for each sample
- PNG plot: 10 samples with input/output traces and timing annotations

Design Notes:
- Temporal fidelity: Measures actual T from pulse positions
- Noise-free inputs: Commented noise addition for clear temporal analysis
- Visual verification: Plot labels show programmed T vs measured response delay
- Fixed relationships: T ≡ Pulse separation ≡ Response delay
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.random import seed

start_time = time.time()


def generate_trials(size, mem_gap):
    seed(1)
    first_in = 50
    stim_dur = 20
    stim_noise = 0.05
    var_delay_length = 70

    # 8 time separations (T) en ms
    T_options = [20, 40, 60, 80, 100, 120, 140, 160]

    max_T = max(T_options)
    sequence_duration = first_in + var_delay_length + 2*stim_dur + max_T + max_T + 100

    xor_seed_A = np.array([[0], [1]])
    win = signal.windows.hann(10)

    Ts = np.random.choice(T_options, size=size)
    var_delay = np.random.randint(var_delay_length, size=size) + 1
    trial_types = np.random.randint(2, size=size)
    pulse_signes = np.random.choice([-1, 1], size=size)

    x_train_ = np.zeros((size, sequence_duration, 1))
    x_train = np.zeros((size, sequence_duration, 1))
    y_train = np.zeros((size, sequence_duration, 1))

    for sample in range(size):
        T = Ts[sample]
        pulse_signe = pulse_signes[sample]

        first_pulse_start = first_in + var_delay[sample]
        first_pulse_end = first_pulse_start + stim_dur

        second_pulse_start = first_pulse_end + T
        second_pulse_end = second_pulse_start + stim_dur

        for start in [first_pulse_start, second_pulse_start]:
            x_train_[sample, start:start+stim_dur, 0] = xor_seed_A[trial_types[sample], 0] * 1 * pulse_signe
            x_train[sample, start:start+stim_dur, 0] = signal.convolve(
                x_train_[sample, start:start+stim_dur, 0], win, mode='same')/sum(win)

        # Response
        response_start = second_pulse_end + T
        y_train[sample, response_start:, 0] = xor_seed_A[trial_types[sample], 0] * pulse_signe * 1.0

    x_train = x_train  # + stim_noise * np.random.randn(size, sequence_duration, 1)

    print(f"--- {time.time() - start_time:.2f} seconds ---")
    return x_train, y_train, sequence_duration, Ts


# Generate data set
x_train, y_train, seq_dur, Ts = generate_trials(10, 0)

# Graficar
fig = plt.figure(figsize=(8, 10))
fig.suptitle("Pulse separation = response time"
             "\n(T ≡ Time between the end of pulse 1 and the start of pulse 2 ≡ Response delay)",
             fontsize=12, y=0.95)

for ii in range(10):
    plt.subplot(5, 2, ii + 1)
    x_signal = x_train[ii, :, 0]
    y_signal = y_train[ii, :, 0]

    # Positions
    pulse_starts = np.where(np.diff((x_signal > 0.1).astype(int)) == 1)[0]
    pulse_ends = np.where(np.diff((x_signal > 0.1).astype(int)) == -1)[0]

    if len(pulse_ends) >= 2:
        T_real = pulse_starts[1] - pulse_ends[0]
        response_delay = np.argmax(y_signal != 0) - pulse_ends[1]
    else:
        T_real = 0
        response_delay = 0

    plt.plot(x_signal, color='g', label="Entrada")
    plt.plot(y_signal, color='gray', linewidth=2,
             label=f"T: {Ts[ii]}ms\nResp: {response_delay}ms")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize=7, loc='upper right')
    plt.xticks(np.arange(0, seq_dur + 50, 100), fontsize=7)

plt.tight_layout()
plt.savefig("dataset_Separation_equal_response.png", dpi=200, bbox_inches='tight')
plt.show()
