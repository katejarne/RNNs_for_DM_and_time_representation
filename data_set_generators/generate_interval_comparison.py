"""
Dual Interval Comparison Task Generator (TICT) from H. Diaz et al. https://doi.org/10.1073/pnas.2420356122
MIT License - C. Jarne V. 1.0 - 2024

This script creates training data for temporal interval discrimination tasks comparing
two pairs of pulses. The output polarity indicates which interval (Δ1/Δ2) was longer through
fixed response delays and sustained output signals.

Key Features:
- Dual interval comparison: Two pulse pairs with Δ1 (25/50ms) and Δ2 (50/25ms)
- Fixed separation: 100ms between interval pairs
- Decision coding: +1 output if Δ1 < Δ2, -1 if Δ1 > Δ2
- Temporal structure: 4 pulses with configurable delays + 20ms response gap
- Input conditioning: Hann-window shaped pulses (10ms)

Parameters:
- size: Number of trials
- mem_gap: Response delay after final pulse (20ms)
- short: Base interval duration (25ms)
- interval_sep: Fixed separation between interval pairs (100ms)

Technical Implementation:
1. Input Generation:
   - 4 pulses arranged as [P1-P2]-100ms-[P3-P4]
   - Interval relationships:
     - Type 0: Δ1=25ms, Δ2=50ms → +1 response
     - Type 1: Δ1=50ms, Δ2=25ms → -1 response
   - First pulse jitter: 0-70ms onset variability

2. Output Logic:
   - 20ms fixed response delay after final pulse
   - Sustained output until trial end (±1)
   - Polarity encodes interval relationship

3. Temporal Structure:
   - Initial delay: 50ms ± jitter
   - Interval phase: 2×10ms pulse pairs
   - Response phase: 20ms delay + sustained signal
   - Max trial length: 440ms

4. Reproducibility: Seeded randomization (NumPy seed=1)

Outputs:
- x_train: 4-pulse inputs [samples × 440ms × 1 channel]
- y_train: Decision-coded outputs [samples × 440ms × 1 channel]
- trial_types: Ground truth labels (0=Δ1<Δ2, 1=Δ1>Δ2)
- PNG plot: 10 samples with input/output traces and trial types

Design Notes:
- Binary decision task: Focuses on relative interval duration comparison
- Fixed experimental structure: Controlled separation between interval pairs
- Visual clarity: Noise-free inputs for clean temporal relationships
- Symmetric design: Equal number of short-long and long-short trials
- Constant amplitude: All pulses use ±1 magnitude for contrast
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.random import seed

start_time = time.time()


def generate_trials(size, mem_gap):
    short = 25
    interval_sep = 100
    seed(1)
    first_in = 50
    stim_dur = 10
    stim_noise = 0.05
    var_delay_length = 70

    # Fixed intervals
    delta1_options = [short, 2*short]  # delta1 it could be 50 or 100
    delta2_options = [2*short, short]  # delta2 will be 100 if delta1 is 50, and 50 if delta1 is 100

    # Total sequence duration (calculated for the worst case)
    sequence_duration = first_in + var_delay_length + 4*stim_dur + max(delta1_options) + max(delta2_options) + 200 + 100

    win = signal.windows.hann(10)

    # Trial type selection (delta1 < delta2 o delta1 > delta2)
    trial_types = np.random.randint(2, size=size)  # 0: delta1 < delta2, 1: delta1 > delta2
    pulse_signes = np.ones(size)  # np.random.choice([-1, 1], size=size)

    x_train_ = np.zeros((size, sequence_duration, 1))
    x_train = np.zeros((size, sequence_duration, 1))
    y_train = np.zeros((size, sequence_duration, 1))

    for sample in range(size):
        pulse_signe = pulse_signes[sample]

        # Assign delta1 and delta2 according to the trial type
        if trial_types[sample] == 0:
            delta1 = short   # delta1 < delta2
            delta2 = 2 * delta1
        else:
            # delta1 > delta2
            delta2 = short
            delta1 = 2 * delta2

        first_pulse_start = first_in + np.random.randint(var_delay_length)
        first_pulse_end = first_pulse_start + stim_dur

        second_pulse_start = first_pulse_end + delta1
        second_pulse_end = second_pulse_start + stim_dur

        third_pulse_start = second_pulse_end + interval_sep
        third_pulse_end = third_pulse_start + stim_dur

        fourth_pulse_start = third_pulse_end + delta2
        fourth_pulse_end = fourth_pulse_start + stim_dur

        # Pulse generation
        for start in [first_pulse_start, second_pulse_start, third_pulse_start, fourth_pulse_start]:
            x_train_[sample, start:start+stim_dur, 0] = 1 * pulse_signe
            x_train[sample, start:start+stim_dur, 0] = signal.convolve(
                x_train_[sample, start:start+stim_dur, 0], win, mode='same')/sum(win)

        # Response: 10 ms after the falling edge of the second pulse of the second pair
        response_start = fourth_pulse_end + mem_gap
        if delta1 < delta2:
            y_train[sample, response_start:, 0] = 1.0 * pulse_signe  # Positivo
        else:
            y_train[sample, response_start:, 0] = -1.0 * pulse_signe  # Negativo

    x_train = x_train  # + stim_noise * np.random.randn(size, sequence_duration, 1)

    print(f"--- {time.time() - start_time: .2f} seconds ---")
    return x_train, y_train, sequence_duration, trial_types


# Generate data set
x_train, y_train, seq_dur, trial_types = generate_trials(10,20)

# Graficar
fig = plt.figure(figsize=(8, 10))
fig.suptitle("Comparison of two intervals (Delta1 y Delta2)", fontsize=12)

for ii in range(10):
    plt.subplot(5, 2, ii + 1)
    x_signal = x_train[ii, :, 0]
    y_signal = y_train[ii, :, 0]

    plt.plot(x_signal, color='g', label="Input")
    plt.plot(y_signal, color='gray', linewidth=2, label=f"Response (Type: {trial_types[ii]})")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize=7, loc='upper right')
    plt.xticks(np.arange(0, seq_dur + 50, 100), fontsize=7)

plt.tight_layout()
plt.savefig("dataSet_comparacion_intervals.png", dpi=200, bbox_inches='tight')
plt.show()
