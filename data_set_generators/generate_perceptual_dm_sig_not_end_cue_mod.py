"""
Cue-Modulated Perceptual Decision Task Generator (Cued Integration time Decision-Making)
MIT License - C. Jarne V. 1.0 - 2024

This script generates training data for cue-dependent decision tasks where outputs depend
on both stimulus integration and contextual cues. Features conditional response logic with
enforced 3:1 ratio of decision vs neutral trials.

Key Features:
- Dual-channel input: Stimulus (Channel 0) + Binary cue (Channel 1)
- Conditional processing: Cue presence determines response requirement
- Sample balancing: 75% decision trials (±1), 25% neutral trials (0)
- Temporal structure: 20ms pre-cue window + variable stimulus onset (0-150ms jitter)
- Dynamic integration: 20ms stimulus segment analysis

Parameters:
- size: Total trials
- mem_gap: Integration window duration (20ms)
- positive_negative_ratio: Decision vs neutral trial ratio (3:1)
- var_delay_length: Maximum stimulus onset jitter (150ms)

Technical Implementation:
1. Input Generation:
   - Channel 0: Hann-window shaped Gaussian noise (σ=1)
   - Channel 1: 10ms cue pulse (-20:-10ms pre-stimulus)
   - Stimulus duration: 20ms (mem_gap) + 100ms continuation

2. Output Logic:
   - Cue Present: Output = sign(∫stimulus) maintained post 10ms delay
   - Cue Absent: Sustained zero output
   - Response lock: 10ms post-integration period

3. Temporal Structure:
   - Cue phase: 10ms pulse at -20:-10ms
   - Stimulus phase: 20ms integration + 100ms continuation
   - Response phase: 130ms+
   - Total trial length: 340ms

4. Reproducibility: Seeded randomization (NumPy seed=3)

Outputs:
- x_train: Dual-channel inputs [samples × 340ms × 2 channels]
- y_train: Cue-dependent decisions [samples × 340ms × 1 channel]
- seq_dur: Fixed sequence duration
- PNG plot: 10 samples showing cue-stimulus-response relationships

Design Notes:
- Contextual dependency: Cue gates integration requirement
- Anti-bias enforcement: Strict 3:1 active/inactive trial ratio
- Temporal precision: Cue precedes stimulus by 20ms
- Noise continuity: Stimulus continues 5× beyond integration window
- Channel segregation: Explicit cue channel vs noisy stimulus channel
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
from numpy.random import seed
from scipy import signal

start_time = time.time()


def generate_trials(size, mem_gap, positive_negative_ratio=3):
    seed(3)
    # Parameters
    first_in = 50  # time for firs stimulus
    stim_dur = mem_gap
    stim_noise = 1
    var_delay_length = 150
    out_gap = 250 - 20
    seq_dur = first_in + stim_dur + mem_gap + var_delay_length + (out_gap - mem_gap)
    win = scipy.signal.windows.hann(10)

    var_delay = np.random.randint(var_delay_length, size=size) + 1 if var_delay_length != 0 \
        else np.zeros(size, dtype=int)
    x_train_ = np.zeros((size, seq_dur, 2))
    x_train = np.zeros((size, seq_dur, 2))
    y_train = np.zeros((size, seq_dur, 1))

    decision_count = 0
    zero_output_count = 0
    max_decision_samples = int(size * (positive_negative_ratio / (positive_negative_ratio + 1)))
    max_zero_output_samples = size - max_decision_samples

    ii = 0
    while ii < size:
        trial_type = np.random.randint(2)  # stimulus type A or B
        trial_type_2 = np.random.randint(2)  # with or without "cue"

        if trial_type_2 == 1 and decision_count < max_decision_samples:
            # Cases with positive or negative decision
            x_train_[ii, first_in + var_delay[ii]:first_in + stim_dur + var_delay[ii] + mem_gap + 100, 0] = \
                stim_noise * np.random.randn(stim_dur + mem_gap + 100)
            x_train[ii, first_in + var_delay[ii]:first_in + stim_dur + var_delay[ii] + mem_gap + 100, 0] = \
                signal.convolve(x_train_[ii, first_in + var_delay[ii]:first_in + stim_dur + var_delay[ii] + mem_gap
                                                                      + 100, 0], win, mode='same') / sum(win)
            x_train[ii, first_in + var_delay[ii] - 20:first_in + var_delay[ii] - 10, 1] = 1  # Cue presente
            y_train[ii, first_in + stim_dur + var_delay[ii] + mem_gap + 10:, 0] = \
                np.sign(scipy.integrate.simps(x_train[ii, first_in + var_delay[ii]:first_in
                                                                                   + stim_dur + var_delay[ii], 0]))
            decision_count += 1
            ii += 1
        elif trial_type_2 == 0 and zero_output_count < max_zero_output_samples:
            # zero output cases
            x_train_[ii, first_in + var_delay[ii]:first_in + stim_dur + var_delay[ii] + mem_gap + 100, 0] = \
                stim_noise * np.random.randn(stim_dur + mem_gap + 100)
            x_train[ii, first_in + var_delay[ii]:first_in + stim_dur + var_delay[ii] + mem_gap + 100, 0] = \
                signal.convolve(x_train_[ii, first_in + var_delay[ii]:first_in + stim_dur + var_delay[ii]
                                                                      + mem_gap + 100, 0], win, mode='same') / sum(win)
            x_train[ii, first_in + var_delay[ii] - 20:first_in + var_delay[ii] - 10, 1] = 0  # No cue
            y_train[ii, first_in + stim_dur + var_delay[ii] + mem_gap + 10:, 0] = 0
            zero_output_count += 1
            ii += 1

    print(f"Generated {decision_count} decision samples and {zero_output_count} zero-output samples.")
    return x_train, y_train, seq_dur

# Data set example
sample_size = 100
x_train, y_train, seq_dur = generate_trials(sample_size, 20)

fig = plt.figure(figsize=(6, 8))
fig.suptitle("Data Set Training Samples\n (amplitude in arb. units time in ms)", fontsize=20)
for ii in np.arange(10):
    plt.subplot(5, 2, ii + 1)
    plt.plot(x_train[ii, :, 0], color='g', label="Input")
    plt.plot(x_train[ii, :, 1], color='pink', label="Cue")
    plt.plot(y_train[ii, :, 0], color='gray', linewidth=2, label="Expected Output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize=5, loc=3)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xticks(np.arange(0, seq_dur + 50, 100), fontsize=8)

fig_name = "dataset_perceptual_dm_int_cue.png"
plt.savefig(fig_name, dpi=200)
plt.show()
