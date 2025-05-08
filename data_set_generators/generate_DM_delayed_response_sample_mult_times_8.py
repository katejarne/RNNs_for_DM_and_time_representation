"""
Multi-interval Amplitude-based Decision-Making Task Generator with Latency Coding (8 intervals)
MIT License - C. Jarne V. 1.0 - 2024

This script generates training data for precise temporal coding tasks where 8 input pulse heights
(1-8) map to 8 distinct response latencies (25-200ms in 25ms steps). Outputs maintain input polarity
while encoding timing information through response delay.

Key Features:
- 8:1 temporal resolution: Linear height-latency mapping (Height×25ms)
- Polarity preservation: ± inputs produce ± outputs with matched timing
- Temporal variability: Stimulus jitter (0-70ms) + response timing noise
- Input conditioning: Hann-window shaped pulses (20ms) + Gaussian noise (σ=0.05)
- Adaptive visualization: y-axis scaled to [-9,9] for pulse height 8

Parameters:
- size: Number of trials (samples)
- mem_gap: Fixed processing delay (0ms in demo)
- stim_dur: Constant stimulus duration (20ms)
- var_delay_length: Maximum onset jitter (70ms)

Technical Implementation:
1. Input Processing:
   - 8-level pulse heights (1-8) with random ± polarity
   - 10-sample Hann window smoothing
   - Temporal jitter: Stimulus onset variability (1-70ms)
   - Additive noise: N(0, 0.05)

2. Output Logic:
   - Binary responses (±1) with latency coding
   - Height-to-delay mapping: {1:25ms, 2:50ms, ..., 8:200ms}
   - Sustained response from latency point to trial end

3. Temporal Structure:
   - Initial fixation: 50ms
   - Stimulus phase: 20ms ± jitter
   - Response phase: 105-280ms (delay-dependent)
   - Total trial length: 370ms

4. Reproducibility: Fixed seed (NumPy seed=1)

Outputs:
- x_train: Noisy inputs [samples × 370ms × 1 channel] (max|8|)
- y_train: Time-coded outputs [samples × 370ms × 1 channel] (±1)
- sequence_duration: Fixed trial length
- PNG plot: 10-sample visualization showing latency progression

Design Notes:
- Inverse time-amplitude relationship: Higher inputs → Later responses
- Constant response magnitude: All outputs ±1 regardless of input height
- Non-overlapping responses: Max delay (200ms) ensures 170ms response duration
- Input scaling: Pulse heights (1-8) directly affect input amplitude range
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
    out_gap = 250 - 20
    xor_seed_A = np.array([[0], [1]])
    sequence_duration = first_in + stim_dur + mem_gap + var_delay_length + (out_gap - mem_gap)
    win = signal.windows.hann(10)

    var_delay = np.zeros(size, dtype=int) if var_delay_length == 0 \
        else np.random.randint(var_delay_length, size=size) + 1
    out_t = mem_gap + first_in + stim_dur

    trial_types = np.random.randint(2, size=size)
    pulse_heights = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=size)
    pulse_signes = np.random.choice([-1, 1], size=size)

    height_to_time = {1: 25, 2: 50, 3: 75, 4: 100, 5: 125, 6: 150, 7: 175, 8: 200}

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
        # else output remains in 0

    # Adding input noise
    x_train = x_train + stim_noise * np.random.randn(size, sequence_duration, 1)

    print("--- %s seconds to generate learning dataset---" % (time.time() - start_time))

    return x_train, y_train, sequence_duration


# Generate dataset
x_train, y_train, seq_dur = generate_trials(10, 0)

# Plot results
fig = plt.figure(figsize=(6, 8))
fig.suptitle("10 Training Samples\n (amplitude in arb. units, time in ms)", fontsize=20)
for ii in range(10):
    plt.subplot(5, 2, ii + 1)
    plt.plot(x_train[ii, :, 0], color='g', label="Input")
    plt.plot(y_train[ii, :, 0], color='gray', linewidth=2)
    plt.ylim([-9, 9])
    plt.legend(fontsize=5, loc=3)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xticks(np.arange(0, seq_dur + 50, 100), fontsize=8)

fig_name = "dataset_with_8_response_times.png"
plt.savefig(fig_name, dpi=200)
plt.show()
