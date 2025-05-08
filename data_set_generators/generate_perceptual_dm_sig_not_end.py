"""
Perceptual Decision-Making with Sustained Stimulus (Fixed Integration time Decision-Making)
MIT License - C. Jarne V. 1.0 - 2024

This script generates training data for decision tasks requiring integration of initial stimulus
segments while maintaining responses during ongoing input. Outputs depend on early stimulus
integration but must persist through subsequent input variations.

Key Features:
- Partial integration window: Decision based on first 50ms of variable-length stimulus
- Sustained complexity: Stimulus continues 150ms post-decision
- Temporal variability: 0-150ms stimulus onset jitter
- Dynamic responses: 10ms decision delay after integration period
- Noise profile: Hann-window shaped Gaussian noise (σ=1)

Parameters:
- size: Number of trials
- mem_gap: Integration window duration (50ms)
- var_delay_length: Maximum stimulus onset jitter (150ms)
- stim_noise: Noise magnitude multiplier

Technical Implementation:
1. Input Generation:
   - Gaussian noise shaped by 10-sample Hann window
   - Total stimulus duration: mem_gap + 150ms
   - Variable onset delay (0-150ms)

2. Output Logic:
   - Integration window: First mem_gap ms after stimulus onset
   - Decision timing: 10ms post-integration period
   - Response maintenance: Sustained through remaining 150ms stimulus

3. Temporal Structure:
   - Initial quiet period: 50ms
   - Stimulus phase: 200ms (50ms integration + 150ms continuation)
   - Response phase: 210ms+
   - Total trial length: 460ms

4. Reproducibility: Seeded randomization (NumPy seed=2)

Outputs:
- x_train: Sustained stimuli [samples × 460ms × 1 channel]
- y_train: Decisions [samples × 460ms × 1 channel] (±1)
- seq_dur: Fixed sequence duration
- PNG plot: 10 samples showing decision persistence during ongoing input

Design Notes:
- Cognitive load: Requires ignoring post-decision stimulus content
- Integration precision: Fixed 50ms analysis window despite onset jitter
- Output stability: Maintains sign(∫) decision through noisy input
- Experimental rigor: Tests memory maintenance under continuing distraction
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy 
from scipy import signal
from numpy.random import seed
from scipy import signal

start_time = time.time()


def generate_trials(size, mem_gap):
    seed(2)
    # mem_gap = 200 # output reaction length
    first_in = 50   # time to start the first stimulus
    stim_dur = mem_gap  # 100  #stimulus duration
    stim_noise = 1  # 0.03 # noise
    var_delay_length = 150    # change for a variable length stimulus
    out_gap = 250-20  # -60#50 #how much length add to the sequence duration
    sample_size = size  # sample size
    
    dm_seed_A = np.array([[1], [1]])  # np.array([[0], [1]])
    seq_dur = first_in+stim_dur+mem_gap+var_delay_length+(out_gap-mem_gap)
    win = scipy.signal.windows.hann(10)
 
    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1

    trial_types = np.random.randint(2, size=sample_size)
    x_train_ = np.zeros((sample_size, seq_dur, 1))
    x_train = np.zeros((sample_size, seq_dur, 1))
    y_train = 0 * np.ones((sample_size, seq_dur, 1))

    for ii in np.arange(sample_size): 
        x_train_[ii, first_in + var_delay[ii]:first_in + stim_dur + var_delay[ii]+mem_gap+100, 0] = \
            dm_seed_A[trial_types[ii], 0] * stim_noise * np.random.randn(stim_dur+mem_gap+100)
        x_train[ii, first_in + var_delay[ii]:first_in + stim_dur + var_delay[ii]+mem_gap+100, 0] = \
            signal.convolve(x_train_[ii, first_in+var_delay[ii]:first_in + stim_dur + var_delay[ii]+mem_gap+100, 0]
                            , win, mode='same') / sum(win)
        y_train[ii, first_in + stim_dur + var_delay[ii]+mem_gap+10:, 0] = \
            np.sign(scipy.integrate.simps(x_train[ii, first_in+var_delay[ii]:first_in + stim_dur+var_delay[ii], 0],
                                          x=None, dx=1, axis=-1, even='avg'))

    x_train = x_train  # + 0.01 * np.random.randn(sample_size, seq_dur, 1)
    y_train = y_train  # + stim_noise * np.random.randn(sample_size, seq_dur, 1)
   
    print("--- %s seconds to generate learning dataset---" % (time.time() - start_time))
    return x_train, y_train, seq_dur


# Generate a sample dataset

sample_size = 10
x_train, y_train, seq_dur = generate_trials(sample_size, 50)

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
    plt.xticks(np.arange(0, seq_dur+50, 100), fontsize=8)

fig_name = "dataset_perceptual_dm_int_persistent_input.png"
plt.savefig(fig_name, dpi=200)
plt.show()


