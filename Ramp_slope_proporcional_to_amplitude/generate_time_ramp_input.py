############################################
#                                          #
#     A "perceptual DM" data set generator #
#            of samples                    #
#     with adjutable parameters            #
#                                          #
#   Mit License C. Jarne V. 1.0 2020       #
#                                          #
############################################

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy 
from scipy import signal
from numpy.random import seed
from scipy import signal

start_time = time.time()

def generate_trials(size,amplitude):
    seed(2)
    mem_gap          = 20 # output reaction length
    first_in         = 50#time to start the first stimulus  
    stim_dur         = 100  #stimulus duration
    stim_noise       = 1#0.03 #noise
    var_delay_length = 0    #change for a variable length stimulus
    out_gap          = 300-20#-60#50 #how much lenth add to the sequence duration   
    sample_size      = size # sample size
    rec_noise        = 0
    
    xor_seed_A = np.array([[0],[1]])
    
    xor_y            = np.array([0,1])
    seq_dur          = first_in+stim_dur+mem_gap+var_delay_length+(out_gap-mem_gap)
    win              = signal.hann(10)
 
    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=np.int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1
    second_in = first_in + stim_dur + mem_gap
    
    out_t          = mem_gap+ first_in+stim_dur
    x              = np.arange(seq_dur)
    trial_types    = np.random.randint(2, size=sample_size)
    x_train_       = np.zeros((sample_size, seq_dur, 1))    
    x_train        = np.zeros((sample_size, seq_dur, 1))        
    y_train        = np.zeros((sample_size, seq_dur, 1))
 
    
    
    #amplitudes= 0.1
    ramp_dur = mem_gap*1/amplitude
    
    for ii in np.arange(sample_size): 
        x_train[ii, first_in:, 0]=xor_seed_A[trial_types[ii], 0]*amplitude
       #ramp definition:
        ramp = np.linspace(0, 1, num=int(ramp_dur), endpoint=False)
        y_train[ii, int(out_t + var_delay[ii]):int(out_t + var_delay[ii] + ramp_dur), 0] = ramp*1*xor_seed_A[trial_types[ii], 0] 
        y_train[ii, int(out_t + var_delay[ii] + ramp_dur):, 0] = 1*xor_seed_A[trial_types[ii], 0]

    mask = np.zeros((sample_size, seq_dur))
    for sample in np.arange(sample_size):
        mask[sample,:] = [1 for y in y_train[sample,:,:]]
       
    #x_train = x_train + stim_noise * np.random.randn(sample_size, seq_dur, 1)
    y_train = y_train# + stim_noise * np.random.randn(sample_size, seq_dur, 1)
   
    print("--- %s seconds to generate learning dataset---" % (time.time() - start_time))
    return(x_train, y_train,mask,seq_dur)


       
#To test the rule that you want to teach

sample_size=10
amplitude=0.2
x_train,y_train, mask,seq_dur = generate_trials(sample_size,amplitude) 

fig     = plt.figure(figsize=(6,8))
fig.suptitle("Data Set Training Samples\n (amplitude in arb. units time in mS)",fontsize = 20)
for ii in np.arange(10):
    plt.subplot(5, 2, ii + 1)    
    plt.plot(x_train[ii, :, 0],color='g',label="Input")
    plt.plot(y_train[ii, :, 0],color='gray',linewidth=2,label="Expected Output")
    plt.ylim([-2.5, 2.5])
    plt.legend(fontsize= 5,loc=3)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
figname = "data_set_perceptual_dm.png" 
plt.savefig(figname,dpi=200)
plt.show()
 
    


