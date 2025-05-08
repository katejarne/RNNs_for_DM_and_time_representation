# plot_neural_activity.py
"""
Print activity from trained networks  with stimuli
MIT License - C. Jarne V. 1.0 - 2024

Code defines a function `plot_sample` that visualizes neural network activity h_i(t) and performs
frequency analysis for a specific input sample. The code generates various plots to display the network’s
response to input data.
The function takes several parameters including `sample_number`, the index of the sample to analyze,
`neurons`, the number of units in the network layer, and `model`, a trained neural network model.
It first extracts and visualizes the activations of each neuron within a specified layer, converting
these activations to a format suitable for dimensionality reduction via PCA. This allows the
function to project high-dimensional data to 3D spaces, displaying it in subplots for better
interpretability of patterns and trends within neural activity.

The visualizations include:
1. Time-series plots for input, output, and target sequences, illustrating the network’s performance.
2. Individual neuron activation plots, overlaid with colors for distinction.
3. A 3D PCA plot of the neural states, capturing global relationships in activity trajectories.

The code includes frequency and phase analysis on each neuron's activation, utilizing
Fourier transforms to identify dominant frequency components. Local minima in the activations are
also found to estimate peak-to-peak frequencies, helping to quantify the temporal dynamics of each
neuron. Finally, these metrics (dominant frequency, phase, and peak-to-peak frequency) are stored
and displayed on a dedicated plot for comparison across neurons.

"""
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def plot_sample(sample_number, neurons, x_train, y_train, model, seq_dur, i, plot_dir, f, string_name,
                mem_gap):

    frequencies = []
    
    seq_dur = len(x_train[sample_number, :, 0])
    test = x_train[sample_number:sample_number+1, :, :]
    colors = plt.cm.rainbow(np.linspace(0, 1, neurons+1))

    # Status for the activity value at the indicated layer
    layer = 0

    # First layer:
    layer_outputs = model.layers[layer](test)
    layer_output = layer_outputs[layer]
    tensor_np = layer_output.numpy()
    # print(layer_output)
    layer_output_T = tensor_np.T

    # print("Output layer", layer_output_T)
    array_red_list = []
    y_pred = model.predict(test)

    for ii in np.arange(0, neurons, 1):
        neurona_serie = np.reshape(layer_output_T[ii], len(layer_output_T[ii]))
        array_red_list.append(neurona_serie)
    
    array_red = np.asarray(array_red_list)
    # sdv_3d = sklearn.decomposition.TruncatedSVD(n_components=3)

    # X_3d = sdv_3d.fit_transform(array_red.T)
    pca = PCA(n_components=3)
    X_pca_ = pca.fit(array_red)
    X_pca = pca.components_

    print("------------")
    pca_axis_x = X_pca[0]
    pca_axis_y = X_pca[1]
    pca_axis_z = X_pca[2]

    # How many 3d angular views you want to define (to make video series)
    yy = np.arange(70, 80, 10)

    kk = 70  # to fix a particular angular view

    fig = plt.figure(figsize=cm2inch(19, 7))
    plt.subplot(2, 2, 1) 
    plt.plot(test[0, :, 0], color='g', label='Input')
    if test.shape[2] == 2:
        plt.plot(test[0, :, 1], color='pink', label='Input')
    # plt.plot(test[0, :, 1],color='pink',label='Input Context')
    plt.plot(y_train[sample_number, :, 0], color='grey', linewidth=3, label='Target Output')
    plt.plot(y_pred[0, :, 0], color='r', linewidth=2, label=' Output')
    plt.xlim(0, seq_dur+1)
    plt.ylim([-2.5, 2.5])
    plt.yticks([])
    plt.xticks(np.arange(0, seq_dur+1, 50), fontsize=8)
    plt.legend(fontsize=5, loc=1)

    ### 
    plt.subplot(2, 2, 3) 
    plt.plot(test[0, :, 0], color='g', label='Input')
    # plt.plot(test[0, :, 1],color='pink',label='Input B')

    for ii in np.arange(0, int(neurons), 1):
        if ii == 0 or ii == int(neurons)-1:
            plt.plot(layer_output_T[ii], color="black", linewidth=1)
        else:
            plt.plot(layer_output_T[ii], color=colors[ii], linewidth=1, alpha=0.35)
        plt.xlim(-1, seq_dur+1)
        plt.ylim([-2.5, 2.5])
        plt.xlabel('time [ms]', fontsize=10)
        plt.ylabel('Amplitude arb. units', fontsize=10)
        plt.yticks([])
        plt.xticks(np.arange(0, seq_dur+1, 50), fontsize=8)
    plt.plot(y_pred[0, :, 0], color='r', linewidth=2, label=' Output\n Some individual states')
    leg = plt.legend(fontsize=5, loc=4)

    # plt.subplot(2, 2, 4)
    fig.suptitle(r"Activity of the units ($h_i(t)$ time series) - PCA 3D plot", fontsize=12)
    ax = fig.add_subplot(122, projection='3d')

    x = X_pca[0]
    y = X_pca[1]
    z = X_pca[2]
    N = len(z)
    
    for ik in range(N-1):
        ax.plot(x[ik:ik+2], y[ik:ik+2], z[ik:ik+2], color=plt.cm.viridis(ik/N))
    ax.scatter(pca_axis_x[0], pca_axis_y[0], pca_axis_z[0], s=70,c='r', marker="^", label=' Start ')
    ax.scatter(pca_axis_x[-1], pca_axis_y[-1], pca_axis_z[-1], s=70, c='b', marker="^", label=' Stop ')

    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_zticks(())
    ax.view_init(elev=10, azim=kk)

    ax.legend(fontsize=6)
    figname = str(plot_dir)+"/sample_"+str(sample_number)+"_pca_3D_"+str(layer) + \
              "_individual_neurons_state_"+str(i)+'_'+str(kk)+"_"+str(f)+"_"+str(string_name)+".png"
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if sample_number == 3 or sample_number == 4:
        data_series = tensor_np  # np.random.rand(200, 100)
        output_folder = 'frames'

        # Crear la carpeta si no existe
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_filename = str(string_name)+'sample_'+str(sample_number)+'_combined_animation.gif'
        stimulus= test[0, :, 0]
        out_signal= y_pred[0, :, 0]

    else:
        pass

    # freq. estimation Figure

    colors_short = plt.cm.rainbow(np.linspace(0, 1, 6))
    fig = plt.figure(figsize=cm2inch(13, 8))
    fft_freqs = []
    phases = []
    amplitudes = []
    for ii in np.arange(0, int(neurons), 1):

        h_i_time_serie = layer_output_T[ii]
        t_ = np.arange(len(h_i_time_serie))
        freq = 0

        peakind_min = signal.argrelmin(h_i_time_serie, axis=0, order=10)
        peakind_min = peakind_min[:6]
        amp_peak_min_x = h_i_time_serie[peakind_min]

        print("Position of the minimum: ", peakind_min)
        print("Amplitude at the minimum:", amp_peak_min_x)
        # print(amp_peak_min_x[3])
        if len(amp_peak_min_x)>4:
            amplitudes.append(abs(amp_peak_min_x[3]))
        else:
            amplitudes.append(abs(0))
        # FFT
        fourier_transform = np.fft.fft(h_i_time_serie[0:150])
        # Each component of FFT
        freqs = np.fft.fftfreq(len(h_i_time_serie[0:150]), d=0.001)
        # Dominant component
        dominant_freq_index = np.argmax(np.abs(fourier_transform))
        dominant_freq = freqs[dominant_freq_index]
        print("Dominant frequency:", dominant_freq)
        fft_freqs.append(abs(dominant_freq))
        # Signal phase
        phase = np.angle(fourier_transform[dominant_freq_index])
        print("Phase:", phase)
        phases.append(phase)

        if len(amp_peak_min_x) > 1:
            pepe = t_[peakind_min]
            # print("pepe", pepe)
            if len(pepe) > 6:
                freq = 1/(0.001*float(pepe[5]-pepe[4]))
        else:
            freq = 0

        ff = '%.2f' % freq
        print("frequency", ff)
        frequencies.append(freq)
        if ii < 5:
            plt.plot(h_i_time_serie, color=colors_short[ii], linewidth=1, alpha=0.5,
                     label="Freq= "+str(ff))
            plt.scatter(peakind_min, amp_peak_min_x, c=colors_short[ii], alpha=0.45)
        if ii == 0:
            plt.plot(h_i_time_serie, color="k", linewidth=1,
                     label="Freq= "+str(ff))
            """
            plt.vlines(x=peakind_min[5], ymin=amp_peak_min_x[5], ymax=0,
                           color="k", linestyle='--')
            plt.vlines(x=peakind_min[4], ymin=amp_peak_min_x[4], ymax=0,
                           color="k", linestyle='--')
            plt.scatter(peakind_min[5], amp_peak_min_x[5], c="b", marker="^")
            plt.scatter(peakind_min[4], amp_peak_min_x[4], c="b", marker="^")
            """
        else:
            pass
    plt.xlim(-1, seq_dur+1)
    plt.ylim([-1.5, 1.5])
    plt.xlabel('time [ms]', fontsize=10)
    plt.ylabel('amplitude [arb. units]', fontsize=10)
    plt.yticks([])
    plt.xticks(np.arange(0, seq_dur+1, 50), fontsize=8)
    plt.plot(test[0, :, 0], color='g', label='Input A')
    plt.plot(y_pred[0, :, 0], color='r', linewidth=2, label=r' Output: 5 ind. $h(t)$ states')
    # plt.legend(fontsize= 3.5,loc=3)
    leg = plt.legend(fontsize=5, loc=3)
    # leg.get_frame().set_linewidth(0.0)

    figname = str(plot_dir)+"/S_"+str(sample_number)+"_pca_3D_"+str(layer) +\
              "_neurons_states_"+str(i)+'_'+str(kk)+"_"+str(f)+"_"+str(string_name)+".png"
    # plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if len(frequencies) > 4:
        print("Frequency estimated from Units activity \n(estimated per input)", frequencies)
        # media= np.average(frequencies)

    plt.figure(figsize=(7, 6))
    plt.title('Histogram for units', fontsize=18)
    mean = np.mean(frequencies)

    plt.axvline(x=mean, color="red", linestyle='--')
    mean_s = '%.2f' % mean
    histo, bins = np.histogram(frequencies)
    mode = bins[np.argmax(histo)]
    mode_s = '%.2f' % mode
    fft_freqs = np.array(fft_freqs)
    fft_freqs_ = fft_freqs[fft_freqs<100]
    mean_ft = np.mean(fft_freqs_)
    mean_fts = '%.2f' % mean_ft
    plt.hist(frequencies, bins=100, color="gray", alpha=0.35,
             label="Frequencies \n"+"Mean value: "+str(mean_s)+"\nMode: "+str(mode_s))
    plt.hist(fft_freqs, bins=100, range=[0, 100], color="pink", alpha=0.35,
             label="Frequencies fft \n"+"Mean value: "+str(mean_fts))

    plt.xlabel('frequency  [1/s]', fontsize=16)

    leg = plt.legend(fontsize=12, loc=1)
    leg.get_frame().set_linewidth(0.0)
    # plt.savefig(plot_dir+"/freq_histo_"+str(sample_number)+'_'
    #            +str(i)+"_"+str(f)+'_'+str(string_name)+"_0.png",dpi=300, bbox_inches='tight')
    plt.close(fig)

    plt.figure(figsize=(7, 6))
    plt.title('Histogram for units phases', fontsize=18)
    mean_p = np.mean(phases)
    mean_ps = '%.2f' % mean_p
    plt.hist(phases, bins=100, color="gray", alpha=0.35,
             label="Phases \n"+"Mean value: "+str(mean_ps))
    # plt.savefig(plot_dir+"/phase_histo_"+str(sample_number)+'_'
    #            +str(i)+"_"+str(f)+'_'+str(string_name)+"_0.png",dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    plt.title('Histogram for units amplitudes', fontsize=18)
    mean_ap = np.mean(amplitudes)
    mean_as = '%.2f' % mean_ap
    plt.hist(amplitudes, bins=20, color="green", alpha=0.35,
             label="Phases \n"+"Mean value: "+str(mean_as))
    # plt.savefig(plot_dir+"/amplitudes_histo_"+str(sample_number)+'_'
    #            +str(i)+"_"+str(f)+'_'+str(string_name)+"_0.png",dpi=300, bbox_inches='tight')
    plt.close(fig)

    return frequencies
