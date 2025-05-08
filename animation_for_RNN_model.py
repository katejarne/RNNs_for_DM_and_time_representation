# animation_for_RNN_model.py
"""
MIT License - C. Jarne V. 1.0 - 2025
This code generates an animated visualization comparing
neural network dynamics for opposing stimuli (positive/negative)
in a perceptual decision-making task. It creates synchronized
multi-panel views showing: 3D PCA trajectories of population
activity, temporal activation patterns of individual neurons,
and real-time network outputs alongside inputs.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from utils.net_constraint_create import *

# data set generators
from data_set_generator_for_animation.data_set_animation_DM_integral import *
# from data_set_generator_for_animation.data_set_animation_DM_simple import *


def load_model(model_path):

    model = tf.keras.models.load_model(model_path,compile=False)
    """
    model = tf.keras.models.load_model(model_path,  custom_objects={'NonNegLast':NonNegLast,
                                                     'NonNegLast_input':NonNegLast_input,
                                                     'my_init_exi_ini' : my_init_exi_ini,'my_init_rec':my_init_rec},
                                       compile=False)
    """
    return model


"""

def custom_simple_rnn(**config):
    if 'time_major' in config:
        del config['time_major']  # Eliminar el argumento no reconocido
    return tf.keras.layers.SimpleRNN(**config)

def load_model(model_path):
    custom_objects = {
        'NonNegLast': NonNegLast,
        'NonNegLast_input': NonNegLast_input,
        'my_init_exi_ini': my_init_exi_ini,
        'my_init_rec': my_init_rec,
        'SimpleRNN': custom_simple_rnn  # Usa la función personalizada
    }

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    return model
"""


def process_stimulus(model, stimulus):
    print(stimulus.shape)
    output = model.predict(stimulus)
    layer_output = model.layers[0](stimulus)
    tensor_np = layer_output.numpy()
    return output, tensor_np


def generate_inputs_for_animation(model, stimulus_pos, stimulus_neg):
    output_pos, data_series_pos = process_stimulus(model, stimulus_pos)
    output_neg, data_series_neg = process_stimulus(model, stimulus_neg)
    return data_series_pos, data_series_neg, output_pos, output_neg


def update(frame, data_series_pos, data_series_neg, circles_pos, circles_neg, norm, title):
    # Update colors for the positive stimulus
    data_pos = data_series_pos[frame]
    colors_pos = plt.cm.Spectral(norm(data_pos))
    for i, circle in enumerate(circles_pos):
        circle.set_color(colors_pos[i])

    # Update colors for the negative stimulus
    data_neg = data_series_neg[frame]
    colors_neg = plt.cm.Spectral(norm(data_neg))
    for i, circle in enumerate(circles_neg):
        circle.set_color(colors_neg[i])
    title.set_text(f'RNN Units - Time step {frame}')
    return circles_pos + circles_neg


def update_all_plots(frame, x_pos, y_pos, z_pos, x_neg, y_neg, z_neg,
                     ax_3d_pos, ax_3d_neg, line_3d_pos, dot_3d_pos, line_3d_neg, dot_3d_neg,
                     ax_2d_pos, ax_2d_neg, line_2d_pos, dot_2d_pos, line_2d_neg, dot_2d_neg,
                     data_series_pos, data_series_neg, circles_pos, circles_neg, norm, title,fig):
    fig.suptitle(f'RNN Units - Time step {frame}')
    # Update the positive stimulus 3D plot
    line_3d_pos.set_data(x_pos[:frame+1], y_pos[:frame+1])
    line_3d_pos.set_3d_properties(z_pos[:frame+1])
    dot_3d_pos.set_data([x_pos[frame]], [y_pos[frame]])
    dot_3d_pos.set_3d_properties([z_pos[frame]])

    # Update the negative stimulus 3D plot
    line_3d_neg.set_data(x_neg[:frame+1], y_neg[:frame+1])
    line_3d_neg.set_3d_properties(z_neg[:frame+1])
    dot_3d_neg.set_data([x_neg[frame]], [y_neg[frame]])
    dot_3d_neg.set_3d_properties([z_neg[frame]])

    # Update the 2D plot for positive stimulus
    line_2d_pos.set_data(np.arange(frame+1), data_series_pos[:frame+1, 0])
    dot_2d_pos.set_data(frame, data_series_pos[frame, 0])

    # Update the 2D plot for negative stimulus
    line_2d_neg.set_data(np.arange(frame+1), data_series_neg[:frame+1, 0])
    dot_2d_neg.set_data(frame, data_series_neg[frame, 0])

    # Update the grid of circles for both stimuli
    update(frame, data_series_pos, data_series_neg, circles_pos, circles_neg, norm, title)

    return (line_3d_pos, dot_3d_pos, line_3d_neg, dot_3d_neg,
            line_2d_pos, dot_2d_pos, line_2d_neg, dot_2d_neg,
            *circles_pos, *circles_neg, title)


def create_combined_animation(data_series_pos, data_series_neg, stimulus_pos, stimulus_neg,
                              out_signal_pos, out_signal_neg,
                              pca_axis_x_pos, pca_axis_y_pos, pca_axis_z_pos,
                              pca_axis_x_neg, pca_axis_y_neg, pca_axis_z_neg,
                              output_folder, output_filename):

    fig = plt.figure(figsize=(20, 10))
    title = fig.suptitle(f'RNN Units - Time step 0', fontsize=16)

    # Subplot 1: 3D PCA for positive stimulus
    ax_3d_pos = fig.add_subplot(247, projection='3d')
    # ax_3d_pos.axis('off')
    ax_3d_pos.plot(pca_axis_x_pos, pca_axis_y_pos, pca_axis_z_pos, color='salmon', alpha=0.5)
    dot_3d_pos, = ax_3d_pos.plot([pca_axis_x_pos[0]], [pca_axis_y_pos[0]], [pca_axis_z_pos[0]], 'ko')
    line_3d_pos, = ax_3d_pos.plot([], [], [], 'k', alpha=0.8)
    ax_3d_pos.scatter(pca_axis_x_pos[0], pca_axis_y_pos[0], pca_axis_z_pos[0], s=70, c='r', marker="^", label='Start')
    ax_3d_pos.scatter(pca_axis_x_pos[-1], pca_axis_y_pos[-1], pca_axis_z_pos[-1], s=70, c='b', marker="^", label='Stop')

    # Subplot 2: 3D PCA for negative stimulus
    ax_3d_neg = fig.add_subplot(245, projection='3d')
    # ax_3d_neg.axis('off')
    ax_3d_neg.plot(pca_axis_x_neg, pca_axis_y_neg, pca_axis_z_neg, color='salmon', alpha=0.5)
    dot_3d_neg, = ax_3d_neg.plot([pca_axis_x_neg[0]], [pca_axis_y_neg[0]], [pca_axis_z_neg[0]], 'ko')
    line_3d_neg, = ax_3d_neg.plot([], [], [], 'k', alpha=0.8)
    ax_3d_neg.scatter(pca_axis_x_neg[0], pca_axis_y_neg[0], pca_axis_z_neg[0], s=70, c='r', marker="^", label='Start')
    ax_3d_neg.scatter(pca_axis_x_neg[-1], pca_axis_y_neg[-1], pca_axis_z_neg[-1], s=70, c='b', marker="^", label='Stop')

    # Subplot 3: 2D plot for positive stimulus
    ax_2d_pos = fig.add_subplot(221)
    ax_2d_pos.axis('off')
    colors = cm.rainbow(np.linspace(0, 1, 100+1))
    for ii in np.arange(100):
        ax_2d_pos.plot(data_series_pos[:, ii], color=colors[ii], linewidth=1, alpha=0.35)
    ax_2d_pos.plot(data_series_pos[:, 0], color='k')
    print("aca", stimulus_pos.shape)
    ax_2d_pos.plot(stimulus_pos[0, :], color="g", linewidth=2, label="Input")
    ax_2d_pos.plot(out_signal_pos[0, :], color="r", linewidth=2, label="Output")
    dot_2d_pos, = ax_2d_pos.plot(0, data_series_pos[0, 0], 'ko')
    line_2d_pos, = ax_2d_pos.plot([], [], 'k', alpha=0.8)
    ax_2d_pos.set_xlim(0, len(data_series_pos))
    ax_2d_pos.set_ylim(-2.5, 2.5)
    ax_2d_pos.set_xlim(0, 300)
    ax_2d_pos.set_xlabel('Time (ms)', fontsize=12)
    ax_2d_pos.set_ylabel('Amplitude (arb. units)', fontsize=12)

    # Subplot 4: 2D plot for negative stimulus
    ax_2d_neg = fig.add_subplot(222)
    ax_2d_neg.axis('off')
    for ii in np.arange(100):
        ax_2d_neg.plot(data_series_neg[:, ii], color=colors[ii], linewidth=1, alpha=0.35)
    ax_2d_neg.plot(data_series_neg[:, 0], color='k')
    ax_2d_neg.plot(stimulus_neg[0, :], color="g", linewidth=2, label="Input")
    ax_2d_neg.plot(out_signal_neg[0, :], color="r", linewidth=2, label="Output")
    dot_2d_neg, = ax_2d_neg.plot(0, data_series_neg[0, 0], 'ko')
    line_2d_neg, = ax_2d_neg.plot([], [], 'k', alpha=0.8)
    ax_2d_neg.set_xlim(0, len(data_series_neg))
    ax_2d_neg.set_ylim(-2.5, 2.5)
    ax_2d_neg.set_xlim(0, 300)
    ax_2d_neg.set_xlabel('Time (ms)', fontsize=12)
    ax_2d_neg.set_ylabel('Amplitude (arb. units)', fontsize=12)
    # print(out_signal_neg[:, :])

    # Grid of circles for positive stimulus
    ax_grid_pos = fig.add_subplot(246, aspect='equal')
    ax_grid_pos.axis('off')
    norm = plt.Normalize(vmin=-0.5, vmax=0.5)
    circles_pos = [plt.Circle((j, 10-i-1), 0.4, color='white', alpha=0.8) for i in range(10) for j in range(10)]
    for circle in circles_pos:
        ax_grid_pos.add_patch(circle)
    ax_grid_pos.set_xlim(-1, 10)
    ax_grid_pos.set_ylim(-1, 10)

    # Grid of circles for negative stimulus
    ax_grid_neg = fig.add_subplot(248, aspect='equal')
    ax_grid_neg.axis('off')
    circles_neg = [plt.Circle((j, 10-i-1), 0.4, color='white', alpha=0.8) for i in range(10) for j in range(10)]
    for circle in circles_neg:
        ax_grid_neg.add_patch(circle)
    ax_grid_neg.set_xlim(-1, 10)
    ax_grid_neg.set_ylim(-1, 10)
    plt.tight_layout()
    title = fig.suptitle(f'RNN Units - Time step 0', fontsize=16)
    # plt.savefig("figure_culo.png")

    anim = FuncAnimation(fig, update_all_plots, frames=np.arange(0, 300, 2),
                         fargs=(pca_axis_x_pos, pca_axis_y_pos, pca_axis_z_pos,
                                pca_axis_x_neg, pca_axis_y_neg, pca_axis_z_neg,
                                ax_3d_pos, ax_3d_neg, line_3d_pos, dot_3d_pos, line_3d_neg, dot_3d_neg,
                                ax_2d_pos, ax_2d_neg, line_2d_pos, dot_2d_pos, line_2d_neg, dot_2d_neg,
                                data_series_pos, data_series_neg, circles_pos, circles_neg, norm, title,fig),
                         interval=50, blit=True)

    # anim.save(f'{output_folder}/{output_filename}', writer='ffmpeg', dpi=300)
    anim.save(f'{output_folder}/{output_filename}', writer='pillow', fps=10)
    plt.close()


# Load RNN
current_directory = os.path.dirname(__file__)
path_model = current_directory + \
             "/weights/05_Perceptual_dm_delayed_response/orthogonal_rrn_no_bias_term/weights_N_100_0/100_final.hdf5"
model = load_model(path_model)

# Generate positive and negative stimuli DM TASK
# x_train, y_train, seq_dur, end_of_pulse, change_point = generate_single_trial(1, 1)
# stim_pos = np.array(x_train).reshape(-1)
# x_train, y_train, seq_dur, end_of_pulse, change_point = generate_single_trial(2,1)
# stim_neg = np.array(x_train).reshape(-1)

# For the Integration signal:

x_train, y_train, seq_dur, end_of_pulse, change_point = generate_single_trial(ensure_negative_integral=True)
stim_pos = np.array(x_train).reshape(-1)
x_train, y_train, seq_dur, end_of_pulse, change_point = generate_single_trial()
stim_neg = np.array(x_train).reshape(-1)

"""
x_train, y_train, seq_dur, end_of_pulse, change_point = generate_single_trial()
stim_pos = np.array(x_train).reshape(-1)
x_train, y_train, seq_dur, end_of_pulse, change_point = generate_single_trial(ensure_negative_integral=True)
stim_neg = np.array(x_train).reshape(-1)
"""

# Adjusts the format of stim_pos and stim_neg before passing them to process_stimulus
stim_pos = np.expand_dims(stim_pos, axis=0)
stim_neg = np.expand_dims(stim_neg, axis=0)

stim_pos = np.expand_dims(stim_pos, axis=-1)
stim_neg = np.expand_dims(stim_neg, axis=-1)

# Process stimuli and generate inputs for animation
data_series_pos, data_series_neg, out_pos, out_neg = generate_inputs_for_animation(model, stim_pos, stim_neg)

# PCA
data_series_pos_ = np.squeeze(data_series_pos, axis=0)
data_series_neg_ = np.squeeze(data_series_neg, axis=0)

pca_pos = PCA(n_components=3)
pca_neg = PCA(n_components=3)

X_pca_p = pca_pos.fit_transform(data_series_pos_.T)
X_pca_n = pca_neg.fit_transform(data_series_neg_.T)

vector_pca_pos = pca_neg.components_
vector_pca_neg = pca_pos.components_

# PCA axes

pca_axis_x_p, pca_axis_y_p, pca_axis_z_p = vector_pca_pos[0], vector_pca_pos[1], vector_pca_pos[2]
pca_axis_x_n, pca_axis_y_n, pca_axis_z_n = vector_pca_neg[0], vector_pca_neg[1], vector_pca_neg[2]

print(pca_axis_x_n[0], pca_axis_y_n[0], pca_axis_z_n[0])
print(pca_axis_x_p[0], pca_axis_y_p[0], pca_axis_z_p[0])


# Create and save the combined animation
create_combined_animation(data_series_pos_, data_series_neg_, stim_pos, stim_neg, out_pos, out_neg,
                          pca_axis_x_p, pca_axis_y_p, pca_axis_z_p, pca_axis_x_n, pca_axis_y_n, pca_axis_z_n,
                          'plots', 'animation_comparison.gif')
