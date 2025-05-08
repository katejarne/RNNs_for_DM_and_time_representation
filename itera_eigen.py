# itera_eigen.py
"""
MIT License - C. Jarne V. 1.0 - 2025
This code performs spectral analysis on RNNs trained for
delayed decision-making tasks. It systematically processes
multiple saved models, extracting their recurrent weight
matrices to calculate eigenvalue spectra. For each network,
it computes the magnitudes of the top 5 eigenvalues, which indicate
key dynamical properties. The script generates comparative
records of spectral characteristics through their weight matrix configurations.
It creates an output directory called "eigen_results"
and stores inside for each network a .txt with the 5 eigenvalues
whose distance with the unit circle is greater.
"""
import os
from tensorflow.keras.models import load_model
import numpy.linalg as LA
from utils.net_constraint_create import *

# Initial config: Main directory with trained networks (one in each subdir)
current_directory = os.path.dirname(__file__)
base_r_dir = os.path.join(current_directory, "weights", "03_DM_delayed_response_4_times")
output_dir = os.path.join(current_directory, "eigen_results")
task_name = "01_DM_delayed_response"

# GPU config
os.environ['TF_DISABLE_GPU'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class IdentityInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        if shape[0] != shape[1]:
            raise ValueError("Identity matrix initializer requires a square matrix shape.")
        return np.identity(shape[0], dtype=dtype)


def process_network(model_path, output_path):
    # Load model with custom objects
    custom_objects = {
        'NonNegLast': NonNegLast,
        'NonNegLast_input': NonNegLast_input,
        'my_init_exi_ini': my_init_exi_ini,
        'my_init_rec': my_init_rec,
        'SimpleRNN': tf.keras.layers.SimpleRNN,
        'IdentityInitializer': IdentityInitializer
    }

    try:
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        print(f"Loading Error  {model_path}: {str(e)}")
        return

    # Extract recurring weight matrix
    recurrent_weights = None
    for layer in model.layers:
        if 'rnn' in layer.name.lower():
            weights = layer.get_weights()
            if len(weights) > 1:  # [input_weights, recurrent_weights, biases]
                recurrent_weights = weights[1]
                break

    if recurrent_weights is None:
        print(f"No recurring weights were found in {model_path}")
        return

    # Calculate eigenvalues
    eigenvalues = LA.eig(recurrent_weights)[0]

    # Calculate modules and obtain the 5 largest
    magnitudes = np.abs(eigenvalues)
    top5_magnitudes = sorted(magnitudes, reverse=True)[:5]

    # Save
    with open(output_path, 'w') as f:
        f.write("Top5_Eigenvalues\n")
        f.write("\t".join(f"{val:.6f}" for val in top5_magnitudes) + "\n")

# Traversing directory structure
for root, dirs, files in os.walk(base_r_dir):
    for file in files:
        if file.endswith("final.hdf5"):

            model_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, base_r_dir)

            # Create file name
            filename = f"Eigen_{task_name}_{relative_path.replace(os.sep, '_')}.txt"
            output_path = os.path.join(output_dir, filename)

            # Create dir
            os.makedirs(output_dir, exist_ok=True)

            # Process net
            print(f"Procesando: {relative_path}")
            process_network(model_path, output_path)

print("Done!")
