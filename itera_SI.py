# itera_SI.py
"""
MIT License - C. Jarne V. 1.0 - 2025
This code systematically analyzes temporal organization in neural
activity across multiple RNNs trained on decision-making tasks.
It computes a Sequentiality Index (SI) by processing trial
data from each network and evaluating population dynamics during
stimulus integration and response periods. The implementation
classifies decisions into positive/negative/null categories
based on output thresholds, then correlates these behavioral
outcomes with structural properties of the neural activity sequences.
It generates comparative datasets linking network architectures to their
temporal processing characteristics.
"""

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Initializer
from utils.net_constraint_create import *
from utils.SI_computing import compute_SI

# initial config
current_directory = os.path.dirname(__file__)
base_r_dir = os.path.join(current_directory, "weights", "01_DM_delayed_response")
output_dir = os.path.join(current_directory, "SI_results")
task_name = "01_DM_delayed_response"
task = "Simple DM"

# GPU config
os.environ['TF_DISABLE_GPU'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class IdentityInitializer(Initializer):
    def __call__(self, shape, dtype=None):
        if shape[0] != shape[1]:
            raise ValueError("Identity matrix initializer requires a square matrix shape.")
        return np.identity(shape[0], dtype=dtype)


def classify_decision(y_signal, threshold=0.5):
    """Classify the decision as positive, negative or null."""
    max_val = np.max(y_signal[250:])
    min_val = np.min(y_signal[250:])

    if max_val > threshold:
        return "positive"
    elif min_val < -threshold:
        return "negative"
    else:
        return "no_signal"


"""
    positive_indices = np.where(y_signal > threshold)[0]
    negative_indices = np.where(y_signal < -threshold)[0]

    return positive_indices, negative_indices
"""
# ========== Dictionary for storing positive, negative or null results ==========

results = {
    "positive": {"signal": [], "inter": [], "decision": []},
    "negative": {"signal": [], "inter": [], "decision": []},
    "no_signal": {"signal": [], "inter": [], "decision": []}
}

# To process individual networks


def process_network(model_path, output_path):
    # Cargar modelo
    custom_objects = {
        'NonNegLast': NonNegLast,
        'NonNegLast_input': NonNegLast_input,
        'my_init_exi_ini': my_init_exi_ini,
        'my_init_rec': my_init_rec,
        'SimpleRNN': lambda **cfg: tf.keras.layers.SimpleRNN(**cfg),
        'IdentityInitializer': IdentityInitializer
    }

    try:
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        print(f"Error cargando {model_path}: {str(e)}")
        return

    # Data generator

    if task == "Integral DM":
        from data_set_generators.generate_perceptual_dm import generate_trials
        mem_gap = 200

    if task in ["Simple DM", "Simple DM Long-short"]:
        # be aware of internal conf for data_set_generator to create one data set or the other
        from data_set_generators.generate_DM_delayed_response_sample import generate_trials
        mem_gap = 0

    x_train, y_train, _ = generate_trials(30, mem_gap=mem_gap)

    # Process samples
    si_results = []
    for sample_idx, (x_sample, y_sample) in enumerate(zip(x_train, y_train)):
        x_sample = x_sample[np.newaxis, :, :]

        # Extrac activity
        layer_output = model.layers[0](x_sample)
        activity = np.squeeze(layer_output.numpy().T)  # (neuronas, tiempo)

        # Compute SI
        si = compute_SI(activity)
        decision_type = classify_decision(y_sample[:, 0])

        si_results.append({
            'sample_id': sample_idx,
            'decision_type': decision_type,
            'SI': si
        })

    # Save results
    with open(output_path, 'w') as f:
        f.write("Sample_ID\tDecision_Type\tSI_Score\n")
        for res in si_results:
            line = f"{res['sample_id']}\t{res['decision_type']}\t{res['SI']:.4f}\n"
            f.write(line)


# Run over dir
for root, dirs, files in os.walk(base_r_dir):
    for file in files:
        if file.endswith("final.hdf5"):

            model_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, base_r_dir)

            # file name
            filename = f"SI_{task_name}_{relative_path.replace(os.sep, '_')}.txt"
            output_path = os.path.join(output_dir, filename)

            # Creaat dir
            os.makedirs(output_dir, exist_ok=True)

            # Process net
            print(f"Procesando: {relative_path}")
            process_network(model_path, output_path)

print("Procesamiento completo!")
