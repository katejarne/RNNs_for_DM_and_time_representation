# recompute_accuracy.py (CORREGIDO)
"""
Post-hoc accuracy recalculation for already-trained networks.
Generates a single test set per task (fixed seed), loads each saved model,
predicts, and computes accuracy using a robust response-window detection
that EXCLUDES trials with no response (all-zero target).
Overwrites 'train_stats.txt' with corrected values.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import defaultdict
import importlib

# Silence TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ----------------------------------------------------------------------
# 1. Task configuration (must match your train_ensemble.py)
# ----------------------------------------------------------------------
TASK_CONFIG = {
    "Simple_DM": {
        "generator": "data_set_generators.generate_DM_delayed_response_sample",
        "mem_gap": 0,
        "sample_size": 500,
        "output_type": "binary",
    },
    "Simple_DM_Long-short": {
        "generator": "data_set_generators.generate_DM_delayed_response_sample_L_H",
        "mem_gap": 0,
        "sample_size": 500,
        "output_type": "binary",
    },
    "Simple_DM_4_times": {
        "generator": "data_set_generators.generate_DM_delayed_response_sample_mult_times_4",
        "mem_gap": 0,
        "sample_size": 500,
        "output_type": "multi",
    },
    "Simple_DM_8_times": {
        "generator": "data_set_generators.generate_DM_delayed_response_sample_mult_times_8",
        "mem_gap": 0,
        "sample_size": 500,
        "output_type": "multi",
    },
    "Simple_DM_8_time_encoded": {
        "generator": "data_set_generators.generate_DM_delayed_response_sample_mult_times_8_intervals",
        "mem_gap": 0,
        "sample_size": 500,
        "output_type": "multi",
    },
    "Integral_DM": {
        "generator": "data_set_generators.generate_perceptual_dm",
        "mem_gap": 200,
        "sample_size": 500,
        "output_type": "binary",
    },
    "Integral_DM_signal_keep": {
        "generator": "data_set_generators.generate_perceptual_dm_sig_not_end",
        "mem_gap": 50,
        "sample_size": 500,
        "output_type": "binary",
    },
    "Integral_DM_Cue": {
        "generator": "data_set_generators.generate_perceptual_dm_sig_not_end_cue_mod",
        "mem_gap": 50,
        "sample_size": 500,
        "output_type": "binary",
    },
    "Multi_Ampli": {
        "generator": "data_set_generators.generate_DM_delayed_response_sample_mult_amplitude_8",
        "mem_gap": 100,
        "sample_size": 500,
        "output_type": "multi",
    },
    "interval_compare": {
        "generator": "data_set_generators.generate_interval_comparison",
        "mem_gap": 20,
        "sample_size": 500,
        "output_type": "binary",
    },
}

# ----------------------------------------------------------------------
# 2. Robust accuracy function (NO-RESPONSE trials excluded)
# ----------------------------------------------------------------------
def compute_accuracy(y_true, y_pred, output_type):
    """
    For each trial, first check if there is a response (|target| > 0.1 anywhere).
    If not, skip the trial (neutral). Otherwise, detect onset as first time
    where |target| > 0.1, average from onset to end, and compare.
    """
    n_trials, n_steps, _ = y_true.shape
    valid_trials = 0
    correct = 0
    for i in range(n_trials):
        # Check if any response exists
        if np.max(np.abs(y_true[i, :, 0])) <= 0.1:
            continue   # neutral trial, skip
        valid_trials += 1

        # Find first time point where target deviates from zero
        onset = np.argmax(np.abs(y_true[i, :, 0]) > 0.1)
        if onset == 0 and np.abs(y_true[i, 0, 0]) <= 0.1:
            # fallback (shouldn't be needed)
            onset = int(0.8 * n_steps)
        t_resp = np.mean(y_true[i, onset:, 0])
        p_resp = np.mean(y_pred[i, onset:, 0])

        if output_type == "multi":
            # Discretise to nearest integer in [-8, 8] (0 excluded)
            t_int = np.clip(np.round(t_resp).astype(int), -8, 8)
            p_int = np.clip(np.round(p_resp).astype(int), -8, 8)
            if t_int != 0 and p_int == t_int:
                correct += 1
        else:
            # Binary: compare sign (positive/negative)
            if (t_resp > 0 and p_resp > 0) or (t_resp < 0 and p_resp < 0):
                correct += 1
    if valid_trials == 0:
        return 0.0
    return correct / valid_trials


def compute_timing_error(y_true, y_pred, threshold=0.5):
    """Mean absolute difference in steps of first threshold crossing."""
    errors = []
    for i in range(y_true.shape[0]):
        # Skip trials with no response
        if np.max(np.abs(y_true[i, :, 0])) <= 0.1:
            continue
        t_idx = np.argmax(np.abs(y_true[i, :, 0]) > threshold)
        p_idx = np.argmax(np.abs(y_pred[i, :, 0]) > threshold)
        if t_idx > 0 or np.abs(y_true[i, 0, 0]) > threshold:
            if p_idx > 0 or np.abs(y_pred[i, 0, 0]) > threshold:
                errors.append(abs(p_idx - t_idx))
    return np.mean(errors) if errors else float('nan')


# ----------------------------------------------------------------------
# 3. Custom objects (needed to load models saved with AsymmetricInitializer)
# ----------------------------------------------------------------------
class AsymmetricInitializer(tf.keras.initializers.Initializer):
    def __init__(self, base_initializer, asymmetry_factor=1.0):
        self.base_initializer = base_initializer
        self.asymmetry_factor = asymmetry_factor

    def __call__(self, shape, dtype=None):
        W0 = self.base_initializer(shape, dtype=dtype)
        W_sym = (W0 + tf.transpose(W0)) / 2.0
        W_anti = (W0 - tf.transpose(W0)) / 2.0
        return W_sym + self.asymmetry_factor * W_anti

    def get_config(self):
        return {
            'base_initializer': tf.keras.initializers.serialize(self.base_initializer),
            'asymmetry_factor': self.asymmetry_factor,
        }

    @classmethod
    def from_config(cls, config):
        base = tf.keras.initializers.deserialize(config['base_initializer'])
        return cls(base_initializer=base, asymmetry_factor=config['asymmetry_factor'])

CUSTOM_OBJECTS = {'AsymmetricInitializer': AsymmetricInitializer}


# ----------------------------------------------------------------------
# 4. Main recalculation loop
# ----------------------------------------------------------------------
def main():
    results_dir = "../results"  # same as run_experiment.py
    fixed_seed = 12345        # for reproducibility of test sets

    # Store test sets per task to avoid regeneration
    test_sets = {}

    for dir_name in sorted(os.listdir(results_dir)):
        dir_path = os.path.join(results_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue

        # Parse task and init from directory name (e.g., Simple_DM_Orthogonal)
        for init_suffix in ['Normal', 'Orthogonal']:
            if dir_name.endswith('_' + init_suffix):
                task_tag = dir_name[:-(len(init_suffix) + 1)]
                init = init_suffix
                break
        else:
            print(f"Skipping unparseable directory: {dir_name}")
            continue

        if task_tag not in TASK_CONFIG:
            print(f"Unknown task tag: {task_tag}, skipping")
            continue

        cfg = TASK_CONFIG[task_tag]
        print(f"\nProcessing task={task_tag} init={init}")

        # Obtain or generate test set for this task (once)
        if task_tag not in test_sets:
            tf.random.set_seed(fixed_seed)
            np.random.seed(fixed_seed)
            gen_module = importlib.import_module(cfg["generator"])
            generate_trials = gen_module.generate_trials
            result = generate_trials(cfg["sample_size"], cfg["mem_gap"])
            if len(result) == 3:
                x_test, y_test, _ = result
            else:
                x_test, y_test, _, _ = result
            test_sets[task_tag] = (x_test, y_test)
            # Count valid trials
            valid_count = sum(np.max(np.abs(y_test[i,:,0])) > 0.1 for i in range(y_test.shape[0]))
            print(f"  Generated test set of size {x_test.shape[0]} (valid trials: {valid_count})")
        else:
            x_test, y_test = test_sets[task_tag]

        # Process each replica net_XX
        rep_dirs = sorted([d for d in os.listdir(dir_path) if d.startswith('net_')])
        for rep_name in rep_dirs:
            rep_path = os.path.join(dir_path, rep_name)
            model_path = os.path.join(rep_path, 'weights_final.keras')
            if not os.path.exists(model_path):
                # fallback: try any .hdf5
                hdf5_files = [f for f in os.listdir(rep_path) if f.endswith('.hdf5')]
                if hdf5_files:
                    model_path = os.path.join(rep_path, hdf5_files[0])
                else:
                    print(f"  {rep_name}: no model found, skipping")
                    continue

            # Load model
            tf.keras.backend.clear_session()
            model = load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)

            # Predict on test set
            y_pred = model.predict(x_test, verbose=0)

            # Compute new metrics
            new_acc = compute_accuracy(y_test, y_pred, cfg["output_type"])
            new_te = compute_timing_error(y_test, y_pred)
            new_mse = np.mean((y_test - y_pred)**2)   # test MSE

            # Update train_stats.txt
            stats_path = os.path.join(rep_path, 'train_stats.txt')
            old_lines = []
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    old_lines = f.readlines()

            updated_lines = []
            for line in old_lines:
                if line.startswith('accuracy'):
                    updated_lines.append(f'accuracy     = {new_acc:.4f}\n')
                elif line.startswith('timing_error'):
                    updated_lines.append(f'timing_error = {new_te:.2f}\n')
                elif line.startswith('final_mse'):
                    updated_lines.append(line)  # keep original training MSE
                    updated_lines.append(f'test_mse     = {new_mse:.6f}\n')
                else:
                    updated_lines.append(line)
            if not old_lines:
                updated_lines = [
                    f'accuracy     = {new_acc:.4f}\n',
                    f'timing_error = {new_te:.2f}\n',
                    f'test_mse     = {new_mse:.6f}\n',
                ]
            with open(stats_path, 'w') as f:
                f.writelines(updated_lines)

            print(f"  {rep_name}: acc={new_acc:.3f}, timing_err={new_te:.1f}, test_mse={new_mse:.5f}")

    print("\nAll networks updated. Now run 'python analyze_ensemble.py' to regenerate the table.")

if __name__ == "__main__":
    main()
