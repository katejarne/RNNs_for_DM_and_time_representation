# train_ensemble.py
"""
Trains N_REPLICAS independent RNNs for a single (task, init) configuration,
ensuring all networks have training MSE <= mse_threshold.
Retries with fresh random seeds until N_REPLICAS successful networks are obtained.
"""

import argparse
import os
import shutil
import warnings

os.environ['TF_DISABLE_GPU']          = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '3'

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models      import Sequential
from tensorflow.keras.layers      import SimpleRNN, Dense
from tensorflow.keras.callbacks   import ModelCheckpoint, Callback
from keras                        import optimizers


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task',       required=True)
    p.add_argument('--init',       required=True, choices=['Normal', 'Orthogonal'])
    p.add_argument('--n_replicas', type=int, default=10)
    p.add_argument('--n_rec',      type=int, default=100)
    p.add_argument('--out_dir',    required=True)
    p.add_argument('--mse_threshold', type=float, default=0.1,
                   help='maximum training MSE to accept a network')   # <<< NUEVO
    return p.parse_args()


# ── Task configuration ────────────────────────────────────────────────────────
TASK_CFG = {
    ("Simple DM",           "data_set_generators.generate_DM_delayed_response_sample",           0,       15050, 20),
    ("Simple DM Long-short","data_set_generators.generate_DM_delayed_response_sample_L_H",           0,       15050, 20),
    ("Simple DM 4 times",   "data_set_generators.generate_DM_delayed_response_sample_mult_times_4", 0,    15050, 20),
    ("Simple DM 8 times",   "data_set_generators.generate_DM_delayed_response_sample_mult_times_8", 0,    15050, 20),
    ("Simple DM 8 time encoded", "data_set_generators.generate_DM_delayed_response_sample_mult_times_8_intervals", 0, 15050, 20),
    ("Integral DM",         "data_set_generators.generate_perceptual_dm",                        200, 6*15050, 40),
    ("Integral DM signal keep", "data_set_generators.generate_perceptual_dm_sig_not_end",        50,  6*15050, 40),
    ("Integral DM Cue",     "data_set_generators.generate_perceptual_dm_sig_not_end_cue_mod",    50,    15050, 40),
    ("Multi Ampli",         "data_set_generators.generate_DM_delayed_response_sample_mult_amplitude_8", 100, 6*15050, 40),
    ("interval compare",    "data_set_generators.generate_interval_comparison",                  20,    15050, 20),
}


# ── Callbacks ─────────────────────────────────────────────────────────────────
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=0.00005, verbose=0):
        super().__init__()
        self.monitor = monitor
        self.value   = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is not None and current < self.value:
            if self.verbose:
                print(f"  Early stopping at epoch {epoch}: {self.monitor}={current:.6f}")
            self.model.stop_training = True


# ── Initialiser ───────────────────────────────────────────────────────────────
class AsymmetricInitializer(tf.keras.initializers.Initializer):
    def __init__(self, base_initializer, asymmetry_factor=1.0):
        self.base_initializer  = base_initializer
        self.asymmetry_factor  = asymmetry_factor

    def __call__(self, shape, dtype=None):
        W0     = self.base_initializer(shape, dtype=dtype)
        W_sym  = (W0 + tf.transpose(W0)) / 2.0
        W_anti = (W0 - tf.transpose(W0)) / 2.0
        return W_sym + self.asymmetry_factor * W_anti

    def get_config(self):
        return {
            'base_initializer': tf.keras.initializers.serialize(self.base_initializer),
            'asymmetry_factor': self.asymmetry_factor,
        }


def make_recurrent_initializer(init_scheme, N_rec):
    if init_scheme == 'Orthogonal':
        return 'orthogonal'
    base = tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=1.0 * np.sqrt(1.0 / N_rec))
    return AsymmetricInitializer(base_initializer=base, asymmetry_factor=1.0)


# ── Build model ───────────────────────────────────────────────────────────────
def build_model(N_rec, input_shape, init_scheme):
    rec_init = make_recurrent_initializer(init_scheme, N_rec)
    model = Sequential([
        SimpleRNN(units=N_rec, return_sequences=True,
                  kernel_initializer='glorot_uniform',
                  recurrent_initializer=rec_init,
                  use_bias=True,          # <<< MODIFICADO: antes era False, ahora True como en tus scripts originales
                  input_shape=input_shape),
        Dense(units=1),
    ])
    adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999,
                           epsilon=1e-8, clipnorm=1.0)
    model.compile(loss='mse', optimizer=adam)
    return model


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_loss(history, path):
    plt.figure(figsize=(7, 4))
    plt.plot(history.history['loss'], label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.title('Training loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_io_samples(x_train, y_train, y_pred, path, n=6):
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    for ii, ax in enumerate(axes.flat):
        if ii >= n:
            break
        ax.plot(x_train[ii, :, 0], color='g',    lw=1, label='Input')
        if x_train.shape[2] == 2:
            ax.plot(x_train[ii, :, 1], color='pink', lw=1, label='Cue')
        ax.plot(y_train[ii, :, 0], color='gray', lw=2, label='Target')
        ax.plot(y_pred[ii,  :, 0], color='r',    lw=1, label='Predicted')
        ax.set_ylim([-2.5, 2.5])
        ax.legend(fontsize=5, loc=3)
        ax.tick_params(labelsize=7)
    fig.text(0.5, 0.02, 'Time step', ha='center', fontsize=9)
    fig.text(0.02, 0.5, 'Amplitude (arb. units)', va='center',
             rotation='vertical', fontsize=9)
    plt.suptitle('Sample input / target / predicted', fontsize=11)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    plt.savefig(path, dpi=150)
    plt.close()

def compute_accuracy(y_true, y_pred, task, mem_gap=0, margin=10):
    """
    Accuracy basado en el signo del promedio de la salida durante la ventana de
    respuesta fija: desde (mem_gap + margin) hasta el final de la secuencia.
    Para tareas multi‑amplitud, discretiza al entero más cercano.
    """
    n_trials, n_steps, _ = y_true.shape
    start = min(mem_gap + margin, n_steps - 1)   # al menos 1 paso de ventana
    correct = 0

    for i in range(n_trials):
        # Promedio en la ventana de respuesta
        t_resp = np.mean(y_true[i, start:, 0])
        p_resp = np.mean(y_pred[i, start:, 0])

        if "multi" in task or "8 times" in task:
            t_int = int(np.clip(np.round(t_resp), -8, 8))
            p_int = int(np.clip(np.round(p_resp), -8, 8))
            if t_int != 0 and t_int == p_int:
                correct += 1
        else:
            # Decisión binaria: comparación directa >0, evitando np.sign(0)
            if (t_resp > 0 and p_resp > 0) or (t_resp < 0 and p_resp < 0):
                correct += 1
    return correct / n_trials

def compute_timing_error(y_true, y_pred, threshold=0.5):
    n_trials = y_true.shape[0]
    errors   = []
    for i in range(n_trials):
        crossings_t = np.where(np.abs(y_true[i, :, 0]) > threshold)[0]
        crossings_p = np.where(np.abs(y_pred[i, :, 0]) > threshold)[0]
        if len(crossings_t) > 0 and len(crossings_p) > 0:
            errors.append(abs(int(crossings_p[0]) - int(crossings_t[0])))
    return float(np.mean(errors)) if errors else float('nan')


# ── Main training loop ────────────────────────────────────────────────────────
def main():
    args = parse_args()

    task        = args.task
    init        = args.init
    n_replicas  = args.n_replicas
    N_rec       = args.n_rec
    out_dir     = args.out_dir
    mse_thresh  = args.mse_threshold
    max_retries = n_replicas * 10   # límite de intentos totales

    if task not in [cfg[0] for cfg in TASK_CFG]:
        raise ValueError(f"Unknown task: {task!r}")

    # Extraer configuración correspondiente a la tarea
    gen_module, mem_gap, sample_size, epochs = None, None, None, None
    for tkey, gmod, mgap, ss, ep in TASK_CFG:
        if tkey == task:
            gen_module, mem_gap, sample_size, epochs = gmod, mgap, ss, ep
            break

    # Importar generador
    import importlib
    gen = importlib.import_module(gen_module)
    generate_trials = gen.generate_trials

    print(f"[train_ensemble] task={task!r}  init={init!r}  "
          f"mem_gap={mem_gap}  sample_size={sample_size}  epochs={epochs}  "
          f"mse_threshold={mse_thresh}")

    # Generar dataset una vez (compartido entre todos los intentos)
    print("Generating dataset …")
    result = generate_trials(sample_size, mem_gap)
    if len(result) == 3:
        x_train, y_train, seq_dur = result
    else:
        x_train, y_train, seq_dur, _ = result

    input_shape = (x_train.shape[1], x_train.shape[2])
    print(f"  x_train shape: {x_train.shape}  y_train shape: {y_train.shape}")

    # ── Bucle con reintentos ──────────────────────────────────────────────────
    success_count = 0
    attempt = 0

    while success_count < n_replicas and attempt < max_retries:
        print(f"\n--- Attempt {attempt+1} | Succesful networks: {success_count}/{n_replicas} ---")

        # Semillas únicas para cada intento (base grande para variar bien)
        seed = 42 + attempt * 1000
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Carpeta temporal
        temp_dir = os.path.join(out_dir, f"net_temp_{attempt:03d}")
        os.makedirs(temp_dir, exist_ok=True)

        tf.keras.backend.clear_session()
        model = build_model(N_rec, input_shape, init)

        callbacks = [
            EarlyStoppingByLossVal(monitor='loss', value=0.00005, verbose=1),
        ]

        history = model.fit(
            x_train[50:sample_size], y_train[50:sample_size],
            epochs=epochs, batch_size=64,
            callbacks=callbacks,
            verbose=0,
        )

        final_mse = float(history.history['loss'][-1])

        # Predicción sobre las primeras 50 muestras
        x_pred = x_train[:50]
        y_pred = model.predict(x_pred, verbose=0)

        # Guardar modelo temporalmente en temp_dir
        model_path_temp = os.path.join(temp_dir, 'weights_final.keras')
        model.save(model_path_temp)

        # Gráficos de pérdida y ejemplos (siempre se guardan en temp_dir)
        plot_loss(history, os.path.join(temp_dir, 'training_loss.png'))
        plot_io_samples(x_train[:6], y_train[:6], y_pred[:6],
                        os.path.join(temp_dir, 'io_samples.png'))

        # Decidir si aceptamos
        if final_mse <= mse_thresh:
            # Éxito: renombrar carpeta temporal a net_XX
            final_dir = os.path.join(out_dir, f"net_{success_count:02d}")
            # Si por alguna razón ya existe, eliminarla (no debería)
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            os.rename(temp_dir, final_dir)

            # Guardar estadísticas en la carpeta renombrada
            n_epochs_done = len(history.history['loss'])
            converged = final_mse < 0.00005
            # accuracy = compute_accuracy(y_train[:50], y_pred, task)
            accuracy = compute_accuracy(y_train[:50], y_pred, task, mem_gap, margin=10)
            timing_err = compute_timing_error(y_train[:50], y_pred)

            stats_path = os.path.join(final_dir, 'train_stats.txt')
            with open(stats_path, 'w') as fh:
                fh.write(f"task         = {task}\n")
                fh.write(f"init         = {init}\n")
                fh.write(f"replica      = {success_count}\n")
                fh.write(f"final_mse    = {final_mse:.6f}\n")
                fh.write(f"n_epochs     = {n_epochs_done}\n")
                fh.write(f"converged    = {converged}\n")
                fh.write(f"accuracy     = {accuracy:.4f}\n")
                fh.write(f"timing_error = {timing_err:.2f}\n")
                fh.write(f"seed_used    = {seed}\n")

            print(f"  ✔ Aceptada → {final_dir}  (MSE={final_mse:.5f}, acc={accuracy:.3f}, epochs={n_epochs_done})")
            success_count += 1
        else:
            # Falló: eliminar carpeta temporal
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"  ✘ Descartada (MSE={final_mse:.5f} > {mse_thresh})")

        attempt += 1

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f"\n[train_ensemble] Finalizado: {success_count}/{n_replicas} redes exitosas en {attempt} intentos.")
    if success_count < n_replicas:
        print(f"  ADVERTENCIA: Solo se obtuvieron {success_count} redes. Aumentá --mse_threshold o --max_retries.")


if __name__ == '__main__':
    main()
