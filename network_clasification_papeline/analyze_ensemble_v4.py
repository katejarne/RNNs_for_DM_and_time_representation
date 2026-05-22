# analyze_ensemble_v4.py
"""
==========
analyze_ensemble_v4.py
==========

Versión 4 del análisis de régimen dual, incorporando las sugerencias de K.:

1.  Reemplaza el PBR crudo por log10(PBR) en las tablas (con dos decimales)
    y añade un pequeño epsilon para estabilidad numérica.
2.  Genera una figura de dinámicas típicas que muestra un ejemplo de cada
    régimen dual (Oscillatory, Ramping/Decaying, Mixed) por tarea.
3.  Añade un criterio positivo de rampa/decaimiento (ramp index) basado en
    la correlación lineal de la traza PC1 durante la ventana de delay.
    La actividad ahora se clasifica en tres categorías:
    - Oscillatory si PBR >= pbr_threshold
    - Ramping/Decaying si PBR < pbr_threshold y ramp_index >= ramp_threshold
    - Mixed en caso contrario
    La clasificación dual combina el resultado espectral y de actividad:
    Oscillatory solo si ambos coinciden en Oscillatory,
    Ramping/Decaying solo si ambos coinciden en Ramping/Decaying,
    Mixed en cualquier otro caso.
"""

import argparse
import os
import csv
import warnings

os.environ['TF_DISABLE_GPU']         = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from scipy.stats import pearsonr
from numpy import linalg as LA

import tensorflow as tf


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='results')

    # Umbrales espectrales
    p.add_argument('--osc_threshold',   type=float, default=0.15,
                   help='OI >= this value → Oscillatory (spectral)')
    p.add_argument('--mixed_threshold', type=float, default=0.05,
                   help='OI >= this value → Mixed (spectral)')

    # Umbral para la actividad – PBR
    p.add_argument('--pbr_threshold',   type=float, default=5.0,
                   help='PBR >= this value → Oscillatory (activity). '
                        'PBR = max(P(f)) / (median(P(f)) + epsilon)')

    # Umbral para el ramp index
    p.add_argument('--ramp_threshold',  type=float, default=0.8,
                   help='Ramp index (abs linear correlation) >= this value '
                        '→ Ramping/Decaying (activity) when PBR < pbr_thr')

    # Ventana de delay/integración
    p.add_argument('--delay_start_frac', type=float, default=0.35)
    p.add_argument('--delay_end_frac',   type=float, default=0.85)

    # Parámetros PSD
    p.add_argument('--nperseg', type=int, default=64)
    p.add_argument('--n_trials', type=int, default=20)

    # Regularización
    p.add_argument('--epsilon', type=float, default=1e-10,
                   help='Small value to stabilise PBR computation')

    return p.parse_args()


# ── Custom objects ────────────────────────────────────────────────────────────
class AsymmetricInitializer(tf.keras.initializers.Initializer):
    def __init__(self, base_initializer, asymmetry_factor=1.0):
        self.base_initializer = base_initializer
        self.asymmetry_factor = asymmetry_factor

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

    @classmethod
    def from_config(cls, config):
        base = tf.keras.initializers.deserialize(config['base_initializer'])
        return cls(base, asymmetry_factor=config['asymmetry_factor'])


CUSTOM_OBJECTS = {'AsymmetricInitializer': AsymmetricInitializer}


# ── Spectral oscillation index ────────────────────────────────────────────────
def oscillation_index_spectral(W_rec):
    eigs    = LA.eigvals(W_rec)
    moduli  = np.abs(eigs)
    idx     = np.argmax(moduli)
    lam_dom = eigs[idx]
    mod_dom = moduli[idx]
    oi      = abs(lam_dom.imag) / mod_dom if mod_dom > 0 else 0.0
    return float(oi), lam_dom, eigs


def classify_spectral(oi, osc_thr, mixed_thr):
    if oi >= osc_thr:
        return 'Oscillatory'
    elif oi >= mixed_thr:
        return 'Mixed'
    else:
        return 'Ramping/Decaying'


# ── Run network forward ──────────────────────────────────────────────────────
def run_network_forward(model, T=400, n_trials=20, noise_std=0.05):
    rnn_layer = None
    for layer in model.layers:
        if 'simple_rnn' in layer.name.lower() or 'rnn' in layer.name.lower():
            rnn_layer = layer
            break
    if rnn_layer is None:
        return None

    N = rnn_layer.units
    try:
        input_dim = model.input_shape[-1]
    except Exception:
        input_dim = 1

    H_list = []
    for k in range(n_trials):
        sign = 1.0 if k % 2 == 0 else -1.0
        pulse_end = int(0.30 * T)
        x = np.zeros((1, T, input_dim), dtype=np.float32)
        x[0, :pulse_end, 0] = sign
        x += np.random.randn(*x.shape).astype(np.float32) * noise_std

        try:
            W_in, W_rec_arr, b = rnn_layer.get_weights()
        except ValueError:
            try:
                W_in, W_rec_arr = rnn_layer.get_weights()
                b = np.zeros(N, dtype=np.float32)
            except Exception:
                return None

        h = np.zeros(N, dtype=np.float32)
        H_trial = np.zeros((T, N), dtype=np.float32)
        for t in range(T):
            xt = x[0, t, :]
            h  = np.tanh(W_in.T @ xt + W_rec_arr.T @ h + b)
            H_trial[t] = h
        H_list.append(H_trial)

    return np.array(H_list)   # (n_trials, T, N)


# ── PBR and ramp index ───────────────────────────────────────────────────────
def peak_to_background_and_ramp(H, delay_start_frac, delay_end_frac,
                                nperseg=64, fs=1.0, epsilon=1e-10):
    """
    Returns:
      log10_pbr : float   log10( max(P) / (median(P) + epsilon) )
      pbr       : float   raw PBR (for classification)
      peak_freq : float
      freqs     : ndarray
      psd       : ndarray
      ramp_idx  : float   abs(pearson r) between time and raw PC1 (delay window)
      pc1_raw   : ndarray (delay window, raw)
    """
    n_trials, T, N = H.shape
    H_mean = H.mean(axis=0)   # (T, N)

    t_start = int(delay_start_frac * T)
    t_end   = int(delay_end_frac   * T)
    H_delay = H_mean[t_start:t_end, :]   # (T_delay, N)
    T_delay = H_delay.shape[0]

    if T_delay < 8:
        return 0.0, 0.0, 0.0, np.array([0.0]), np.array([0.0]), 0.0, np.zeros(T_delay)

    # PCA – first PC
    H_c = H_delay - H_delay.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(H_c, full_matrices=False)
        pc1_raw = H_c @ Vt[0]
    except np.linalg.LinAlgError:
        pc1_raw = H_delay[:, 0]

    # ---- ramp index ----
    time_vec = np.arange(T_delay)
    # Pearson r between time and PC1 (absolute value)
    r_val, _ = pearsonr(time_vec, pc1_raw)
    ramp_idx = abs(r_val)

    # ---- detrend and PSD ----
    pc1_det = sp_signal.detrend(pc1_raw, type='linear')

    _nperseg = min(nperseg, T_delay // 2)
    if _nperseg < 4:
        _nperseg = T_delay
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        freqs, psd = sp_signal.welch(pc1_det, fs=fs, nperseg=_nperseg,
                                     window='hann', scaling='density')

    f_min = 2.0 / T_delay * fs
    nondc_mask = (freqs > f_min) & (freqs <= fs/2)
    if not np.any(nondc_mask):
        return 0.0, 0.0, 0.0, freqs, psd, ramp_idx, pc1_raw

    psd_band = psd[nondc_mask]
    if psd_band.sum() == 0:
        return 0.0, 0.0, 0.0, freqs, psd, ramp_idx, pc1_raw

    peak_val   = np.max(psd_band)
    background = np.median(psd_band)
    pbr        = float(peak_val / (background + epsilon))
    log10_pbr  = float(np.log10(pbr + 1e-30))

    peak_idx  = np.argmax(psd_band)
    peak_freq = float(freqs[nondc_mask][peak_idx])

    return log10_pbr, pbr, peak_freq, freqs, psd, ramp_idx, pc1_raw


def classify_activity_from_pbr_and_ramp(pbr, ramp_idx, pbr_thr, ramp_thr):
    if pbr >= pbr_thr:
        return 'Oscillatory'
    elif ramp_idx >= ramp_thr:
        return 'Ramping/Decaying'
    else:
        return 'Mixed'


# ── Dual classification (three‑by‑three) ─────────────────────────────────────
def classify_dual(regime_spectral, regime_activity):
    if regime_spectral == 'Oscillatory' and regime_activity == 'Oscillatory':
        return 'Oscillatory'
    if regime_spectral == 'Ramping/Decaying' and regime_activity == 'Ramping/Decaying':
        return 'Ramping/Decaying'
    return 'Mixed'


# ── Henrici index ────────────────────────────────────────────────────────────
def henrici_index(W_rec):
    eigs   = LA.eigvals(W_rec)
    F_norm = np.linalg.norm(W_rec, 'fro')
    if F_norm == 0:
        return 0.0
    return float(np.sqrt(max(F_norm**2 - np.sum(np.abs(eigs)**2), 0.0)) / F_norm)


# ── Plots ────────────────────────────────────────────────────────────────────
def plot_eigenvalues(eigs, lam_dom, regime, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(eigs.real, eigs.imag, s=8, c='k', alpha=0.7, label='Eigenvalues')
    ax.scatter(lam_dom.real, lam_dom.imag, s=40, c='red', zorder=5,
               label=f'Dom. λ ({regime})')
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), '--', color='steelblue', lw=1,
            alpha=0.6, label='Unit circle')
    ax.axvline(0, color='gray', lw=0.5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel(r'Re($\lambda$)', fontsize=11)
    ax.set_ylabel(r'Im($\lambda$)', fontsize=11)
    ax.set_title(f'Eigenvalue spectrum — {regime}', fontsize=10)
    ax.legend(fontsize=7)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_psd_and_ramp(freqs, psd, peak_freq, pbr, log10_pbr, ramp_idx,
                      regime_activity, path, delay_start_frac, delay_end_frac):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.semilogy(freqs, psd, color='navy', lw=1.5)
    if peak_freq > 0:
        ax.axvline(peak_freq, color='crimson', lw=1.2, ls='--',
                   label=f'Peak f={peak_freq:.3f}')
    ax.set_xlabel('Freq (cyc/step)')
    ax.set_ylabel('PSD')
    ax.set_title(f'PBR (log10)={log10_pbr:.2f}\n{regime_activity}', fontsize=9)
    ax.legend(fontsize=7)

    ax2 = axes[1]
    ax2.bar(['Peak','Background'], [np.max(psd), np.median(psd)],
            color=['crimson','gray'])
    ax2.set_ylabel('Power density')
    ax2.set_title(f'Raw PBR = {pbr:.1f}')
    ax2.tick_params(labelsize=8)

    ax3 = axes[2]
    ax3.text(0.5, 0.5, f'Ramp index\n(linear corr)\n{ramp_idx:.3f}',
             ha='center', va='center', fontsize=14, transform=ax3.transAxes)
    ax3.set_title('Ramp metric', fontsize=10)
    ax3.axis('off')

    plt.suptitle('Activity criterion — PC1 detrended PSD & ramp index',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pc1_trace(H, delay_start_frac, delay_end_frac, regime_dual, path):
    n_trials, T, N = H.shape
    H_mean  = H.mean(axis=0)
    t_start = int(delay_start_frac * T)
    t_end   = int(delay_end_frac   * T)

    H_c = H_mean - H_mean.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(H_c, full_matrices=False)
        pc1_full = H_c @ Vt[0]
    except np.linalg.LinAlgError:
        pc1_full = H_mean[:, 0]

    fig, ax = plt.subplots(figsize=(8, 3))
    time = np.arange(T)
    ax.plot(time, pc1_full, color='steelblue', lw=1.2, label='PC1 (trial-avg)')
    ax.axvspan(t_start, t_end, alpha=0.15, color='orange', label='Delay window')
    ax.set_xlabel('Time step', fontsize=10)
    ax.set_ylabel('PC1 projection', fontsize=10)
    ax.set_title(f'Trial-averaged PC1 trajectory — {regime_dual}', fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Typical dynamics figure ──────────────────────────────────────────────────
def plot_typical_dynamics(csv_rows, out_dir):
    # For each task, pick one replica per dual regime and plot its PC1 trace
    tasks = sorted({r['task'] for r in csv_rows})
    regimes_order = ['Oscillatory', 'Ramping/Decaying', 'Mixed']
    n_regimes = len(regimes_order)

    fig, axes = plt.subplots(len(tasks), n_regimes,
                             figsize=(4 * n_regimes, 3 * len(tasks)),
                             squeeze=False)
    for i, task in enumerate(tasks):
        for j, reg in enumerate(regimes_order):
            ax = axes[i, j]
            candidates = [r for r in csv_rows
                          if r['task'] == task and r['regime_dual'] == reg]
            if not candidates:
                ax.text(0.5, 0.5, 'None', ha='center', va='center')
                ax.set_title(f'{task}\n{reg}', fontsize=8)
                continue
            # Pick the first one (or with highest accuracy)
            chosen = candidates[0]
            # Need to load the model and generate PC1 trace
            model_dir = os.path.join(out_dir,
                                     chosen['task'].replace(' ', '_') + '_' + chosen['init'],
                                     f"net_{chosen['replica']:02d}")
            model_path = os.path.join(model_dir, 'weights_final.keras')
            if not os.path.exists(model_path):
                for fn in os.listdir(model_dir):
                    if fn.endswith('_final.hdf5') or fn.endswith('final.hdf5'):
                        model_path = os.path.join(model_dir, fn)
                        break
            try:
                tf.keras.backend.clear_session()
                model = tf.keras.models.load_model(model_path,
                                                   custom_objects=CUSTOM_OBJECTS,
                                                   compile=False)
                T_model = 400  # fallback
                try:
                    T_model = model.input_shape[1] or 400
                except Exception:
                    pass
                H = run_network_forward(model, T=T_model, n_trials=20)
                if H is not None:
                    H_mean = H.mean(axis=0)
                    t_start = int(0.35 * T_model)
                    t_end   = int(0.85 * T_model)
                    H_c = H_mean - H_mean.mean(axis=0)
                    try:
                        _, _, Vt = np.linalg.svd(H_c, full_matrices=False)
                        pc1 = H_c @ Vt[0]
                    except np.linalg.LinAlgError:
                        pc1 = H_mean[:, 0]
                    ax.plot(np.arange(T_model), pc1, lw=1)
                    ax.axvspan(t_start, t_end, alpha=0.1, color='orange')
            except Exception:
                ax.text(0.5, 0.5, 'Load err', ha='center', va='center')
            ax.set_title(f'{task}\n{reg} ({chosen["init"]})', fontsize=8)
            ax.tick_params(labelsize=6)
    plt.suptitle('Typical PC1 dynamics per regime', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = os.path.join(out_dir, 'typical_dynamics.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  Typical dynamics figure → {fig_path}")


# ── Read train stats ─────────────────────────────────────────────────────────
def read_train_stats(path):
    stats = {}
    if not os.path.exists(path):
        return stats
    with open(path) as fh:
        for line in fh:
            if '=' in line:
                k, v = line.split('=', 1)
                stats[k.strip()] = v.strip()
    return stats


# ── Process one network ──────────────────────────────────────────────────────
def process_network(rep_dir, args):
    model_path = os.path.join(rep_dir, 'weights_final.keras')
    if not os.path.exists(model_path):
        for fn in os.listdir(rep_dir):
            if fn.endswith('_final.hdf5') or fn.endswith('final.hdf5'):
                model_path = os.path.join(rep_dir, fn)
                break
        else:
            print(f"  [SKIP] No model file in {rep_dir}")
            return None

    try:
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(
            model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
    except Exception as e:
        print(f"  [ERROR] Load failed {model_path}: {e}")
        return None

    # W_rec
    W_rec = None
    for layer in model.layers:
        if 'simple_rnn' in layer.name.lower() or 'rnn' in layer.name.lower():
            weights = layer.get_weights()
            W_rec   = np.array(weights[1])
            break
    if W_rec is None:
        print(f"  [ERROR] No RNN layer in {model_path}")
        return None

    # Spectral
    oi, lam_dom, eigs = oscillation_index_spectral(W_rec)
    regime_spectral   = classify_spectral(oi, args.osc_threshold,
                                           args.mixed_threshold)
    h_idx             = henrici_index(W_rec)

    plot_eigenvalues(eigs, lam_dom, regime_spectral,
                     os.path.join(rep_dir, 'eigenvalues.png'))

    # Activity (PBR + ramp)
    try:
        T_model = model.input_shape[1] or 400
    except Exception:
        T_model = 400

    log10_pbr   = 0.0
    pbr         = 0.0
    peak_freq   = 0.0
    freqs_psd   = np.array([0.0])
    psd_arr     = np.array([0.0])
    ramp_idx    = 0.0
    regime_activity = 'Mixed'

    H = run_network_forward(model, T=T_model, n_trials=args.n_trials)
    if H is not None and H.shape[0] > 0:
        log10_pbr, pbr, peak_freq, freqs_psd, psd_arr, ramp_idx, pc1_raw = \
            peak_to_background_and_ramp(
                H,
                delay_start_frac=args.delay_start_frac,
                delay_end_frac=args.delay_end_frac,
                nperseg=args.nperseg,
                epsilon=args.epsilon,
            )
        regime_activity = classify_activity_from_pbr_and_ramp(
            pbr, ramp_idx, args.pbr_threshold, args.ramp_threshold)

        plot_psd_and_ramp(freqs_psd, psd_arr, peak_freq, pbr, log10_pbr,
                          ramp_idx, regime_activity,
                          os.path.join(rep_dir, 'psd_delay.png'),
                          args.delay_start_frac, args.delay_end_frac)
        plot_pc1_trace(H, args.delay_start_frac, args.delay_end_frac,
                       regime_activity,
                       os.path.join(rep_dir, 'pc1_trace.png'))
    else:
        print("    [WARN] Could not run forward pass; metrics set to zero.")

    # Dual
    regime_dual = classify_dual(regime_spectral, regime_activity)

    # Read training stats
    ts = read_train_stats(os.path.join(rep_dir, 'train_stats.txt'))
    result = {
        'regime_spectral':  regime_spectral,
        'regime_activity':  regime_activity,
        'regime_dual':      regime_dual,
        'oi':               oi,
        'log10_pbr':        log10_pbr,
        'pbr':              pbr,
        'ramp_idx':         ramp_idx,
        'peak_freq':        peak_freq,
        'henrici':          h_idx,
        'final_mse':        float(ts.get('final_mse',    'nan')),
        'accuracy':         float(ts.get('accuracy',     'nan')),
        'timing_error':     float(ts.get('timing_error', 'nan')),
        'converged':        ts.get('converged', 'False') == 'True',
    }

    # Save per-network classification
    class_path = os.path.join(rep_dir, 'dynamics_class_v4.txt')
    with open(class_path, 'w') as fh:
        fh.write("=" * 60 + "\n")
        fh.write("Dual-criterion classification (v4)\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"regime_spectral   = {regime_spectral}\n")
        fh.write(f"  oscillation_idx (OI) = {oi:.4f}\n")
        fh.write(f"  dominant_eig         = {lam_dom:.4f}\n")
        fh.write(f"regime_activity   = {regime_activity}\n")
        fh.write(f"  log10(PBR)            = {log10_pbr:.2f}\n")
        fh.write(f"  raw PBR               = {pbr:.1f}\n")
        fh.write(f"  ramp_index            = {ramp_idx:.4f}\n")
        fh.write(f"  peak_frequency        = {peak_freq:.4f}\n")
        fh.write(f"regime_dual       = {regime_dual}\n")
        fh.write("-" * 60 + "\n")
        fh.write(f"henrici_idx     = {h_idx:.4f}\n")
        fh.write(f"final_mse       = {result['final_mse']:.6f}\n")
        fh.write(f"accuracy        = {result['accuracy']:.4f}\n")
        fh.write(f"timing_error    = {result['timing_error']:.2f}\n")

    return result


# ── Tag to row map ───────────────────────────────────────────────────────────
TAG_TO_ROW = {
    'Simple_DM':              'Simple Delayed Binary DM',
    'Simple_DM_Long-short':   'Context-dependent Binary DM',
    'Simple_DM_8_times':      'Multi-interval Amplitude-based',
    'Simple_DM_4_times':      'Multi-interval Distance-based',
    'Integral_DM':            'Windowed Evidence Integration',
}

ROW_ORDER = [
    'Simple Delayed Binary DM',
    'Context-dependent Binary DM',
    'Multi-interval Amplitude-based',
    'Multi-interval Distance-based',
    'Windowed Evidence Integration',
]


# ── Formatting utilities ─────────────────────────────────────────────────────
def fmt_frac(n, total=10):
    return f"{n}/{total}"

def fmt_metric(values):
    vals = [v for v in values if not np.isnan(v)]
    if not vals:
        return "—"
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"

def fmt_index(values, decimals=2):
    vals = [v for v in values if not np.isnan(v)]
    if not vals:
        return "—"
    return f"{np.mean(vals):.{decimals}f}±{np.std(vals):.{decimals}f}"


# ── Text table ───────────────────────────────────────────────────────────────
def write_text_table(table_data, txt_path, args):
    with open(txt_path, 'w') as fh:
        fh.write("Table 2 (v4): Dual-criterion classification (PBR log10 & ramp)\n")
        fh.write("=" * 140 + "\n")
        hdr = (f"{'Task':<36} {'Init':<11} "
               f"{'Osc':>5} {'Ramp':>5} {'Mix':>5}  "
               f"{'OI':>8} {'log10(PBR)':>12} {'RampIdx':>9}  "
               f"{'Accuracy':>12} {'TimErr':>12} {'MSE':>14}\n")
        fh.write(hdr)
        fh.write("-" * 140 + "\n")

        for row_label in ROW_ORDER:
            if row_label not in table_data:
                fh.write(f"{row_label:<36}  (no data)\n")
                continue
            for init in ['Normal', 'Orthogonal']:
                results = table_data[row_label].get(init, [])
                n = len(results)
                if n == 0:
                    fh.write(f"{row_label:<36} {init:<11}  (no data)\n")
                    continue
                n_osc  = sum(1 for r in results if r['regime_dual'] == 'Oscillatory')
                n_ramp = sum(1 for r in results if r['regime_dual'] == 'Ramping/Decaying')
                n_mix  = sum(1 for r in results if r['regime_dual'] == 'Mixed')
                oi_str    = fmt_index([r['oi']         for r in results], 2)
                logpbr_str= fmt_index([r['log10_pbr']  for r in results], 2)
                ramp_str  = fmt_index([r['ramp_idx']   for r in results], 2)
                acc       = fmt_metric([r['accuracy']     for r in results])
                te        = fmt_metric([r['timing_error'] for r in results])
                mse       = fmt_metric([r['final_mse']    for r in results])
                fh.write(
                    f"{row_label:<36} {init:<11} "
                    f"{fmt_frac(n_osc,n):>5} {fmt_frac(n_ramp,n):>5} {fmt_frac(n_mix,n):>5}  "
                    f"{oi_str:>8} {logpbr_str:>12} {ramp_str:>9}  "
                    f"{acc:>12} {te:>12} {mse:>14}\n"
                )
            fh.write("\n")

        fh.write("\n\nThresholds used:\n")
        fh.write(f"  Spectral  — Oscillatory : OI >= {args.osc_threshold}\n")
        fh.write(f"  Spectral  — Mixed       : OI >= {args.mixed_threshold}\n")
        fh.write(f"  Activity  — Oscillatory : PBR >= {args.pbr_threshold}\n")
        fh.write(f"  Activity  — Ramping/Decaying : PBR < {args.pbr_threshold} "
                 f"AND ramp_idx >= {args.ramp_threshold}\n")
        fh.write(f"  Delay window            : [{args.delay_start_frac:.0%} – "
                 f"{args.delay_end_frac:.0%}] of trial\n")
        fh.write("\nDefinitions:\n")
        fh.write("  OI  = |Im(λ*)| / |λ*|,  λ* = dominant eigenvalue of W_rec\n")
        fh.write("  log10(PBR) = log10( max(P(f)) / (median(P(f)) + epsilon) )\n")
        fh.write("  ramp_idx   = |Pearson r| between time and raw PC1 (delay window)\n")
        fh.write("  Dual regime: Oscillatory iff both criteria Oscillatory;\n")
        fh.write("               Ramping/Decaying iff both criteria Ramping/Decaying;\n")
        fh.write("               Mixed otherwise.\n")


# ── LaTeX v4 ─────────────────────────────────────────────────────────────────
LATEX_HEADER_V4 = r"""\begin{table}[htbp]
\centering
\resizebox{\textwidth}{!}{
\begin{tabular}{|p{3.2cm}|c|c|c|c|c|c|c|c|c|}
\hline
\rowcolor{gray!20}
\textbf{Task} & \textbf{Init} &
\textbf{Osc} & \textbf{Ramp} & \textbf{Mixed} &
\textbf{OI} & \textbf{log10(PBR)} & \textbf{Ramp} &
\textbf{Accuracy} & \textbf{Timing Err} & \textbf{MSE} \\
\hline
\rowcolor{gray!10}
\multicolumn{11}{|l|}{%
  \textit{Dual criterion: Oscillatory iff OI $\geq$ osc\_thr \textbf{and}
  PBR $\geq$ pbr\_thr; Ramping/Decaying iff OI $<$ mixed\_thr \textbf{and}
  PBR $<$ pbr\_thr \textbf{and} RampIdx $\geq$ ramp\_thr; Mixed otherwise.}} \\
\hline"""

LATEX_FOOTER_V4 = r"""\end{tabular}
}
\caption{%
  \textbf{Dual-criterion classification of trained RNNs (v4).}
  Spectral criterion: Oscillation Index $\mathrm{OI} = |\mathrm{Im}(\lambda^*)|/|\lambda^*|$,
  with $\lambda^*$ the dominant eigenvalue of $\mathbf{W}^{\mathrm{rec}}$.
  Activity criterion uses the Peak-to-Background Ratio (log10 scale)
  $\log_{10}\mathrm{PBR} = \log_{10}(\max P(f)/(\mathrm{median}\,P(f)+\epsilon))$,
  combined with a ramp index (absolute Pearson correlation between time and the
  first-PC projection of hidden activity during the delay epoch).
  A network is classified as \textbf{Oscillatory} only if both criteria agree on oscillatory,
  as \textbf{Ramping/Decaying} if both agree on ramping/decaying, and as \textbf{Mixed} otherwise.
  Values: mean $\pm$ SD across 10 replicas.
}
\label{tab:regime_classification_v4}
\end{table}"""


def build_latex_table_v4(table_data):
    lines = [LATEX_HEADER_V4]

    for row_label in ROW_ORDER:
        if row_label not in table_data:
            continue
        row_data  = table_data[row_label]
        row_span  = len(row_data)
        first     = True

        for init in ['Normal', 'Orthogonal']:
            if init not in row_data:
                continue
            results = row_data[init]
            n       = len(results)
            n_osc   = sum(1 for r in results if r['regime_dual'] == 'Oscillatory')
            n_ramp  = sum(1 for r in results if r['regime_dual'] == 'Ramping/Decaying')
            n_mix   = sum(1 for r in results if r['regime_dual'] == 'Mixed')
            oi_str     = fmt_index([r['oi']         for r in results], 2)
            logpbr_str = fmt_index([r['log10_pbr']  for r in results], 2)
            ramp_str   = fmt_index([r['ramp_idx']   for r in results], 2)
            acc        = fmt_metric([r['accuracy']     for r in results])
            te         = fmt_metric([r['timing_error'] for r in results])
            mse        = fmt_metric([r['final_mse']    for r in results])

            task_cell = (f"\\multirow{{{row_span}}}{{*}}{{{row_label}}}"
                         if first else "")
            first = False

            line = (
                f"{task_cell} & {init}"
                f" & {fmt_frac(n_osc, n)}"
                f" & {fmt_frac(n_ramp, n)}"
                f" & {fmt_frac(n_mix, n)}"
                f" & {oi_str}"
                f" & {logpbr_str}"
                f" & {ramp_str}"
                f" & {acc}"
                f" & {te}"
                f" & {mse}"
                f" \\\\"
            )
            lines.append(line)
        lines.append("\\hline")

    lines.append(LATEX_FOOTER_V4)
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    results_dir = args.results_dir

    print(f"\n[analyze_ensemble_v4] Scanning {results_dir}")
    print(f"  Spectral thresholds : OI_osc={args.osc_threshold}  "
          f"OI_mix={args.mixed_threshold}")
    print(f"  Activity thresholds : PBR_osc={args.pbr_threshold}  "
          f"ramp_thr={args.ramp_threshold}")
    print(f"  Delay window        : [{args.delay_start_frac:.0%} – "
          f"{args.delay_end_frac:.0%}] of trial")
    print(f"  Trials per network  : {args.n_trials}")

    table_data = {}
    csv_rows   = []

    for entry in sorted(os.listdir(results_dir)):
        tag_dir = os.path.join(results_dir, entry)
        if not os.path.isdir(tag_dir):
            continue

        for init in ['Normal', 'Orthogonal']:
            if entry.endswith('_' + init):
                task_tag = entry[:-(len(init) + 1)]
                break
        else:
            print(f"  [SKIP] Cannot parse init from: {entry}")
            continue

        row_label = None
        for k, v in TAG_TO_ROW.items():
            if task_tag == k:
                row_label = v
                break
        if row_label is None:
            print(f"  [SKIP] Unknown task tag: {task_tag!r}")
            continue

        print(f"\n  Task={row_label!r}  Init={init}")
        table_data.setdefault(row_label, {}).setdefault(init, [])

        rep_dirs = sorted(
            d for d in os.listdir(tag_dir)
            if d.startswith('net_') and os.path.isdir(os.path.join(tag_dir, d))
        )
        for rep_name in rep_dirs:
            rep_dir = os.path.join(tag_dir, rep_name)
            rep_idx = int(rep_name.split('_')[1])
            print(f"    {rep_name} … ", end='', flush=True)

            result = process_network(rep_dir, args)
            if result is None:
                print("SKIPPED")
                continue

            print(
                f"spectral={result['regime_spectral']:18s}  "
                f"activity={result['regime_activity']:18s}  "
                f"dual={result['regime_dual']:18s}  "
                f"OI={result['oi']:.3f}  log10(PBR)={result['log10_pbr']:.2f}  "
                f"ramp={result['ramp_idx']:.2f}  acc={result['accuracy']:.3f}"
            )
            table_data[row_label][init].append(result)
            csv_rows.append({
                'task':             row_label,
                'init':             init,
                'replica':          rep_idx,
                'regime_spectral':  result['regime_spectral'],
                'regime_activity':  result['regime_activity'],
                'regime_dual':      result['regime_dual'],
                'oi':               result['oi'],
                'log10_pbr':        result['log10_pbr'],
                'pbr':              result['pbr'],
                'ramp_idx':         result['ramp_idx'],
                'peak_freq':        result['peak_freq'],
                'henrici':          result['henrici'],
                'accuracy':         result['accuracy'],
                'timing_error':     result['timing_error'],
                'final_mse':        result['final_mse'],
                'converged':        result['converged'],
            })

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, 'table2_v4_summary.csv')
    fieldnames = ['task', 'init', 'replica',
                  'regime_spectral', 'regime_activity', 'regime_dual',
                  'oi', 'log10_pbr', 'pbr', 'ramp_idx', 'peak_freq', 'henrici',
                  'accuracy', 'timing_error', 'final_mse', 'converged']
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  CSV  → {csv_path}")

    # ── Text table ────────────────────────────────────────────────────────────
    txt_path = os.path.join(results_dir, 'table2_v4.txt')
    write_text_table(table_data, txt_path, args)
    print(f"  TXT  → {txt_path}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    latex_path = os.path.join(results_dir, 'table2_v4_latex.tex')
    with open(latex_path, 'w') as fh:
        fh.write(build_latex_table_v4(table_data))
    print(f"  TEX  → {latex_path}")

    # ── Typical dynamics figure ───────────────────────────────────────────────
    if csv_rows:
        plot_typical_dynamics(csv_rows, results_dir)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY — Table 2 (v4, PBR log10 + ramp index)")
    print("=" * 80)
    with open(txt_path) as fh:
        print(fh.read())


if __name__ == '__main__':
    main()
