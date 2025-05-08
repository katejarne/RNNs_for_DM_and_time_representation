import numpy as np
from scipy.stats import entropy


def compute_SI(activity, entrpy_bins=20, window_size=5, r_threshold=0.1):
    """
    Direct adaptation of the original function of A. Emin Orhan and
    Wei Ji Ma https://doi.org/10.1038/s41593-018-0314-y

    Parameters:
    activity : numpy.ndarray (neurons x time)
    entrpy_bins : número de bins para la entropía (20 en el paper)
    window_size : ventana alrededor del pico para calcular relación señal-fondo (5)
    r_threshold : umbral para considerar neuronas activas (0.1)

    Returns:
    SI : float (sequentiality index para each input)
    """
    # Reformat to (trials, time, units) - we assume 1 trial per sample
    hidr = np.transpose(activity)[np.newaxis, :, :]  # (1, T, N)

    bs = hidr.shape[0]  # 1 trial
    ts = hidr.shape[1]  # time points
    n_units = hidr.shape[2]  # Neurons

    SI_trial_vec = np.zeros(bs)

    for b in range(bs):
        hidr_t = hidr[b, :, :]  # (T, N)

        # 1. Select neurons with sufficient activity
        selected_indx = np.nonzero(np.mean(hidr_t, axis=0) > r_threshold)[0]
        hidr_t = hidr_t[:, selected_indx]  # (T, active_units)

        if hidr_t.size == 0:
            SI_trial_vec[b] = np.nan
            continue

        # 2. Calculate peak times
        peak_times = np.argmax(hidr_t, axis=0)  # (active_units,)

        # 3. Calculate entropy of the peak distribution
        hist = np.histogram(peak_times, bins=entrpy_bins, range=(0, ts))[0]
        entrpy = entropy(hist + 0.1)  # Suavizado para evitar log(0)

        # 4. Calculate signal-to-background ratio (ridge-to-background)
        window_size = min(window_size, ts//2)  # valid window
        r2b_ratio = np.zeros(len(selected_indx))

        for nind in range(len(selected_indx)):
            pt = peak_times[nind]
            start = max(0, pt - window_size//2)
            end = min(ts, pt + window_size//2)

            # Average in the peak window
            ridge = np.mean(hidr_t[start:end, nind])

            # media outside the window (background)
            mask = np.zeros(ts, dtype=bool)
            mask[start:end] = True
            backgr = np.mean(hidr_t[~mask, nind])

            # Avoid log(0) and division by zero
            ridge = max(ridge, 1e-6)
            backgr = max(backgr, 1e-6)

            r2b_ratio[nind] = np.log(ridge) - np.log(backgr)

        # 5. Combine metrics
        SI_trial_vec[b] = np.nanmean(r2b_ratio) + entrpy

    return np.nanmean(SI_trial_vec)
