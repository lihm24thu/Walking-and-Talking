import numpy as np
import pandas as pd
import os

from sklearn.metrics import r2_score
from scipy.stats import zscore
from sklearn.linear_model import Ridge

from tqdm.auto import tqdm


# Single-channel Linear Decoding

## Build neural lag features for ridge regression
def build_neural_lag_features(sig_ds, n_lags):
    """
    Neural lag expansion for decoding
    sig_ds: [n_channels, T]
    return: [T, n_channels * n_lags]
    """
    n_channels, T = sig_ds.shape
    X = np.zeros((T, n_channels * n_lags))
    for lag in range(n_lags):
        if lag == 0:
            X[:, lag*n_channels:(lag+1)*n_channels] = sig_ds.T
        else:
            X[lag:, lag*n_channels:(lag+1)*n_channels] = sig_ds[:, :-lag].T
    return X


## Speech detection
def decoding_ridge_detection(
    seeg_trials,
    detection_trials,
    all_channels,
    bad_channels,
    file_name,
    seeg_sr=400,
    detection_sr=100,
    alpha=1.0,
    save_dir=None,
):
    """
    Single-channel ridge encoding model for speech detection.
    Each electrode is decoded independently.
    Neural predictor: One channel + previous 600 ms history.
    Target: Binary speech incident label, speaking or non-speaking.
    Evaluation: Leave-one-trial-out R².
    """

    # 600 ms causal history
    # 100 Hz sampling:

    n_lags = int(0.6 * detection_sr)
    X_trials = []
    Y_trials = []

    # Preprocess each trial
    for sig, detect in zip(seeg_trials, detection_trials):

        sig = sig.numpy() if hasattr(sig, "numpy") else sig
        detect = detect.numpy() if hasattr(detect, "numpy") else detect

        if detect.ndim != 1:
            continue

        # Align time
        down_factor = seeg_sr // detection_sr

        T = min(
            detect.shape[0],
            sig.shape[1] // down_factor
        )

        sig = sig[:, :T * down_factor]
        detect = detect[:T]

        # Downsample: 400 Hz -> 100 Hz
        sig_ds = sig.reshape(
            sig.shape[0],
            T,
            down_factor
        ).mean(axis=2)

        # Channel-wise normalization
        sig_ds = zscore(
            sig_ds,
            axis=1
        )

        detect = zscore(
            np.asarray(detect, dtype=np.float32)
        )

        X = build_neural_lag_features(
            sig_ds,
            n_lags
        )

        X_trials.append(X)
        Y_trials.append(detect)

    n_trials = len(X_trials)

    keep_idx = [
        i for i, ch in enumerate(all_channels)
        if ch not in bad_channels
    ]

    used_ch_names = [
        all_channels[i]
        for i in keep_idx
    ]

    n_channels = len(all_channels)

    X_trials_channel = {}

    for ch_idx in keep_idx:

        X_trials_channel[ch_idx] = [
            X[:, ch_idx::n_channels]
            for X in X_trials
        ]

    # Decode channel by channel
    r2_array = np.zeros(len(keep_idx))

    for ch_i, ch_idx in enumerate(
        tqdm(
            keep_idx,
            desc=f"channels"
        )
    ):
        y_true_all = []
        y_pred_all = []

        # Leave-one-trial-out CV
        for test_i in range(n_trials):

            X_test = X_trials_channel[ch_idx][test_i]
            y_test = Y_trials[test_i]

            X_train = np.concatenate(
                [
                    X_trials_channel[ch_idx][i]
                    for i in range(n_trials)
                    if i != test_i
                ],
                axis=0
            )
            y_train = np.concatenate(
                [
                    Y_trials[i]
                    for i in range(n_trials)
                    if i != test_i
                ],
                axis=0
            )

            model = Ridge(
                alpha=alpha
            )
            model.fit(
                X_train,
                y_train
            )
            y_pred = model.predict(
                X_test
            )

            y_true_all.append(y_test)
            y_pred_all.append(y_pred)

        # Overall channel performance
        y_true_all = np.concatenate(
            y_true_all
        )

        y_pred_all = np.concatenate(
            y_pred_all
        )

        r2_array[ch_i] = r2_score(
            y_true_all,
            y_pred_all
        )

    mean_r2 = np.mean(r2_array)

    print(
        f"mean R²={mean_r2:.4f}, "
        f"max R²={np.max(r2_array):.4f}"
    )

    if save_dir:
        os.makedirs(
            save_dir,
            exist_ok=True
        )

        pd.DataFrame(
            {
                "Channel": used_ch_names,
                "R2": r2_array
            }
        ).to_csv(
            os.path.join(
                save_dir,
                f"{file_name}"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    return {
        "r2_array": r2_array,
        "channel_names": used_ch_names,
        "mean_r2": mean_r2
    }


## Mel-spectrogram envelope decoding
def decoding_ridge_envelope(
    seeg_trials,
    mel_trials,
    all_channels,
    bad_channels,
    file_name,
    seeg_sr=400,
    mel_sr=100,
    alpha=1.0,
    save_dir=None,
):
    """
    Single-channel linear speech envelope decoding.
    Predictor:
        One neural channel with strictly causal
        600 ms history.
    Target:
        Summed Mel spectral energy.
    Evaluation:
        Leave-one-trial-out R².
    Regularization:
        Fixed Ridge alpha=1.0
    """

    # 600 ms causal history
    # Mel sampling rate = 100 Hz

    n_lags = int(0.6 * mel_sr)
    X_trials = []
    Y_trials = []

    for sig, mel in zip(seeg_trials, mel_trials):

        sig = sig.numpy() if hasattr(sig, "numpy") else sig
        mel = mel.numpy() if hasattr(mel, "numpy") else mel

        if mel.ndim == 1:
            continue

        if mel.shape[0] == 40:
            mel = mel.T

        # Align neural and Mel time
        down_factor = seeg_sr // mel_sr

        T = min(
            sig.shape[1] // down_factor,
            mel.shape[0]
        )

        sig = sig[:, :T * down_factor]
        mel = mel[:T]

        mel = np.asarray(
            mel,
            dtype=np.float32
        )
        mel_energy = np.sum(
            mel,
            axis=1
        )
        mel_energy = zscore(
            mel_energy
        )

        # Neural preprocessing
        sig_ds = sig.reshape(
            sig.shape[0],
            T,
            down_factor
        ).mean(axis=2)


        sig_ds = zscore(
            sig_ds,
            axis=1
        )

        # Construct causal lag features
        X = build_neural_lag_features(
            sig_ds,
            n_lags
        )

        X_trials.append(X)
        Y_trials.append(
            mel_energy
        )

    n_trials = len(X_trials)

    keep_idx = [
        i for i, ch in enumerate(all_channels)
        if ch not in bad_channels
    ]

    used_ch_names = [
        all_channels[i]
        for i in keep_idx
    ]
    n_channels = len(all_channels)

    # Pre-extract single-channel features
    X_trials_channel = {}

    for ch_idx in keep_idx:

        X_trials_channel[ch_idx] = [
            X[:, ch_idx::n_channels]
            for X in X_trials
        ]

    # Channel-wise decoding
    r2_array = np.zeros(
        len(used_ch_names)
    )

    best_alpha_per_ch = np.ones(
        len(used_ch_names)
    ) * alpha

    for ch_i, ch_idx in enumerate(
        tqdm(
            keep_idx,
            desc=f"decoding"
        )
    ):
        y_true_all = []
        y_pred_all = []

        # Leave-one-trial-out CV
        for test_i in range(n_trials):
            X_test = X_trials_channel[ch_idx][test_i]
            y_test = Y_trials[test_i]

            X_train = np.concatenate(
                [
                    X_trials_channel[ch_idx][i]
                    for i in range(n_trials)
                    if i != test_i
                ],
                axis=0
            )
            y_train = np.concatenate(
                [
                    Y_trials[i]
                    for i in range(n_trials)
                    if i != test_i
                ],
                axis=0
            )

            # Fixed ridge regression
            model = Ridge(
                alpha=alpha
            )
            model.fit(
                X_train,
                y_train
            )

            y_pred = model.predict(
                X_test
            )

            y_true_all.append(
                y_test
            )
            y_pred_all.append(
                y_pred
            )

        y_true_all = np.concatenate(
            y_true_all
        )
        y_pred_all = np.concatenate(
            y_pred_all
        )

        r2_array[ch_i] = r2_score(
            y_true_all,
            y_pred_all
        )

    mean_r2 = np.nanmean(
        r2_array
    )

    print(
        f"mean R² = {mean_r2:.4f}, "
        f"max R² = {np.nanmax(r2_array):.4f}"
    )

    if save_dir:
        os.makedirs(
            save_dir,
            exist_ok=True
        )

        pd.DataFrame(
            {
                "Channel": used_ch_names,
                "R2": r2_array,
                "BestAlpha": best_alpha_per_ch
            }
        ).to_csv(
            os.path.join(
                save_dir,
                f"{file_name}"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    return dict(
        r2_array=r2_array,
        best_alpha=best_alpha_per_ch,
        channel_names=used_ch_names,
        mean_r2=mean_r2
    )