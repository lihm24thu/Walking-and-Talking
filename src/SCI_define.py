# Compute Saliency Consistency Index (SCI) for Fig. 5

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

def compute_cross_state_fscore_for_task(
    patient_ids,
    decoding_csv_root_detection,
    decoding_csv_root_decoding,
    detection_file_name, # For example: f"{pid}_detection.csv"
    decoding_file_name_bed, # For example: f"{pid}_decoding_bed.csv"
    decoding_file_name_fre, # For example: f"{pid}_decoding_fre.csv"
    save_root,
    bad_channels_dict=None,
    decoding_modes=("rest", "moving"),
    step_select=0,
    task_type="detection",  # "detection" or "decoding"
    r2_threshold=0.0
):
    """
    SCI definition:
        SCI = 2 * z1 * z2 / [(z1 + z2) + 1e-6]

    File reading code can be adjusted flexibly.
    """
    os.makedirs(save_root, exist_ok=True)

    bad_channels_dict = bad_channels_dict or {}

    def normalize_importance(x):
        x = x.astype(float)
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def compute_functional_stability(dfA, dfB, value_col="importance"):

        dfA = dfA.copy()
        dfB = dfB.copy()
    
        # Normalization
        dfA["norm"] = normalize_importance(dfA[value_col])
        dfB["norm"] = normalize_importance(dfB[value_col])
    
        # Z-score
        dfA["z"] = dfA["norm"]
        dfB["z"] = dfB["norm"]
    
        merged = pd.merge(dfA[["channel", "z"]],
                          dfB[["channel", "z"]],
                          on="channel", suffixes=("_A", "_B"))
        z1, z2 = merged["z_A"].values, merged["z_B"].values
    
        diff = 2 * z1 * z2
        amp = (z1 + z2) + 1e-6

        # Functional similarity metric
        stability_raw = diff / amp
        stability = np.clip(stability_raw, 0, 1)
    
        merged["stability_func"] = stability
        return merged[["channel", "stability_func"]]

    for pid in patient_ids:
        bad_channels = set(bad_channels_dict.get(pid, []))
        dec_data = {}

        # speech detection
        if task_type == "detection":
            for dec_mode in decoding_modes:
                dec_path = os.path.join(
                    decoding_csv_root_detection,
                    f"detection_{dec_mode}",
                    f"detection_{dec_mode}",
                    detection_file_name
                )
                if not os.path.exists(dec_path):
                    print(f"[Skip] Missing detection CSV: {dec_path}")
                    continue

                df = pd.read_csv(dec_path)
                if "channel" not in df.columns or "importance" not in df.columns:
                    raise ValueError(f"{dec_path} missing 'channel' or 'importance'")

                if "step" in df.columns:
                    df = df[df["step"] == step_select].copy()
                df = df[~df["channel"].isin(bad_channels)]

                if r2_threshold > 0:
                    df = df[df["importance"] > r2_threshold]

                dec_data[dec_mode] = df

        # Mel-spectrogram decoding
        elif task_type == "decoding":
            for dec_mode in decoding_modes:
                if dec_mode == "rest":
                    dec_path = os.path.join(
                        decoding_csv_root_decoding,
                        dec_mode,
                        decoding_file_name_bed
                    )
                else:
                    dec_path = os.path.join(
                        decoding_csv_root_decoding,
                        dec_mode,
                        decoding_file_name_fre
                    )

                if not os.path.exists(dec_path):
                    print(f"[Skip] Missing decoding CSV: {dec_path}")
                    continue

                df = pd.read_csv(dec_path)
                if "channel" not in df.columns or "importance" not in df.columns:
                    raise ValueError(f"{dec_path} missing 'channel' or 'importance'")

                df = df[~df["channel"].isin(bad_channels)]
                if r2_threshold > 0:
                    df = df[df["importance"] > r2_threshold]

                dec_data[dec_mode] = df
        else:
            raise ValueError("❌ task_type has to be 'detection' or 'decoding'")

        if len(dec_data) < 2:
            print(f"[Skip] Missing both states for {pid}")
            continue

        dfA, dfB = dec_data[decoding_modes[0]], dec_data[decoding_modes[1]]
        stab_df = compute_functional_stability(dfA, dfB, value_col="importance")

        # Save .csv files
        indiv_save = os.path.join(save_root, f"{task_type}_SEEG{pid}_functional_stability.csv")
        stab_df.to_csv(indiv_save, index=False)
        print(f"✅ Saved per-channel functional stability: {indiv_save}")
