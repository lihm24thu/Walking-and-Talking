#### Corrsponding to Fig. 5
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.spatial.distance import cdist
from tqdm import tqdm
import seaborn as sns

import nibabel as nib
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import resample_img
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn import plotting

# Fig. 5a,d 

## Visualization
def sci_across_mni_xyz(df_all):
    """
    df_all: data frame, columns include:
        "channel", "region", "region_color", "mni_x", "mni_y", "mni_z", "stability_func"
            "region_color" represents the previously defined color for the corresponding region.
            "stability_func" represents the SCI calculated for each channel.
    """
    plot_df = df_all.dropna(
        subset=["mni_x", "mni_y", "mni_z", "stability_func"]
    ).copy()

    coords = ["mni_x", "mni_y", "mni_z"]
    labels = ["MNI X (mm)", "MNI Y (mm)", "MNI Z (mm)"]

    for coord, xlabel in zip(coords, labels):

        plt.figure(figsize=(15, 5))

        plt.scatter(
            plot_df[coord],
            plot_df["stability_func"],
            c=plot_df["region_color"],
            s=100,
            edgecolors="k",
            linewidths=0.3,
            alpha=0.6,
        )

        smooth = lowess(
            plot_df["stability_func"],
            plot_df[coord],
            frac=0.35,
            return_sorted=True
        )

        plt.plot(
            smooth[:, 0],
            smooth[:, 1],
            color="black",
            linewidth=2,
            label="LOWESS"
        )

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_linewidth(2.5)
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel("Functional stability")
        plt.title(f"{xlabel} Projection")
        plt.xlim(-85, 85)

        plt.tight_layout()
        plt.show()


# Fig. 5b,e

## Visualization
def asymm_report_with_permutation(df_group, n_perms, save_root):
    """
    df_group: data frame, with columns:
        "x": represents the MNI x-coordinate.
        "stability_func": represents the SCI calculated.
    """
    os.makedirs(save_root, exist_ok=True)
    df = df_group.copy()
    df["hemisphere"] = np.where(df["x"] < 0, "Left", "Right")
    
    obs_left_mean = df.loc[df["hemisphere"] == "Left", "stability_func"].mean()
    obs_right_mean = df.loc[df["hemisphere"] == "Right", "stability_func"].mean()
    observed_diff = obs_left_mean - obs_right_mean
    
    print(f"\n🎲 Running {n_perms} permutations for hemisphere bias...")
    perm_diffs = []
    all_vals = df["stability_func"].values
    
    for _ in range(n_perms):
        shuffled_vals = np.random.permutation(all_vals)
        df["temp_shuffled"] = shuffled_vals
        p_left = df.loc[df["hemisphere"] == "Left", "temp_shuffled"].mean()
        p_right = df.loc[df["hemisphere"] == "Right", "temp_shuffled"].mean()
        perm_diffs.append(p_left - p_right)
    
    perm_p_val = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    plt.figure(figsize=(12, 8))
    sns.histplot(perm_diffs, kde=True, color="gray", alpha=0.5)
    plt.axvline(observed_diff, color="red", linestyle="--", label=f"Observed Diff: {observed_diff:.3f}")
    plt.title(f"Hemisphere Permutation Test (p={perm_p_val:.4f})", fontsize=18)
    plt.xlabel("Difference (Left - Right)", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)

    ax = plt.gca()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(os.path.join(save_root, "permutation_hemisphere_test.png"))
    plt.close()

# Fig. 5c, f

## Permutation test and zscored SCI on MNI152 brain
def z_monte_carlo(df_all, vox_v=5, sigma = 6.0, n_perm = 500):
    """
    df_all: data frame, columns include:
        "channel", "mni_x", "mni_y", "mni_z", "stability_func"
            "stability_func" represents the SCI calculated for each channel.
    """
    plot_df = df_all.dropna(
        subset=["mni_x", "mni_y", "mni_z", "stability_func"]
    ).copy()

    coords = plot_df[["mni_x", "mni_y", "mni_z"]].values.astype(float)
    values = plot_df["stability_func"].values.astype(float)

    mask_img_1mm = load_mni152_brain_mask()
    target_affine = mask_img_1mm.affine.copy()

    target_affine[:3, :3] = np.diag([vox_v,vox_v,vox_v])

    mask_img_5mm = resample_img(
        mask_img_1mm,
        target_affine=target_affine,
        interpolation="nearest"
    )

    print(mask_img_5mm.shape)
    print(mask_img_5mm.affine)

    mask = mask_img_5mm.get_fdata().astype(bool)

    ijk = np.argwhere(mask)

    grid_points = nib.affines.apply_affine(
        mask_img_5mm.affine,
        ijk
    )

    # Compute vobs
    dist = cdist(
        grid_points,
        coords
    )

    weights = np.exp(
        -(dist**2)/(2*sigma**2)
    )

    Vobs = (
        weights @ values
    ) / (
        weights.sum(axis=1)+1e-12
    )

    Vobs_volume = np.zeros(mask.shape)
    Vobs_volume[mask] = Vobs

    Vnull = np.zeros(
        (n_perm, len(grid_points)),
        dtype=np.float32
    )

    # Monte Carlo permutation test
    for i in tqdm(range(n_perm)):

        perm = np.random.permutation(values)
        Vnull[i] = (
            weights @ perm
        ) / (
            weights.sum(axis=1)+1e-12
        )

    null_mean = Vnull.mean(axis=0)
    null_std = Vnull.std(axis=0)

    null_std[null_std < 1e-12] = 1e-12

    # Compute zsocred SCI
    Z = (
        Vobs
        -
        null_mean
    ) / null_std

    Z_volume = np.zeros(mask.shape)
    Z_volume[mask] = Z

    mask_img = mask_img_5mm
    Vobs_img = nib.Nifti1Image(
        Vobs_volume,
        affine=mask_img.affine
    )

    Z_img = nib.Nifti1Image(
        Z_volume,
        affine=mask_img.affine
    )

    # Save .gz files
    nib.save(
        Vobs_img,
        "Spatial_Stability_Map.nii.gz"
    )

    nib.save(
        Z_img,
        "Spatial_Stability_ZMap.nii.gz"
    )

