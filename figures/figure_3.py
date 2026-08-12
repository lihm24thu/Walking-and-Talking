#### Visualization of linear_decoding.py results

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import wilcoxon, pearsonr, spearmanr, kruskal
import scikit_posthocs as sp

# Fig. 3a,g Comparison of patient-level decoding performance between bedbound and mobility conditions

## Statistical test
def compute_p_value_patient(df_all):
    """
    df_all: data frame that includes R² of all channels.
        "bed_r2": R² in in-bed scenario.
        "free_r2": R² in freely moving scenario.
    """
    pearson_results = []
    spearman_results = []

    for pid, df_pid in df_all.groupby("patient"):

        r, p = pearsonr(
            df_pid["bed_r2"],
            df_pid["free_r2"]
        )

        rho, p_s = spearmanr(
            df_pid["bed_r2"],
            df_pid["free_r2"]
        )

        pearson_results.append({
            "patient": pid,
            "r": r,
            "p": p
        })

        spearman_results.append({
            "patient": pid,
            "rho": rho,
            "p": p_s
        })

    pearson_df = pd.DataFrame(pearson_results)
    spearman_df = pd.DataFrame(spearman_results)

    rho_values = spearman_df["rho"]

    stat, p = wilcoxon(
        rho_values
    )

    return p

## Visualization
def patient_level_linear_decoding_performance(df_all, patient_mean):
    """
    Paired patient-level plot
    Each line = one patient
    Each dot = mean R2 of one patient
    """

    plt.figure(figsize=(6,8))
    x_pos = [0, 1]

    bed_color = "#4C72B0"
    free_color = "#DD8452"

    # Plot paired lines and points
    for _, row in patient_mean.iterrows():
        # line connecting the same patient
        plt.plot(
            x_pos,
            [
                row["bed_r2"],
                row["free_r2"]
            ],
            color="gray",
            alpha=0.6,
            linewidth=2,
            zorder=1
        )

        # bed point
        plt.scatter(
            0,
            row["bed_r2"],
            color=bed_color,
            edgecolor="black",
            linewidth=0.8,
            s=160,
            zorder=2
        )

        # free point
        plt.scatter(
            1,
            row["free_r2"],
            color=free_color,
            edgecolor="black",
            linewidth=0.8,
            s=160,
            zorder=2
        )

    # Significance annotation
    y_max = max(
        patient_mean["bed_r2"].max(),
        patient_mean["free_r2"].max()
    )
    y_min = min(
        patient_mean["bed_r2"].min(),
        patient_mean["free_r2"].min()
    )

    height = y_max - y_min
    line_y = y_max + 0.15 * height

    plt.plot(
        [0, 0, 1, 1],
        [
            line_y,
            line_y + 0.01*height,
            line_y + 0.01*height,
            line_y
        ],
        color="black",
        linewidth=1.8
    )

    p = compute_p_value_patient(df_all)

    if p < 0.001:
        sig_text = "***"
    elif p < 0.01:
        sig_text = "**"
    elif p < 0.05:
        sig_text = "*"
    else:
        sig_text = f"p={p:.3f}"

    plt.text(
        0.5,
        line_y + 0.02*height,
        sig_text,
        ha="center",
        va="bottom",
    )

    ax = plt.gca()

    ax.set_xticks([0,1])
    ax.set_xticklabels(
        ["Bedbound", "Mobility"]
    )

    ax.set_ylabel(
        r"$R^2$"
    )

    ax.tick_params(
        axis="y"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(2.2)
    ax.spines["bottom"].set_linewidth(2.2)

    plt.xlim(-0.3,1.3)
    plt.tight_layout()
    plt.show()

# Fig 3b,h Boxplots summarize the distribution of channel-wise decoding performance (R²) under the in-bed and in-mobility conditions.

## Visualization
def channel_level_linear_decoding_performance(df_all):
    """
    df_all: data frame that includes R² of all channels.
        "bed_r2": R² in in-bed scenario.
        "free_r2": R² in freely moving scenario.
    """
    plot_df = pd.melt(
        df_all,
        value_vars=["bed_r2", "free_r2"],
        var_name="Condition",
        value_name="R2"
    )

    plt.figure(figsize=(6,8))

    box_colors = ["#4C72B0", "#DD8452"]

    sns.boxplot(
        data=plot_df,
        x="Condition",
        y="R2",
        showfliers=False,
        width=0.5,
        palette=box_colors,
        boxprops={'edgecolor': 'black', 'linewidth': 1.5},
        whiskerprops={'color': 'black', 'linewidth': 1.5},
        capprops={'color': 'black', 'linewidth': 1.5},
        medianprops={'color': 'white', 'linewidth': 2}
    )

    sns.stripplot(
        data=plot_df,
        x="Condition",
        y="R2",
        color="black",
        alpha=0.35,
        size=3,
        jitter=0.25
    )

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)

    ax.tick_params(axis='both', which='major')
    plt.xlabel("Condition")
    plt.ylabel(r"$R^2$")

    # Significance annotation
    y_max = plot_df["R2"].max()
    y_min = plot_df["R2"].min()

    line_y = y_max + 0.05 * (y_max - y_min)

    plt.plot(
        [0, 0, 1, 1],
        [line_y, line_y + 0.01, line_y + 0.01, line_y],
        color="black",
        linewidth=1.5
    )

    stat, p = wilcoxon(
    df_all["bed_r2"],
    df_all["free_r2"]
)

    if p < 0.001:
        text = "***"
    elif p < 0.01:
        text = "**"
    elif p < 0.05:
        text = "*"
    else:
        text = f"p={p:.3f}"

    plt.text(
        0.5,
        line_y + 0.012,
        text,
        ha="center",
        va="bottom",
    )

    plt.show()

# Fig. 3c,i Boxplots show the distribution of channel-wise decoding performance differences (ΔR² = R²Bed − R²Mobility) across cortical regions, plotted separately for the left and right hemispheres. 

## Statistical test
def region_pairwise_test(df_region_expanded, hemi):
    """
    df_region_expanded: A data frame generated from `df_all`, containing the four columns "channel_name", "region", "region_single", and "Hemisphere", with each row displaying data for a corresponding channel.
    hemi: a str defining the targeted hemisphere, for example, "Left" or "Right", depending on how you defined "Hemisphere" in df_region_expanded.
    """
    # select hemisphere
    data = df_region_expanded[
        df_region_expanded["Hemisphere"] == hemi
    ].copy()

    region_counts = (
        data["region_single"]
        .value_counts()
    )

    valid_regions = (
        region_counts[
            region_counts >= 3
        ]
        .index
        .tolist()
    )

    data = data[
        data["region_single"]
        .isin(valid_regions)
    ]

    print(
        f"{hemi}:",
        len(data),
        "channels"
    )

    # Overall Kruskal-Wallis
    groups = [
        data.loc[
            data["region_single"]==r,
            "delta_r2"
        ]
        for r in valid_regions
    ]

    H,p_kw = kruskal(*groups)

    print(
        f"Kruskal-Wallis H={H:.3f}, p={p_kw:.5f}"
    )

    # Dunn pairwise test
    dunn = sp.posthoc_dunn(
        data,
        val_col="delta_r2",
        group_col="region_single",
        p_adjust="fdr_bh"
    )
    return data, dunn, p_kw

## Visualization
### Add significance bar
def add_significance_bar(
    ax,
    dunn_matrix,
    regions,
    y_start,
    step=0.015
):

    y = y_start
    comparisons=[]

    for i,r1 in enumerate(regions):
        for j,r2 in enumerate(regions):
            if j <= i:
                continue
            p = dunn_matrix.loc[
                r1,r2
            ]
            if p < 0.05:
                comparisons.append(
                    (
                        r1,
                        r2,
                        p
                    )
                )

    comparisons = sorted(
        comparisons,
        key=lambda x:x[2]
    )

    for r1,r2,p in comparisons:
        x1 = regions.index(r1)
        x2 = regions.index(r2)

        if p <0.001:
            text="***"
        elif p<0.01:
            text="**"
        else:
            text="*"

        ax.plot(
            [
                x1,
                x1,
                x2,
                x2
            ],
            [
                y,
                y+step*0.2,
                y+step*0.2,
                y
            ],
            lw=2,
            c="black"
        )

        ax.text(
            (x1+x2)/2,
            y+step*0.1-0.002,
            text,
            ha="center",
            va="bottom",
            fontsize=35
        )
        y += step

### Plot figures
def region_level_linear_decoding_performance(df_region_expanded):
    region_colors = {
        "Frontal":   (0.85, 0.30, 0.30),
        "Temporal":  (0.30, 0.45, 0.85),
        "Parietal":  (0.30, 0.70, 0.45),
        "Occipital": (0.55, 0.40, 0.75),
        "Limbic":    (0.85, 0.60, 0.30),
        "Insula and Operculum": (0.70, 0.55, 0.45),
        "Others":    (0.60, 0.60, 0.60),
    }

    q_low, q_high = np.percentile(
        df_region_expanded["delta_r2"],
        [1, 99]
    )

    margin = 0.03 * (q_high - q_low)

    global_ylim_low = q_low - margin
    global_ylim_high = q_high + margin

    for hemi in ["Left", "Right"]:

        plot_df, dunn_matrix, p_kw = region_pairwise_test(
            df_region_expanded,
            hemi
        )

        regions = sorted(plot_df["region_single"].unique())

        palette_colors = [
            region_colors.get(r, (0.5,0.5,0.5))
            for r in regions
        ]

        q_low, q_high = np.percentile(
            plot_df["delta_r2"],
            [5, 95]
        )

        plot_df_vis = plot_df[
            plot_df["delta_r2"].between(
                global_ylim_low,
                global_ylim_high
            )
        ].copy()

        fig, ax = plt.subplots(
            figsize=(9, 15)
        )

        sns.boxplot(
            ax=ax,
            data=plot_df_vis,
            x="region_single",
            y="delta_r2",
            order=regions,
            palette=palette_colors,
            showfliers=False,
            width=0.6,
            boxprops=dict(
                edgecolor="black",
                linewidth=2.5
            ),
            whiskerprops=dict(
                color="black",
                linewidth=2.5
            ),
            capprops=dict(
                color="black",
                linewidth=2.5
            ),
            medianprops=dict(
                color="white",
                linewidth=2.8
            )
        )

        sns.stripplot(
            ax=ax,
            data=plot_df_vis,
            x="region_single",
            y="delta_r2",
            order=regions,
            color="black",
            alpha=0.35,
            size=7,
            jitter=0.25
        )

        add_significance_bar(
            ax,
            dunn_matrix,
            regions,
            y_start = global_ylim_high + 0.02,
            step=0.02
        )

        ax.axhline(
            0,
            linestyle="--",
            color="gray",
            linewidth=2
        )

        ax.set_ylabel(
            r"$\Delta R^2$ (Bed - Mobility)"
        )

        ax.set_xlabel(
            "Brain region"
        )

        ax.set_xticklabels(
            regions,
            rotation=45,
            ha="right"
        )

        ax.tick_params(
            axis='y'
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(3)
        ax.spines["bottom"].set_linewidth(3)

        ax.set_ylim(
            global_ylim_low,
            global_ylim_high + 0.1
        )

        ax.set_title(
            f"{hemi} hemisphere\nKruskal-Wallis p = {p_kw:.3g}",
            fontsize=17
        )

        plt.tight_layout()
        plt.show()


# Fig 3e-f, l-m 

## Statistic test
def pearson_compute(df_all):

    # Overall Pearson correlation
    r_all, p_all = pearsonr(
        df_all["bed_r2"],
        df_all["free_r2"]
    )

    print("="*60)
    print("Overall")
    print(f"Pearson r = {r_all:.3f}")
    print(f"p = {p_all:.3e}")

    # Patient-wise Pearson correlation
    pearson_results = []

    for pid, sub in df_all.groupby("patient"):
        if len(sub) < 3:
            continue

        r, p = pearsonr(
            sub["bed_r2"],
            sub["free_r2"]
        )

        pearson_results.append({
            "patient": pid,
            "n_channels": len(sub),
            "pearson_r": r,
            "p": p
        })

    pearson_df = pd.DataFrame(pearson_results)

    print("\nPatient-wise Pearson correlation")
    print(pearson_df)

    print("\nSummary")
    print(
        f"Mean r = {pearson_df['pearson_r'].mean():.3f} ± "
        f"{pearson_df['pearson_r'].std():.3f}"
    )

## Visualization
def all_channels_with_regions_color(df_all):
    """
    df_all: data frame that includes R² of all channels.
        "bed_r2": R² in in-bed scenario.
        "free_r2": R² in freely moving scenario.
        "region": the brain region of the channel
        "region_color": previously defined colors for all regions
    """
    plt.figure(figsize=(12, 12))

    regions = sorted(df_all["region"].unique())

    for region in regions:
        sub = df_all[df_all["region"] == region]

        color = sub["region_color"].iloc[0]

        plt.scatter(
            sub["bed_r2"],
            sub["free_r2"],
            s=200,
            alpha=0.5,
            label=region,
            color=color
        )

    lims = [
        min(df_all["bed_r2"].min(), df_all["free_r2"].min()),
        max(df_all["bed_r2"].max(), df_all["free_r2"].max())
    ]
    plt.plot(lims, lims, "k--", linewidth=1.5, alpha=0.7)

    plt.axvline(x=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.8)
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.8)

    plt.xlabel("In-bed state R²")
    plt.ylabel("In-mobility state R²")

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    ax = plt.gca()
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)

    ax.tick_params(width=2.0, length=6)
    plt.show()