#### Corrsponding to Fig. 4
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, recall_score, roc_auc_score

# Fig. 4b Each point represents one participant, with lines connecting paired measurements from the same participant. The deep learning (DL) model and SVM baseline were evaluated separately during bedbound and freely moving states. The DL model achieved consistently higher AUC values than the SVM baseline in both bedbound (DL: 0.952 ± 0.009; SVM: 0.586 ± 0.079) and freely moving conditions (DL: 0.954 ± 0.011; SVM: 0.576 ± 0.103). Statistical comparisons between models were performed using paired Wilcoxon signed-rank tests across participants. Significance levels are indicated by asterisks (***p < 0.001, **p < 0.01, *p < 0.05).
## Calculate metrics from confusion matrix

def calculate_metrics_from_confusion_matrix(cm_dict):

    """
    cm_fict: confusion matrix of speech detection.
    """
    results = {}
    
    for patient_id, cm in cm_dict.items():
        if isinstance(cm, str) and cm.startswith('b'):
            cm_value = eval(cm.split('=')[1].strip())
        else:
            cm_value = cm
            
        tn, fp, fn, tp = cm_value[0][0], cm_value[0][1], cm_value[1][0], cm_value[1][1]
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        auc_approx = 1 - (fpr + (1 - tpr)) / 2
        
        results[patient_id] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_approx,
            'confusion_matrix': cm_value
        }
    
    return results

## Statistic
def comparison_between_dl_svm(patient_confusion_matrices, patient_confusion_matrices_m, baseline_results, baseline_results_m):
    # Calculate metrics
    dl_bed = calculate_metrics_from_confusion_matrix(
        patient_confusion_matrices,
        'original'
    )

    dl_move = calculate_metrics_from_confusion_matrix(
        patient_confusion_matrices_m,
        'original'
    )

    svm_bed = baseline_results
    svm_move = baseline_results_m

    patients = list(patient_confusion_matrices.keys())

    # Create dataframe
    rows = []

    for p in patients:

        rows.append(
            {
                "Patient": p,
                "Model": "DL",
                "State": "Bed",
                "AUC": dl_bed[p]["auc"],
                "F1": dl_bed[p]["f1"]
            }
        )

        rows.append(
            {
                "Patient": p,
                "Model": "DL",
                "State": "Mobility",
                "AUC": dl_move[p]["auc"],
                "F1": dl_move[p]["f1"]
            }
        )

        rows.append(
            {
                "Patient": p,
                "Model": "SVM",
                "State": "Bed",
                "AUC": svm_bed[p]["auc"],
                "F1": svm_bed[p]["f1"]
            }
        )

        rows.append(
            {
                "Patient": p,
                "Model": "SVM",
                "State": "Mobility",
                "AUC": svm_move[p]["auc"],
                "F1": svm_move[p]["f1"]
            }
        )


    df = pd.DataFrame(rows)

    return df

## Visualization function
def add_significance_bar(
    ax,
    x1,
    x2,
    y,
    h,
    p_value,
    fontsize=25
):
    """
    Draw significance bracket and stars
    """
    # bracket
    ax.plot(
        [x1, x1, x2, x2],
        [y, y+h, y+h, y],
        lw=2.5,
        c="black"
    )

    # significance label
    if p_value < 0.001:
        text = "***"
    elif p_value < 0.01:
        text = "**"
    elif p_value < 0.05:
        text = "*"
    else:
        text = "ns"

    ax.text(
        (x1+x2)/2,
        y+h,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize
    )

## Plot figures
def plot_state_comparison(metric, patients, df):

    fig, ax = plt.subplots(
        figsize=(7,10)
    )

    positions = {
        ("DL","Bed"):0,
        ("DL","Mobility"):1,
        ("SVM","Bed"):3,
        ("SVM","Mobility"):4
    }

    colors = {
        "DL":"#4C72B0",
        "SVM":"#DD8452"
    }

    # individual patient lines
    for model in ["DL","SVM"]:

        for p in patients:

            bed = df[
                (df.Patient==p)
                &
                (df.Model==model)
                &
                (df.State=="Bed")
            ][metric].values[0]


            move = df[
                (df.Patient==p)
                &
                (df.Model==model)
                &
                (df.State=="Mobility")
            ][metric].values[0]


            ax.plot(
                [
                    positions[(model,"Bed")],
                    positions[(model,"Mobility")]
                ],
                [
                    bed,
                    move
                ],
                color=colors[model],
                alpha=0.35,
                linewidth=2
            )

    # scatter points
    for model in ["DL","SVM"]:

        for state in ["Bed","Mobility"]:

            values = df[
                (df.Model==model)
                &
                (df.State==state)
            ][metric]

            ax.scatter(
                np.ones(len(values))
                *
                positions[(model,state)],
                values,
                s=150,
                color=colors[model],
                edgecolor="black",
                linewidth=2,
                zorder=3,
                label=f"{model}-{state}"
            )

    ax.set_xticks(
        list(positions.values())
    )

    ax.set_xticklabels(
        [
            "DL\nBed",
            "DL\nMobility",
            "SVM\nBed",
            "SVM\nMobility"
        ]
    )
    ax.set_ylabel(
        metric
    )
    ax.tick_params(
        axis='y'
    )


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)

    # Wilcoxon test: DL vs SVM
    sig_positions = {
        "Bed": {
            "x1": 0,
            "x2": 3,
            "y": 1.00
        },
        "Mobility": {
            "x1": 1,
            "x2": 4,
            "y": 1.045
        }
    }
    
    for state in ["Bed", "Mobility"]:
        dl_values = df[
            (df.Model=="DL")
            &
            (df.State==state)
        ][metric].values
    
        svm_values = df[
            (df.Model=="SVM")
            &
            (df.State==state)
        ][metric].values
    
        _, p = stats.wilcoxon(
            dl_values,
            svm_values
        )
    
        add_significance_bar(
            ax,
            x1=sig_positions[state]["x1"],
            x2=sig_positions[state]["x2"],
            y=sig_positions[state]["y"],
            h=0.012,
            p_value=p,
            fontsize=28
        )
    
    ax.set_ylim(0,1.08)

    ax.legend().set_visible(False)

    ax.axhline(
        y=0.5,
        color="#999999",
        linestyle="--",
        linewidth=2,
        alpha=0.85,
    )
    
    plt.tight_layout()
    plt.show()


# Fig. 4c Speech detection performance across participants in both in-bed and in-mobility conditions, demonstrating robust decoding across behavioral contexts. 
## Compute group metrics for all patients
def compute_group_metrics(patients_dict):
    """
    patients_dict[pid]['matrix'] = [[tn, fp],
                                    [fn, tp]]
    """
    metrics = {
        'AUC': [],
        'F1_Speaking': [],
        'F1_NonSpeaking': [],
        'Recall_Speaking': [],
        'Recall_NonSpeaking': [],
        'Precision_Speaking': [],
        'Specificity': [],
        'Sensitivity': []
    }

    for pid, info in patients_dict.items():
        cm = info['matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

        y_true = np.array([0]*tn + [1]*fn + [0]*fp + [1]*tp)
        y_pred = np.array([0]*tn + [0]*fn + [1]*fp + [1]*tp)

        metrics['AUC'].append(roc_auc_score(y_true, y_pred))
        metrics['F1_Speaking'].append(f1_score(y_true, y_pred, pos_label=1))
        metrics['F1_NonSpeaking'].append(f1_score(y_true, y_pred, pos_label=0))
        metrics['Recall_Speaking'].append(recall_score(y_true, y_pred, pos_label=1))
        metrics['Specificity'].append(recall_score(y_true, y_pred, pos_label=0))
        metrics['Precision_Speaking'].append(tp / (tp + fp))
        metrics['Specificity'].append(tn / (tn + fp))
        metrics['Sensitivity'].append(tp / (tp + fn))

    return metrics

## Visualization
def plot_dual_state_group_radar(
    patients_state_A,
    patients_state_B,
    label_A="State A",
    label_B="State B",
    color_A="tab:blue",
    color_B="tab:red"
):
    metrics_A = compute_group_metrics(patients_state_A)
    metrics_B = compute_group_metrics(patients_state_B)

    metric_names = list(metrics_A.keys())
    n_metrics = len(metric_names)

    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(
        figsize=(13, 13),
        subplot_kw=dict(polar=True)
    )

    def plot_state(metrics, color, label):
        means = [np.mean(metrics[m]) for m in metric_names]
        mins  = [np.min(metrics[m])  for m in metric_names]
        maxs  = [np.max(metrics[m])  for m in metric_names]

        means += means[:1]
        mins  += mins[:1]
        maxs  += maxs[:1]

        ax.fill_between(
            angles,
            mins,
            maxs,
            color=color,
            alpha=0.25
        )

        ax.plot(
            angles,
            means,
            color=color,
            linewidth=3.5,
            label=label
        )

        ax.scatter(
            angles[:-1],
            means[:-1],
            color=color,
            s=50,
            zorder=3
        )

    plot_state(metrics_A, color_A, label_A)
    plot_state(metrics_B, color_B, label_B)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_ylim(0.8, 1.0)
    y_ticks = np.arange(0.8, 1.01, 0.05) 
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(
        [f"{tick:.2f}" for tick in y_ticks],
        fontsize=28,
        color='black'
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.25, 1.15),
        frameon=False
    )

    plt.tight_layout()
    plt.show()


# Fig. 4d,j Spearman rank correlation
def spearman_between(df1, df2):
    """
    df1 and df2: saliency or decoding performance data frame.
    """
    merged = pd.merge(
        df1,
        df2,
        on="channel",
        how="inner",
        suffixes=("_1", "_2")
    )

    if len(merged) == 0:
        return np.nan, np.nan

    rho, p = spearmanr(
        merged["score_1"],
        merged["score_2"]
    )

    return rho, p


# Fig. 4i Distribution of decoding accuracy (ACC) and F1 scores across 40 Mel-frequency bands for each participant in in-bed and in-mobility conditions. 
## Visualization functions
def load_metrics(data_root, patient_id, state_fname):
    """
    Read the CSV file corresponding to a specific state for a designated patient, and return a dictionary: {'accuracy': ndarray(40,), 'f1_score': ndarray(40,)}.
    """
    pdir = os.path.join(data_root, patient_id)
    path = os.path.join(pdir, state_fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)

    acc = df['accuracy'].values.astype(float)
    f1 = df['f1_score'].values.astype(float)
    return {'accuracy': acc, 'f1_score': f1}

## Visualization
def plot_metric_comparison(
    metric,
    patient_ids,
    bed_fname,
    mobility_fname,
    ymax=1,
    figsize=(14, 6),
    savepath=None
):
    """
    Plot one metric (Accuracy or F1) comparing
    In bed vs Mobility for each patient.
    """

    rows = []

    for pid in patient_ids:
        # -------- In bed --------
        d = load_metrics(pid, bed_fname)

        if metric == "Accuracy":
            values = d["accuracy"]
        elif metric == "F1":
            values = d["f1_score"]
        else:
            raise ValueError(metric)

        rows.append(pd.DataFrame({
            "patient": pid,
            "condition": "In bed",
            "value": values
        }))

        # -------- Mobility --------
        d = load_metrics(pid, mobility_fname)

        if metric == "Accuracy":
            values = d["accuracy"]
        else:
            values = d["f1_score"]

        rows.append(pd.DataFrame({
            "patient": pid,
            "condition": "Mobility",
            "value": values
        }))

    df = pd.concat(rows, ignore_index=True)

    # Overall paired test (patient-level)
    patient_mean = (
        df.groupby(["patient", "condition"])["value"]
          .mean()
          .unstack()
    )
    
    stat_all, p_all = wilcoxon(
        patient_mean["In bed"],
        patient_mean["Mobility"]
    )
    
    print("\n" + "="*60)
    print(f"Overall comparison ({metric})")
    print("="*60)
    print(f"Wilcoxon signed-rank test")
    print(f"Statistic = {stat_all:.3f}")
    print(f"p = {p_all:.6g}")
    
    print()
    print(
        f"In bed : "
        f"{patient_mean['In bed'].mean():.4f} ± "
        f"{patient_mean['In bed'].std():.4f}"
    )
    
    print(
        f"Mobility: "
        f"{patient_mean['Mobility'].mean():.4f} ± "
        f"{patient_mean['Mobility'].std():.4f}"
    )

    # -------------------------------------------------------
    # Plot
    # -------------------------------------------------------
    plt.figure(figsize=figsize)

    boxplot_params = {
        "linewidth": 2,
        "flierprops": dict(
            marker='o',
            markerfacecolor='none',
            markersize=4,
            markeredgewidth=2
        ),
        "whiskerprops": dict(linewidth=2),
        "capprops": dict(linewidth=2),
        "medianprops": dict(linewidth=2, color='black')
    }

    ax = sns.boxplot(
        data=df,
        x="patient",
        y="value",
        hue="condition",
        dodge=True,
        width=0.6,
        showfliers=False,
        palette=["#DD8452", "#4C72B0"],
        **boxplot_params
    )

    # chance level
    ax.axhline(chance_acc, ls="--", lw=2.5, color="gray")

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_color('black')
    ax.spines["bottom"].set_color('black')
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)

    ax.tick_params(
        axis="both", 
        which="major", 
        labelsize=28,
        colors='black',
        width=2.0, 
        length=8 
    )

    ax.set_ylim(0, ymax)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.grid(False)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.show()
    
    return p_values
