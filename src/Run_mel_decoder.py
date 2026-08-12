#### Run to train Mel-spectrogram decoding model SEEGSharedEncoder_GRU_MultiHeadClassifier in models.py
import numpy as np
import os
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import GroupKFold


def compute_ce_loss_multihead(logits, targets):
    """
    logits: [B, F, C]
    targets: [B, F]
    """
    num_heads = logits.size(1)
    loss_list = []
    for f in range(num_heads):
        loss_f = F.cross_entropy(logits[:, f, :], targets[:, f])
        loss_list.append(loss_f)

    return sum(loss_list) / len(loss_list), [l.item() for l in loss_list]

def compute_accuracy_multihead(logits, targets):
    """
    logits: [B, F, C]
    targets: [B, F]
    return: mean_acc, per_head_acc_list
    """
    num_heads = logits.size(1)
    acc_list = []

    for f in range(num_heads):
        pred_f = logits[:, f, :].argmax(dim=-1)
        true_f = targets[:, f]
        acc_f = (pred_f == true_f).float().mean().item()
        acc_list.append(acc_f)

    return sum(acc_list) / len(acc_list), acc_list


def compute_ce_loss_auto(logits, targets):
    if logits.ndim == 3:       # multi-head
        return compute_ce_loss_multihead(logits, targets)
    else:                      # single head
        loss = F.cross_entropy(logits, targets)
        return loss, [loss.item()]


def train_seq2vec_model_cv(
    dataset,
    trial_ids,
    model_config,
    num_epochs=100,
    lr=1e-4,
    warmup_epochs=4,
    batch_size=64,
    patience=10,
    n_splits=5,
    model_save_dir="./checkpoints",
    max_grad_norm=1.0,
    weight_decay=1e-4,
    min_delta=1e-4
):

    os.makedirs(model_save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    sample_x, sample_y = dataset[0]
    input_channels = sample_x.shape[0]

    # K-Fold
    gkf = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(dataset))

    best_models_state = []
    best_models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=np.array(trial_ids))):

        print(f"\n====================== Fold {fold+1}/{n_splits} ======================")

        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=batch_size, shuffle=False
        )

        model = model_config["class"](
            input_channels=input_channels,
            **model_config["kwargs"]
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )

        best_val_loss = float("inf")
        best_state = None
        es_counter = 0

        # Epoch
        for epoch in range(1, num_epochs + 1):
            # Warmup（0.5lr → lr）
            if epoch <= warmup_epochs:
                warm_factor = 0.5 + 0.5 * epoch / warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * warm_factor

            # Train
            model.train()
            train_loss = 0

            train_head_losses = []
            all_train_acc = []

            for X_cpu, y_cpu in train_loader:
                X = X_cpu.to(device).float()
                y = y_cpu.to(device).long()

                optimizer.zero_grad(set_to_none=True)
                logits = model(X)

                loss, head_losses = compute_ce_loss_auto(logits, y)
                mean_acc, per_head_acc = compute_accuracy_multihead(logits, y)
                all_train_acc.append(per_head_acc)
                loss.backward()

                optimizer.step()

                train_loss += loss.item()
                train_head_losses.append(head_losses)

            train_loss /= len(train_loader)
            train_head_losses = np.mean(train_head_losses, axis=0)
            train_head_acc = np.mean(all_train_acc, axis=0)
            train_mean_acc = np.mean(train_head_acc) 

            # Validation
            model.eval()
            val_loss = 0
            val_head_losses = []
            all_val_acc = []

            with torch.no_grad():
                for X_cpu, y_cpu in val_loader:
                    X = X_cpu.to(device).float()
                    y = y_cpu.to(device).long()

                    logits = model(X)
                    loss, head_losses = compute_ce_loss_auto(logits, y)
                    mean_acc, per_head_acc = compute_accuracy_multihead(logits, y)
                    all_val_acc.append(per_head_acc)

                    val_loss += loss.item()
                    val_head_losses.append(head_losses)

            val_loss /= len(val_loader)
            val_head_losses = np.mean(val_head_losses, axis=0)
            val_head_acc = np.mean(all_val_acc, axis=0)
            val_mean_acc = np.mean(val_head_acc)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            head_info_train = " | ".join([f"H{idx}:{v:.3f}" for idx, v in enumerate(train_head_losses)])
            head_info_val =   " | ".join([f"H{idx}:{v:.3f}" for idx, v in enumerate(val_head_losses)])

            acc_info_train = " | ".join([f"H{idx}:{v:.3f}" for idx, v in enumerate(train_head_acc)])
            acc_info_val   = " | ".join([f"H{idx}:{v:.3f}" for idx, v in enumerate(val_head_acc)])
            
            print(
                f"Fold {fold+1} | Epoch {epoch:03d} | LR: {current_lr:.6f} | "
                f"TrainLoss: {train_loss:.3f} ({head_info_train}) | TrainACC: {train_mean_acc:.3f} ({acc_info_train}) | "
                f"ValLoss: {val_loss:.3f} ({head_info_val}) | ValACC: {val_mean_acc:.3f} ({acc_info_val})"
            )

            # Early Stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                es_counter = 0
            else:
                es_counter += 1

            if es_counter >= patience:
                print(f"Early Stop at Epoch {epoch}")
                break

        save_path = f"{model_save_dir}/best_fold_{fold+1}.pth"
        torch.save(best_state, save_path)
        print(f"[Saved] {save_path}")

        best_models_state.append(best_state)
        best_model = copy.deepcopy(model)
        best_model.load_state_dict(best_state)
        best_models.append(best_model)

    print("\nTraining Finished.")
    return best_models_state, best_models