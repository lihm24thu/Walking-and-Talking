#### Run to train Speech Detection Model CNN_BiLSTM_Binary_MMD_SE in models.py
import numpy as np
from tqdm import tqdm
import gc
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score

# Dataset

class Stage1Dataset(Dataset):
    def __init__(self, X, y, d, oversample=True, binary=True, pos_dup_factor=3):
        if binary:
            y = (y > 0).astype(np.int64)

        self.X, self.y, self.d = X, y, d

        if oversample:
            pos_idx = np.where(self.y == 1)[0]
            neg_idx = np.where(self.y == 0)[0]

            if len(pos_idx) and len(neg_idx):
                # Augmentation
                dup = np.random.choice(pos_idx, size=pos_dup_factor * len(pos_idx), replace=True)
                idx = np.concatenate([neg_idx, pos_idx, dup])
                
                # shuffle
                np.random.shuffle(idx)

                self.X = self.X[idx]
                self.y = self.y[idx]
                self.d = self.d[idx]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(self.d[idx], dtype=torch.long)
        )


# Loss for Options
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # assume inputs are logits
        BCE_loss = F.binary_cross_entropy(inputs, targets.float(), reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class MMDLoss(nn.Module):
    def __init__(self, kernel='rbf'):
        super(MMDLoss, self).__init__()
        self.kernel = kernel

    def gaussian_kernel(self, x, y, sigma=1.0):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1).expand(x_size, y_size, dim)
        y = y.unsqueeze(0).expand(x_size, y_size, dim)
        return torch.exp(-((x - y) ** 2).mean(2) / (2 * sigma ** 2))

    def forward(self, x1, x2):
        Kxx = self.gaussian_kernel(x1, x1)
        Kyy = self.gaussian_kernel(x2, x2)
        Kxy = self.gaussian_kernel(x1, x2)
        return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

class CORALLoss(nn.Module):
    def __init__(self):
        super(CORALLoss, self).__init__()

    def forward(self, source, target):
        d = source.size(1)
        source_covar = self._covariance(source)
        target_covar = self._covariance(target)
        loss = torch.mean((source_covar - target_covar) ** 2)
        return loss / (4 * d * d)

    def _covariance(self, x):
        n = x.size(0)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        xc = x - mean_x
        cov = (xc.t() @ xc) / (n - 1)
        return cov

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, embeddings, labels):
        anchors, positives, negatives = self._select_triplets(embeddings, labels)
        if anchors is None:
            return torch.tensor(0.0, device=embeddings.device)
        return self.loss_fn(anchors, positives, negatives)

    def _select_triplets(self, embeddings, labels):
        anchors, positives, negatives = [], [], []
        for i in range(len(embeddings)):
            a, label_a = embeddings[i], labels[i]
            pos_indices = (labels == label_a).nonzero(as_tuple=True)[0]
            neg_indices = (labels != label_a).nonzero(as_tuple=True)[0]
            if len(pos_indices) > 1 and len(neg_indices) > 0:
                p = embeddings[pos_indices[0 if pos_indices[0] != i else 1]]
                n = embeddings[neg_indices[0]]
                anchors.append(a)
                positives.append(p)
                negatives.append(n)
        if len(anchors) == 0:
            return None, None, None
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


# Training

def print_class_distribution(X, y, name=""):
    y = np.asarray(y)
    total = len(y)
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    print(f"📊 Dataset: {name} | Total: {total} | Pos: {pos} ({pos/total:.2%}) | Neg: {neg} ({neg/total:.2%})")


def train_binary_model(model, train_loader, val_loader,
                       epochs=30, lr=1e-4, device="cuda",
                       class_weights=None, use_mmd=True, mmd_lambda=0.1,
                       mmd_detach=False, loss_fn="focal",
                       patience=10, min_delta=0.001,
                       coral_lambda=0.1, triplet_lambda=0.05):

    model.to(device)

    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights).float().to(device)
        pos_weight = weight_tensor[1] / weight_tensor[0]
    else:
        pos_weight = None

    if loss_fn == "focal":
        criterion = FocalLoss(alpha=1.5, gamma=2.5)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mmd_loss_fn = MMDLoss() if use_mmd else None
    coral_loss_fn = CORALLoss() if coral_lambda > 0 else None
    triplet_loss_fn = TripletLoss(margin=1.0) if triplet_lambda > 0 else None

    best_model = None
    best_f1 = 0
    counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tloss = 0
        for x, y, d in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y, d = x.to(device), y.float().to(device), d.to(device)
            optimizer.zero_grad()

            logits, features = model(x, return_feature=True)
            logits = logits.squeeze(1)
            loss = criterion(logits, y)

            # ✅ CORAL
            if coral_lambda > 0:
                s, t = features[d == 0], features[d == 1]
                if len(s) > 1 and len(t) > 1:
                    loss += coral_lambda * coral_loss_fn(s, t)

            # ✅ MMD
            if use_mmd:
                s, t = features[d == 0], features[d == 1]
                if s.ndim == 2 and t.ndim == 2 and len(s) > 1 and len(t) > 1:
                    s_ = s.detach() if mmd_detach else s
                    t_ = t.detach() if mmd_detach else t
                    loss += mmd_lambda * mmd_loss_fn(s_, t_)

            # ✅ Triplet Loss
            if triplet_lambda > 0:
                triplet = triplet_loss_fn(features, y.long())
                loss += triplet_lambda * triplet

            loss.backward()
            optimizer.step()
            tloss += loss.item()

        # Validation
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.float().to(device)
                logits = model(x).squeeze(1)
                val_logits.append(logits.cpu())
                val_labels.append(y.cpu())

        val_logits = torch.cat(val_logits)
        val_labels = torch.cat(val_labels)
        probs = torch.sigmoid(val_logits)
        preds = (probs > 0.5).long()

        auc = roc_auc_score(val_labels, probs)
        f1 = f1_score(val_labels, preds)
        precision = precision_score(val_labels, preds, zero_division=0)
        acc = accuracy_score(val_labels, preds)

        print(f"Epoch {epoch}: loss={tloss/len(train_loader):.4f}, f1={f1:.4f}, precision={precision:.4f}, acc={acc:.4f}, auc={auc:.4f}")

        if f1 > best_f1 + min_delta:
            best_f1 = f1
            best_model = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("🛑 Early stopping triggered: no F1 improvement")
                break

    if best_model is not None:
        model.load_state_dict(best_model)
    return model


def box_re_ranking_loop(model, X_all, y_all, d_all, 
                        train_idx, val_idx,
                        batch_size=16, epochs=10, lr=1e-4,
                        mmd_lambda=0.1, coral_lambda=0.1, triplet_lambda=0.05,
                        class_weights=[1.0, 1.2],
                        freeze_backbone=False,
                        device="cuda",
                        pos_dup_factor=7):

    torch.cuda.empty_cache()
    gc.collect()

    def eval_auc_batched(model, X_val, y_val, batch_size=16, device='cuda'):
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                x = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32).to(device)
                logits = model(x)  # [B, 1]
                prob = torch.sigmoid(logits).squeeze(1)
                probs.append(prob.cpu())
        probs = torch.cat(probs).numpy()
        return roc_auc_score(y_val, probs)

    def freeze_backbone_params(m):
        for n, p in m.named_parameters():
            if 'fc' not in n:
                p.requires_grad = False

    # 1. Split
    y_bin = (y_all != 0).astype(np.int64)
    X_tr, y_tr, d_tr = X_all[train_idx], y_bin[train_idx], d_all[train_idx]
    X_va, y_va, d_va = X_all[val_idx], y_bin[val_idx], d_all[val_idx]

    # 2. Validation
    dl_va = DataLoader(Stage1Dataset(X_va, y_va, d_va, oversample=False),
                       batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Freeze backbone
    model = model.to(device)
    if freeze_backbone:
        freeze_backbone_params(model)

    # 4. Primary Training
    print_class_distribution(X_tr, y_tr, name="Primary Train Set (Before Sampling)")
    dataset = Stage1Dataset(X_tr, y_tr, d_tr, oversample=True, pos_dup_factor=pos_dup_factor)
    print_class_distribution(dataset.X, dataset.y, name="Primary Train Set (After Sampling)")

    dl_tr = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    x_batch, y_batch, d_batch = next(iter(dl_tr))
    print("Batch x shape:", x_batch.shape)

    model = train_binary_model(
        model, dl_tr, dl_va,
        epochs=epochs, lr=lr, device=device,
        class_weights=class_weights,
        use_mmd=True, mmd_lambda=mmd_lambda,
        coral_lambda=coral_lambda,
        triplet_lambda=triplet_lambda
    )

    val_auc = eval_auc_batched(model, X_va, y_va)
    print(f"\n✅ Final validation AUC: {val_auc:.4f}")
    return model


