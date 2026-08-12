import torch
import torch.nn as nn
import torch.nn.functional as F

# Deep Learning Models for Population-level analysis

## Mel-spectrogram decoding

### Embedding Module
class Conv1dChannelEmbedding(nn.Module):
    """
    ][B, C, T] -> [B, T, D]
    Conv → BN → GELU → Conv → BN → GELU
    """
    def __init__(self, input_channels, embed_dim=128, kernel_time=3, padding_time=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=input_channels, 
            out_channels=embed_dim,
            kernel_size=kernel_time, 
            padding=padding_time
        )
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_time,
            padding=padding_time
        )
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = x.transpose(1, 2)
        return x

### Attention Pooling

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.Tanh(),
            nn.Linear(dim//2, 1)
        )

    def forward(self, x):  # x [B, T, D]
        w = self.att(x)                  # [B, T, 1]
        att = torch.softmax(w, dim=1)
        return (att * x).sum(dim=1)      # [B, D]

### Main Model Structure

class SEEGSharedEncoder_GRU_MultiHeadClassifier(nn.Module):
    def __init__(
        self,
        input_channels,
        embed_dim=128,
        rnn_hidden=256,
        rnn_layers=2,
        dropout=0.5,
        n_features=40,
        n_classes=9,
        pooling="att"
    ):
        super().__init__()

        # conv embedding
        self.embedding = Conv1dChannelEmbedding(input_channels, embed_dim)

        # GRU encoder
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        enc_out_dim = rnn_hidden * 2  # bidirectional

        # pooling
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = AttentionPooling(enc_out_dim)

        # multi-head classifiers
        self.shared_fc = nn.Sequential(
            nn.Linear(enc_out_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Linear(128, n_classes) for _ in range(n_features)
        ])

        self.post_rnn_norm = nn.LayerNorm(rnn_hidden * 2)
        self.embed_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_norm(x)
        x = F.dropout(x, p=0.5, training=self.training)

        out, _ = self.rnn(x)
        out = self.post_rnn_norm(out)

        out = F.dropout(out, p=0.5, training=self.training)

        out = self.pool(out)
        out = self.shared_fc(out)

        outs = [h(out).unsqueeze(1) for h in self.heads]
        return torch.cat(outs, dim=1)
