import torch
import torch.nn as nn

from small_beats_model.dataset import WINDOW_BEATS
from small_beats_model.preprocessing import (
    N_MFCC,
    NUM_COLORS,
    STEPS_PER_BEAT,
    VOCAB_SIZE,
)

HIDDEN_DIMS = 512
KERNEL_SIZE = 3
PADDING = (KERNEL_SIZE - 1) // 2
OUTPUT_STEPS = WINDOW_BEATS * STEPS_PER_BEAT * NUM_COLORS
NUM_LAYERS = 3
DROPOUT = 0.1


class SmallBeatsNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_dims = HIDDEN_DIMS
        self.kernel_size = KERNEL_SIZE
        self.padding = PADDING
        self.output_steps = OUTPUT_STEPS
        self.num_layers = NUM_LAYERS
        self.n_mfcc = N_MFCC
        self.dropout = DROPOUT
        self.vocab_size = VOCAB_SIZE

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.n_mfcc,
                out_channels=self.hidden_dims,
                kernel_size=self.kernel_size,
                padding=self.padding,
            ),
            nn.Dropout(p=self.dropout),
            nn.BatchNorm1d(num_features=self.hidden_dims),
            nn.ReLU(),
        )

        self.adapter = nn.AdaptiveAvgPool1d(output_size=self.output_steps)

        self.rnn = nn.GRU(
            input_size=self.hidden_dims,
            hidden_size=self.hidden_dims,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=self.hidden_dims * 2, out_features=self.hidden_dims),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.hidden_dims, out_features=self.vocab_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.adapter(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        logits = self.head(x)
        return logits


if __name__ == "__main__":
    model = SmallBeatsNet()
    input = torch.randn(1, 40, 689)
    output = model(input)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
