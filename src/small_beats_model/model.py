import torch
import torch.nn as nn

from small_beats_model.dataset import WINDOW_BEATS
from small_beats_model.preprocessing import N_MFCC, STEPS_PER_BEAT, VOCAB_SIZE

HIDDEN_DIMS = 256
KERNEL_SIZE = 3
PADDING = 1
OUTPUT_STEPS = WINDOW_BEATS * STEPS_PER_BEAT
NUM_LAYERS = 2


class SmallBeatsNet(nn.Module):
    def __init__(
        self,
        n_mfcc=N_MFCC,
        hidden_dims=HIDDEN_DIMS,
        kernel_size=KERNEL_SIZE,
        padding=PADDING,
        output_steps=OUTPUT_STEPS,
        num_layers=NUM_LAYERS,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=n_mfcc,
                out_channels=hidden_dims,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(num_features=hidden_dims),
            nn.ReLU(),
        )

        self.adapter = nn.AdaptiveAvgPool1d(output_size=output_steps)

        self.rnn = nn.GRU(
            input_size=hidden_dims,
            hidden_size=hidden_dims,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Linear(in_features=hidden_dims * 2, out_features=VOCAB_SIZE)

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
