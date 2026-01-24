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
EMBEDDING_DIMS = 64


class SmallBeatsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMS)
        self.color_embedding = nn.Embedding(NUM_COLORS, EMBEDDING_DIMS)

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=N_MFCC,
                out_channels=HIDDEN_DIMS,
                kernel_size=KERNEL_SIZE,
                padding=PADDING,
            ),
            nn.Dropout(p=DROPOUT),
            nn.BatchNorm1d(num_features=HIDDEN_DIMS),
            nn.ReLU(),
        )

        self.adapter = nn.AdaptiveAvgPool1d(output_size=OUTPUT_STEPS // NUM_COLORS)

        self.rnn = nn.GRU(
            input_size=HIDDEN_DIMS + EMBEDDING_DIMS,
            hidden_size=HIDDEN_DIMS,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=False,
            dropout=DROPOUT,
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=HIDDEN_DIMS, out_features=HIDDEN_DIMS),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(in_features=HIDDEN_DIMS, out_features=VOCAB_SIZE),
        )

    def encode_audio(self, audio):
        x = self.encoder(audio)
        x = self.adapter(x)
        x = x.permute(0, 2, 1)
        x = torch.repeat_interleave(x, repeats=NUM_COLORS, dim=1)
        return x

    def forward_rnn(self, audio_features, prev_tokens, color_ids, hidden=None):
        token_embedding = self.token_embedding(prev_tokens)
        color_embedding = self.color_embedding(color_ids)
        full_embedding = token_embedding + color_embedding
        rnn_input = torch.cat([audio_features, full_embedding], dim=-1)
        x, hidden = self.rnn(rnn_input, hidden)
        logits = self.head(x)
        return logits, hidden

    def forward(self, audio, prev_tokens, color_ids, hidden=None):
        audio_features = self.encode_audio(audio)
        logits, hidden = self.forward_rnn(
            audio_features, prev_tokens, color_ids, hidden
        )
        return logits, hidden
