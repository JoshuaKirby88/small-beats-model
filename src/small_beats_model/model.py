import torch
import torch.nn as nn

from small_beats_model.dataset import WINDOW_BEATS
from small_beats_model.preprocessing import N_MFCC, STEPS_PER_BEAT
from small_beats_model.vocab import Vocab

HIDDEN_DIMS = 512
KERNEL_SIZE = 3
PADDING = (KERNEL_SIZE - 1) // 2
OUTPUT_STEPS = WINDOW_BEATS * STEPS_PER_BEAT
DROPOUT = 0.2
NUM_LAYERS = 3
EMBEDDING_DIMS = 256


class SmallBeatsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab = Vocab()
        self.token_embedding = nn.Embedding(self.vocab.vocab_size, EMBEDDING_DIMS)

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

        # CNN Output:
        # (Batch, 512, Sequence)

        self.adapter = nn.AdaptiveAvgPool1d(output_size=OUTPUT_STEPS)

        # Adapter Output:
        # (Batch, 512, 128)
        # Permute:
        # (Batch 128, 512)
        # Concat audio (512 dims) + token embedding (256 dims)
        # (Batch, 128, 768)

        self.rnn = nn.GRU(
            input_size=HIDDEN_DIMS + EMBEDDING_DIMS,
            hidden_size=HIDDEN_DIMS,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=False,
            dropout=DROPOUT,
        )

        # RNN Output:
        # (Batch, 128, 512)

        self.audio_layer_norm = nn.LayerNorm(HIDDEN_DIMS)
        self.rnn_layer_norm = nn.LayerNorm(HIDDEN_DIMS)

        # Normalized Addition Output:
        # (Batch, 128, 512)

        self.head = nn.Sequential(
            nn.Linear(in_features=HIDDEN_DIMS, out_features=HIDDEN_DIMS),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(in_features=HIDDEN_DIMS, out_features=self.vocab.vocab_size),
        )

    def encode_audio(self, audio):
        x = self.encoder(audio)
        x = self.adapter(x)
        x = x.permute(0, 2, 1)
        return x

    def forward_rnn(self, audio_features, prev_tokens, hidden=None):
        token_embedding = self.token_embedding(prev_tokens)
        rnn_input = torch.cat([audio_features, token_embedding], dim=-1)
        x, hidden = self.rnn(rnn_input, hidden)
        head_input = self.audio_layer_norm(audio_features) + self.rnn_layer_norm(x)
        logits = self.head(head_input)
        return logits, hidden

    def forward(self, audio, prev_tokens, hidden=None):
        audio_features = self.encode_audio(audio)
        logits, hidden = self.forward_rnn(audio_features, prev_tokens, hidden)
        return logits, hidden
