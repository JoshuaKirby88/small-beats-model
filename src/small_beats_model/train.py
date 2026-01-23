import csv
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import torch
import typer
from torch.utils.data import DataLoader, Subset

from small_beats_model.dataset import BeatsDataset
from small_beats_model.loader import RUN_DIR
from small_beats_model.model import SmallBeatsNet
from small_beats_model.preprocessing import VOCAB_SIZE

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 200
TRAIN_VAL_SPLIT = 0.8
LOSS_ROUNDING = 5

app = typer.Typer()
device = torch.device("mps" if torch.mps.is_available() else "cpu")


class Train:
    def __init__(self, overfit_mode=False):
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE
        self.epochs = EPOCHS
        self.train_val_split = TRAIN_VAL_SPLIT
        self.vocab_size = VOCAB_SIZE
        self.loss_rounding = LOSS_ROUNDING
        self.overfit_mode = overfit_mode

        dataset = BeatsDataset()
        if self.overfit_mode:
            self.dataset = cast(BeatsDataset, Subset(dataset, [0] * self.batch_size))
        else:
            self.dataset = dataset

        self.run_dir = RUN_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def get_weights(self):
        if self.overfit_mode:
            return None
        else:
            token_counts = torch.zeros(self.vocab_size)
            for _, targets in self.dataset:
                for token in targets:
                    token_counts[token] += 1
            token_counts = token_counts.clamp(min=1)
            total = token_counts.sum()
            weights = total / (token_counts * self.vocab_size)
            weights = torch.clamp(weights, max=10.0, min=0.1)
            return weights.to(device)

    def load_datasets(self):
        dataset: BeatsDataset = (
            cast(BeatsDataset, Subset(self.dataset, [0] * self.batch_size))
            if self.overfit_mode
            else self.dataset
        )
        shuffle = not self.overfit_mode

        train_size = int(self.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        train_loader = DataLoader(
            train_dataset,
            shuffle=shuffle,
            num_workers=0,
            batch_size=self.batch_size,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=shuffle,
            num_workers=0,
            batch_size=self.batch_size,
        )

        return train_loader, val_loader

    def plot(
        self, output_path: Path, train_losses: list[float], val_losses: list[float]
    ):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Training VS Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig(output_path)

        with open(self.run_dir / "training_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            for epoch, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
                writer.writerow(
                    [
                        epoch,
                        round(t_loss, self.loss_rounding),
                        round(v_loss, self.loss_rounding),
                    ]
                )

    def train(self):
        weights = self.get_weights()
        train_loader, val_loader = self.load_datasets()

        model = SmallBeatsNet()
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            model.train()

            total_train_loss = 0
            for audio, targets in train_loader:
                audio: torch.Tensor = audio.to(device)
                targets: torch.Tensor = targets.to(device)
                optimizer.zero_grad()
                predictions = model(audio)
                train_loss: torch.Tensor = criterion(
                    predictions.permute(0, 2, 1), targets
                )
                train_loss.backward()
                optimizer.step()
                total_train_loss += train_loss.item()

            model.eval()

            total_val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for audio, targets in val_loader:
                    audio, targets = audio.to(device), targets.to(device)
                    predictions = model(audio)
                    val_loss: torch.Tensor = criterion(
                        predictions.permute(0, 2, 1), targets
                    )
                    total_val_loss += val_loss.item()
                    correct += (predictions.argmax(dim=2) == targets).sum().item()
                    total += targets.numel()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)

            print(
                f"Epoch: {epoch} | Train loss: {round(avg_train_loss, self.loss_rounding)} | Val loss: {round(avg_val_loss, self.loss_rounding)}"
            )

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.run_dir / "best_model.pth")
                print(
                    f"New best model with val loss {round(avg_val_loss, self.loss_rounding)} saved"
                )

        self.plot(
            output_path=self.run_dir / "loss_plot.png",
            train_losses=train_losses,
            val_losses=val_losses,
        )


@app.command()
def main(overfit: bool = False):
    trainer = Train(overfit_mode=overfit)
    trainer.train()


if __name__ == "__main__":
    app()
