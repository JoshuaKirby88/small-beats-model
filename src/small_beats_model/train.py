import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from torch.utils.data import DataLoader

from small_beats_model.dataset import BeatsDataset
from small_beats_model.loader import RUN_DIR
from small_beats_model.model import SmallBeatsNet
from small_beats_model.utils import device_type
from small_beats_model.vocab import EMPTY_TOKEN, Vocab

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 30
TRAIN_SPLIT = 0.8
WEIGHT_DECAY = 1e-3
SCHEDULE_FACTOR = 0.5
SCHEDULE_PATIENCE = 3
CLASS_WEIGHT_CLAMP_MAX = 1
CLASS_WEIGHT_CLAMP_MIN = 1
EMPTY_TOKEN_WEIGHT = 0.1

LOSS_LOG_ROUNDING = 2
NUM_WORKERS = 4

app = typer.Typer()


class Train:
    def __init__(self, overfit_mode=False):
        self.overfit_mode = overfit_mode
        self.device = torch.device(device_type)
        self.vocab = Vocab()

        self.run_dir = RUN_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def get_class_weights(self):
        token_counts = torch.zeros(self.vocab.vocab_size)
        for _, _, targets in BeatsDataset():
            for token in targets:
                token_counts[token] += 1
        token_counts = token_counts.clamp(min=1)
        total = token_counts.sum()
        weights = total / (token_counts * self.vocab.vocab_size)
        weights = torch.clamp(
            weights, max=CLASS_WEIGHT_CLAMP_MAX, min=CLASS_WEIGHT_CLAMP_MIN
        )
        weights[EMPTY_TOKEN] = EMPTY_TOKEN_WEIGHT
        return weights.to(self.device)

    def load_datasets(self):
        train_map_ids, val_map_ids = BeatsDataset.split(
            train_split=TRAIN_SPLIT, shuffle=not self.overfit_mode, seed=42
        )
        train_dataset = BeatsDataset(allowed_map_ids=train_map_ids)
        val_dataset = BeatsDataset(allowed_map_ids=val_map_ids)
        train_loader = DataLoader(
            train_dataset,
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            persistent_workers=True,
        )

        return train_loader, val_loader

    def train_loop(
        self,
        model: SmallBeatsNet,
        loader: DataLoader,
        optimizer: torch.optim.Adam,
        criterion: torch.nn.CrossEntropyLoss,
    ):
        model.train()
        total_loss = 0

        for audio, prev_tokens, targets in loader:
            audio: torch.Tensor = audio.to(self.device)
            prev_tokens: torch.Tensor = prev_tokens.to(self.device)
            targets: torch.Tensor = targets.to(self.device)

            optimizer.zero_grad()
            predictions, _ = model(audio, prev_tokens)
            loss: torch.Tensor = criterion(predictions.permute(0, 2, 1), targets)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        return total_loss

    def val_loop(
        self,
        model: SmallBeatsNet,
        loader: DataLoader,
        criterion: torch.nn.CrossEntropyLoss,
    ):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for audio, prev_tokens, targets in loader:
                audio: torch.Tensor = audio.to(self.device)
                prev_tokens: torch.Tensor = prev_tokens.to(self.device)
                targets: torch.Tensor = targets.to(self.device)

                predictions, _ = model(audio, prev_tokens)
                loss: torch.Tensor = criterion(predictions.permute(0, 2, 1), targets)
                total_loss += loss.item()

        return total_loss

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
                        round(t_loss, LOSS_LOG_ROUNDING),
                        round(v_loss, LOSS_LOG_ROUNDING),
                    ]
                )

        print(f"Training run saved to {self.run_dir}")

    def save_meta(self, class_weights: torch.Tensor):
        meta = {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "train_split": TRAIN_SPLIT,
            "weight_decay": WEIGHT_DECAY,
            "schedule_factor": SCHEDULE_FACTOR,
            "schedule_patience": SCHEDULE_PATIENCE,
            "class_weight_clamp_max": CLASS_WEIGHT_CLAMP_MAX,
            "class_weight_clamp_min": CLASS_WEIGHT_CLAMP_MIN,
            "empty_token_weight": EMPTY_TOKEN_WEIGHT,
            "loss_log_rounding": LOSS_LOG_ROUNDING,
            "overfit_mode": self.overfit_mode,
            "device": device_type,
            "class_weights": class_weights.tolist(),
        }

        with open(self.run_dir / "meta.json", "w") as f:
            f.write(json.dumps(meta, indent=2))

    def train(self):
        class_weights = self.get_class_weights()
        self.save_meta(class_weights)
        train_loader, val_loader = self.load_datasets()

        model = SmallBeatsNet()
        model.to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHEDULE_FACTOR, patience=SCHEDULE_PATIENCE
        )
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")

        for epoch in range(EPOCHS):
            train_loss = self.train_loop(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
            )
            val_loss = self.val_loop(
                model=model, loader=val_loader, criterion=criterion
            )

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(
                f"Epoch: {epoch} | Train loss: {round(avg_train_loss, LOSS_LOG_ROUNDING)} | Val loss: {round(avg_val_loss, LOSS_LOG_ROUNDING)}"
            )

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(obj=model.state_dict(), f=self.run_dir / "best_model.pth")
                print(
                    f"New best model with val loss {round(avg_val_loss, LOSS_LOG_ROUNDING)} saved"
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
