from pathlib import Path

import torch
from torch.utils.data import DataLoader

from small_beats_model.dataset import BeatsDataset
from small_beats_model.model import SmallBeatsNet

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 10
TRAIN_VAL_SPLIT = 0.8
MODEL_FILE = Path("models/best_model.pth")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Train:
    def __init__(
        self,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        train_val_split=TRAIN_VAL_SPLIT,
        model_file=MODEL_FILE,
    ):
        if not model_file.parent.exists():
            model_file.parent.mkdir(parents=True, exist_ok=True)

        dataset = BeatsDataset()
        train_size = int(train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        train_loader = DataLoader(
            train_dataset, shuffle=True, num_workers=0, batch_size=batch_size
        )
        val_loader = DataLoader(
            val_dataset, shuffle=True, num_workers=0, batch_size=batch_size
        )

        model = SmallBeatsNet()
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float("inf")

        for epoch in range(epochs):
            model.train()

            for batch_i, (audio, targets) in enumerate(train_loader):
                audio: torch.Tensor = audio.to(device)
                targets: torch.Tensor = targets.to(device)
                optimizer.zero_grad()
                predictions = model(audio)
                loss = criterion(predictions.permute(0, 2, 1), targets)
                loss.backward()
                optimizer.step()

                if batch_i % 10 == 0:
                    print(f"Epoch {epoch} | Batch {batch_i} | Loss: {loss.item():.4f}")

            model.eval()

            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for audio, targets in val_loader:
                    audio, targets = audio.to(device), targets.to(device)
                    predictions = model(audio)
                    loss = criterion(predictions.permute(0, 2, 1), targets)
                    val_loss += loss.item()
                    correct += (predictions.argmax(dim=2) == targets).sum().item()
                    total += targets.numel()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_FILE)
                print(f"New best model with validation loss {val_loss} saved")

            avg_loss = val_loss / len(val_loader)
            accuracy = correct / total

            print(
                f"Validation loss: {val_loss} | Average loss: {avg_loss} | Accuracy: {accuracy}"
            )


if __name__ == "__main__":
    train = Train()
