import torch
from torch.utils.data.dataloader import DataLoader

from small_beats_model.dataset import BeatsDataset
from small_beats_model.model import SmallBeatsNet

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 10

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Train:
    def __init__(
        self, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS
    ):
        dataset = BeatsDataset()
        dataloader = DataLoader(
            dataset, shuffle=True, num_workers=0, batch_size=batch_size
        )

        model = SmallBeatsNet()
        model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch_i, (audio, targets) in enumerate(dataloader):
                audio: torch.Tensor = audio.to(device)
                targets: torch.Tensor = targets.to(device)
                optimizer.zero_grad()
                predictions = model(audio)
                loss = criterion(predictions.permute(0, 2, 1), targets)
                loss.backward()
                optimizer.step()

                if batch_i % 10 == 0:
                    print(f"Epoch {epoch} | Batch {batch_i} | Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train = Train()
