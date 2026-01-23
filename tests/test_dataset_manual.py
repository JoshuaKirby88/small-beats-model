from small_beats_model.dataset import BeatsDataset
from small_beats_model.preprocessing import N_MFCC, TARGET_FRAMES, WINDOW_BEATS


class DatasetTester:
    def __init__(self):
        self.dataset = BeatsDataset()

    def run(self):
        print(f"Length of dataset: {len(self.dataset)}")
        sample = self.dataset[0]
        print(f"Audio shape: {sample[0].shape}")
        print(f"Label shape: {sample[1].shape}")

        print(
            f"Shape of audio is ({N_MFCC}, {TARGET_FRAMES}): {(sample[0].shape[0] == N_MFCC, sample[0].shape[1] == TARGET_FRAMES)}"
        )
        print(
            f"Shape of label is {WINDOW_BEATS * 4}: {sample[1].shape[0] == WINDOW_BEATS * 4}"
        )


if __name__ == "__main__":
    tester = DatasetTester()
    tester.run()
