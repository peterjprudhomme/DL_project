from pathlib import Path

import pandas as pd
import torch
import numpy as np


class ProbeDataset:

    def __init__(self, data_dir="data/processed"):

        data_dir = Path(data_dir)

        self.metadata = pd.read_csv(
            data_dir / "metadata.csv"
        )

        self.activations = torch.load(
            data_dir / "activations.pt",
            map_location="cpu"
        )

    def available_layers(self):
        first = next(iter(self.activations.values()))
        return sorted(first.keys())

    def get_xy(
        self,
        layer,
        label
    ):

        if label not in self.metadata.columns:
            raise ValueError(f"{label} not found in metadata.")

        X = []
        y = []

        for _, row in self.metadata.iterrows():

            idx = row["id"]

            activation = self.activations[idx][layer]

            X.append(
                activation.numpy()
            )

            y.append(
                int(row[label])
            )

        X = np.stack(X)
        y = np.array(y)

        return X, y