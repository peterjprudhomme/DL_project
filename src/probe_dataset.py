from pathlib import Path

import pandas as pd
import torch
import numpy as np

from src.config import model_slug


class ProbeDataset:

    def __init__(self, data_dir=None):

        if data_dir is None:
            data_dir = Path("data/processed") / model_slug()

        data_dir = Path(data_dir)

        self.metadata = pd.read_json(
            data_dir / "metadata.json"
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