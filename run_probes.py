from src.probe_dataset import ProbeDataset
from src.linear_probe import LinearProbe

# This file pulls data from data/processed and runs linear probes on the layers

dataset = ProbeDataset()

for layer in dataset.available_layers():

    X, y = dataset.get_xy(
        layer,
        "refused"
    )

    probe = LinearProbe()

    results = probe.fit(X, y)

    print(
        layer,
        results["accuracy"]
    )

print(dataset.metadata["refused"].value_counts())
print(dataset.metadata["harmful"].value_counts())