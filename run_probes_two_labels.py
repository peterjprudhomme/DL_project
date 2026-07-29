from src.probe_dataset import ProbeDataset
from src.linear_probe import LinearProbe

# This file pulls data from data/processed and runs linear probes on the layers
# for BOTH labels (harmful and refused), one independent probe per (label, layer).

LABELS = ["harmful", "refused"]

dataset = ProbeDataset()

# accuracy[label][layer] -> held-out accuracy of that probe
accuracy = {label: {} for label in LABELS}

for label in LABELS:

    print(f"\n=== label: {label} ===")

    for layer in dataset.available_layers():

        X, y = dataset.get_xy(
            layer,
            label
        )

        probe = LinearProbe()

        results = probe.fit(X, y)

        accuracy[label][layer] = results["accuracy"]

        print(
            layer,
            results["accuracy"]
        )

# Compact summary table: rows = layers, columns = labels.
print("\n=== summary (accuracy) ===")
header = "layer".ljust(8) + "".join(label.ljust(12) for label in LABELS)
print(header)
for layer in dataset.available_layers():
    row = str(layer).ljust(8)
    for label in LABELS:
        row += f"{accuracy[label][layer]:.4f}".ljust(12)
    print(row)

# Class balance for each label so accuracy is read in context.
print()
for label in LABELS:
    print(f"{label} value_counts:")
    print(dataset.metadata[label].value_counts())
    print()
