#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import json
import os

# --- Parameters ---
bin_file = "peak.bin"
json_file = "waveform.json"

# --- Load channel count from JSON ---
if not os.path.exists(json_file):
    raise FileNotFoundError(f"{json_file} not found. Run generate-bins.py first.")
with open(json_file, "r") as f:
    meta = json.load(f)
channel_count = meta["channel_count"]
print(f"Detected channel count: {channel_count}")

# --- Load binary file ---
data = np.fromfile(bin_file, dtype=np.float32)
print(f"Loaded {data.size} samples")

# --- Reshape to (sample_count, channel_count) ---
sample_count = data.size // channel_count
data = data[:sample_count * channel_count]  # trim if not divisible
data = data.reshape((sample_count, channel_count))
print(f"Reshaped to {data.shape} (samples x channels)")

# --- Optional: pick first channel ---
# channel0 = data[:, 0]

# --- Optional: normalize to 0..1 ---
# channel0_norm = channel0 / np.max(channel0)

# --- Plot all channels ---
plt.figure(figsize=(15, 6))
for ch in range(channel_count):
    plt.plot(data[:, ch] + ch, label=f'Channel {ch+1}')  # offset each line for visibility
plt.title("Waveform Channels")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude + Offset")
plt.legend()
plt.grid(True)
plt.show()
