import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
bin_file = "peak.bin"  # or "peak.bin"
num_channels = 16      # matches your waveform processing
dtype = np.uint16      # as in resize.py

# --- Load binary file ---
data = np.fromfile(bin_file, dtype=dtype)
print(f"Loaded {data.size} samples")

# --- Reshape to (num_samples, num_channels) ---
num_samples = data.size // num_channels
data = data[:num_samples * num_channels]  # trim if not divisible
data = data.reshape((num_samples, num_channels))
print(f"Reshaped to {data.shape} (samples x channels)")

# --- Pick first channel ---
channel0 = data[:, 0]

# --- Optional: normalize to 0..1 ---
# channel0_norm = channel0 / np.max(channel0)

# --- Plot ---
plt.figure(figsize=(15, 4))
plt.plot(channel0, color="black")
plt.title("Waveform - Channel 0")
plt.xlabel("Sample Index")
plt.ylabel("Normalized Amplitude")
plt.grid(True)
plt.show()