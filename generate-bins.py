import numpy as np

# Parameters
num_channels = 16
image_width = 5000  # target horizontal pixels
bit_depth = 12  # per value (peak or mean)
dtype = np.uint16  # can hold 12-bit data

# Read raw PCM
raw = np.fromfile("output.raw", dtype=np.int16)
num_samples = len(raw) // num_channels
raw = raw.reshape((num_samples, num_channels))

# Bin samples to match image width
samples_per_bin = num_samples / image_width

# Prepare arrays for peak and mean
peak = np.zeros((image_width, num_channels), dtype=dtype)
mean = np.zeros((image_width, num_channels), dtype=dtype)

for x in range(image_width):
    start = int(x * samples_per_bin)
    end = int((x + 1) * samples_per_bin)
    segment = raw[start:end]

    # Peak amplitude per channel
    peak[x] = np.max(np.abs(segment), axis=0)

    # Mean amplitude per channel
    mean[x] = np.mean(np.abs(segment), axis=0)

# Quantize to 12 bits
max_val = 2**bit_depth - 1
peak = ((peak / 32767) * max_val).astype(dtype)
mean = ((mean / 32767) * max_val).astype(dtype)

# Save compact binary for GPU
peak.tofile("peak.bin")
mean.tofile("mean.bin")
