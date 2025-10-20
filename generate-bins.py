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
peak = ((peak / 32767) * max_val)
mean = ((mean / 32767) * max_val)

# Normalize relative to first channel max peak and mean
first_channel_peak_max = np.max(peak[:, 0])
first_channel_mean_max = np.max(mean[:, 0])

# Scale all channels so that first channel max peak maps to 4095
peak = (peak / first_channel_peak_max) * max_val
mean = (mean / first_channel_mean_max) * max_val

# Clip to 0..4095 and cast to uint16
peak = np.clip(peak, 0, max_val).astype(dtype)
mean = np.clip(mean, 0, max_val).astype(dtype)

max_peak_ch0 = np.max(peak[:, 0])
max_mean_ch0 = np.max(mean[:, 0])
print(f"First channel max peak: {max_peak_ch0}")
print(f"First channel max mean: {max_mean_ch0}")

# Save compact binary for GPU
peak.tofile("peak.bin")
mean.tofile("mean.bin")
