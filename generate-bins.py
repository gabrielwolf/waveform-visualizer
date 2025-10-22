import numpy as np
from scipy.signal import resample

def true_peak(segment, upsample=4):
    """
    segment: (N, num_channels)
    upsample: factor to interpolate
    returns: true peak per channel
    """
    N, C = segment.shape
    true_peaks = np.zeros(C, dtype=np.float64)

    for ch in range(C):
        # Upsample by factor
        seg_upsampled = resample(segment[:, ch], N * upsample)
        true_peaks[ch] = np.max(np.abs(seg_upsampled))

    return true_peaks

# Parameters
num_channels = 16
image_width = 4096  # target horizontal pixels

# Read raw PCM
raw = np.fromfile("output.raw", dtype=np.int16)
num_samples = len(raw) // num_channels
raw = raw.reshape((num_samples, num_channels))

# Bin samples to match image width
samples_per_bin = num_samples / image_width

# Prepare arrays for peak and mean
peak = np.zeros((image_width, num_channels), dtype=np.float32)
mean = np.zeros((image_width, num_channels), dtype=np.float32)

for x in range(image_width):
    start = int(x * samples_per_bin)
    end = int((x + 1) * samples_per_bin)
    segment = raw[start:end]

    # Peak amplitude per channel
    peak[x] = np.max(np.abs(segment), axis=0)

    # True peak per channel
    # peak[x] = true_peak(segment, upsample=8)

    # Mean amplitude per channel
    mean[x] = np.mean(np.abs(segment), axis=0)

    # RMS amplitude per channel
    # mean[x] = np.sqrt(np.mean(segment.astype(np.float64)**2, axis=0))

# Normalize relative to first channel max peak and mean
first_channel_peak_max = np.max(peak[:, 0])
first_channel_mean_max = np.max(mean[:, 0])

# Scale all channels so that first channel max peak maps to 1.0
peak = peak / first_channel_peak_max
mean = mean / first_channel_mean_max

# Save as float32 binary files
peak.astype(np.float32).tofile("peak.bin")
mean.astype(np.float32).tofile("mean.bin")
