import numpy as np
from scipy.signal import resample

num_channels = 16
image_width = 4096          # target horizontal pixels
input_file = "output.raw"   # it has to be float32

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

def perceptual_scale(x, mode="sqrt"):
    if mode == "log":
        return np.log10(1 + np.abs(x)) / np.log10(1 + 1)
    elif mode == "sqrt":
        return np.sqrt(np.abs(x))
    elif mode == "cbrt":
        return np.cbrt(np.abs(x))
    else:  # lin
        return np.abs(x)

# Read raw PCM
raw = np.fromfile(input_file, dtype=np.float32)
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

    # True peak per channel (x60 slower than mean)
    # peak[x] = true_peak(segment, upsample=8)

    # Mean amplitude per channel
    # mean[x] = np.mean(np.abs(segment), axis=0)

    # RMS amplitude per channel
    mean[x] = np.sqrt(np.mean(segment.astype(np.float32)**2, axis=0))
    mean[x] = perceptual_scale(mean[x], mode="sqrt")

# Save as float32 binary files
peak.astype(np.float32).tofile("peak.bin")
mean.astype(np.float32).tofile("mean.bin")
