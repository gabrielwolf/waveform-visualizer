#!/usr/bin/env python3

import argparse
import os
import uuid

import numpy as np
import subprocess
import sys
from scipy.signal import resample

binary_resolution_horizontal = 4096 # target horizontal pixels

def get_channel_count(input_file):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels", "-of", "default=noprint_wrappers=1:nokey=1", input_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return int(result.stdout.strip())
    except subprocess.CalledProcessError as error:
        print(f"Error reading file: {error.stderr.strip()}")
        sys.exit(1)

def generate_raw(input_file, output_dir):
    raw_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + ".raw")
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-f", "f32le", raw_path
    ]
    subprocess.run(cmd, check=True)
    return raw_path

def true_peak(segment, upsample=4):
    """
    segment: (N, channel_count)
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

def generate_peak_and_mean(raw_path, channels, output_dir):
    # Read raw PCM
    raw = np.fromfile(raw_path, dtype=np.float32)
    original_sample_count = len(raw) // channels
    raw = raw.reshape((original_sample_count, channels))

    # Bin samples to match image width
    samples_per_bin = original_sample_count / binary_resolution_horizontal

    # Prepare arrays for peak and mean
    peak = np.zeros((binary_resolution_horizontal, channels), dtype=np.float32)
    mean = np.zeros((binary_resolution_horizontal, channels), dtype=np.float32)

    for x in range(binary_resolution_horizontal):
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

    peak_path = os.path.join(output_dir, "peak.bin")
    mean_path = os.path.join(output_dir, "mean.bin")

    # Save as float32 binary files
    peak.astype(np.float32).tofile(peak_path)
    mean.astype(np.float32).tofile(mean_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an audio file and generate waveform image binaries.")
    parser.add_argument("input_file", help="Input audio file")
    parser.add_argument("--output-dir", default=None, help="Directory for output files")
    parser.add_argument(
        "--uuid-output-dir",
        action="store_true",
        help="Append a UUID to the output directory name to avoid collisions"
    )
    args = parser.parse_args()
    input_file = args.input_file

    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        sys.exit(1)

    fresh_uuid = uuid.uuid4()
    output_dir = os.path.splitext(os.path.basename(input_file))[0]

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.splitext(os.path.basename(input_file))[0]
        if getattr(args, "uuid_output_dir", False):
            output_dir = f"{output_dir}_{uuid.uuid4()}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        channels = get_channel_count(input_file)
        print(f"{channels} channels detected.")
    except SystemExit:
        raise
    except Exception as error:
        print(f"Unexpected error: {error}")
        sys.exit(1)

    try:
        channels = get_channel_count(input_file)
        raw_path = generate_raw(input_file, output_dir)
        meta = generate_peak_and_mean(raw_path, channels, output_dir)
        os.remove(raw_path)
        print(f"Successfully generated waveform and background binaries to ./{output_dir}/")
    except SystemExit:
        raise
    except Exception as error:
        print(f"Error during processing: {error}")
        sys.exit(1)
