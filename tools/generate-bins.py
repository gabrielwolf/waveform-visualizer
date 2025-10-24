#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import uuid

import numpy as np
import subprocess
import sys
from scipy.signal import resample

image_width = 4096 # target horizontal pixels

def get_channel_count(file_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return int(result.stdout.strip())
    except subprocess.CalledProcessError as error:
        print(f"Error reading file: {error.stderr.strip()}")
        sys.exit(1)

def generate_raw(file_path):
    raw_path = os.path.splitext(file_path)[0] + ".raw"
    cmd = [
        "ffmpeg", "-y", "-i", file_path,
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
    sample_count = len(raw) // channels
    raw = raw.reshape((sample_count, channels))

    # Bin samples to match image width
    samples_per_bin = sample_count / image_width

    # Prepare arrays for peak and mean
    peak = np.zeros((image_width, channels), dtype=np.float32)
    mean = np.zeros((image_width, channels), dtype=np.float32)

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

    peak_path = os.path.join(output_dir, "peak.bin")
    mean_path = os.path.join(output_dir, "mean.bin")

    # Save as float32 binary files
    peak.astype(np.float32).tofile(peak_path)
    mean.astype(np.float32).tofile(mean_path)
    return {
        "channels": channels,
        "sample_count": sample_count,
    }

def generate_json(file_path, meta, output_dir):
    data = {
        "input_file": file_path,
        "channel_count": meta["channels"],
        "sample_count": meta["sample_count"],
        "image_width": image_width,
    }
    json_path = os.path.join(output_dir, "waveform.json")
    pathlib.Path(json_path).write_text(json.dumps(data, indent=2))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an audio file and generate waveform image binaries.")
    parser.add_argument("file_path", help="Input audio file")
    parser.add_argument(
        "--uuid-output-dir",
        action="store_true",
        help="Append a UUID to the output directory name to avoid collisions"
    )
    args = parser.parse_args()
    file_path = args.file_path

    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)

    fresh_uuid = uuid.uuid4()
    output_dir = os.path.splitext(os.path.basename(file_path))[0]

    if args.uuid_output_dir:
        output_dir = f"{output_dir}_{fresh_uuid}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        channels = get_channel_count(file_path)
        print(f"Number of channels: {channels}")
    except SystemExit:
        raise
    except Exception as error:
        print(f"Unexpected error: {error}")
        sys.exit(1)

    try:
        channels = get_channel_count(file_path)
        raw_path = generate_raw(file_path)
        meta = generate_peak_and_mean(raw_path, channels, os.path.splitext(file_path)[0])
        os.remove(raw_path)
        generate_json(file_path, meta, os.path.splitext(file_path)[0])
        print(f"Successfully generated waveform and background images to ./{output_dir}/")
    except SystemExit:
        raise
    except Exception as error:
        print(f"Error during processing: {error}")
        sys.exit(1)
