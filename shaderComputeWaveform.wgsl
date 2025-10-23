struct Params {
    firstChannelPeak: f32,
    boost: f32,
    offset: f32,
    _pad: f32,             // padding to align next vec2 to 16 bytes
    width: f32,
    height: f32,
};

@group(0) @binding(0) var<storage, read> waveform: array<f32>;
@group(0) @binding(1) var<storage, read> peaks: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> computeOutput : array<vec2<f32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let width = u32(params.width);
    let height = u32(params.height);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let channelCount = 16u;
    let baseHeight = height / channelCount;
    let remainder = height % channelCount;

    // Compute which channels get the extra pixel: distributed from the center outward
    var extraForChannel = array<u32, 16u>();
    if (remainder > 0u) {
        var remaining = remainder;
        var offset = 0u;
        loop {
            if (remaining == 0u) { break; }
            let leftIndex = i32(channelCount / 2u) - 1 - i32(offset);
            let rightIndex = i32(channelCount / 2u) + i32(offset);
            if (leftIndex >= 0 && remaining > 0u) {
                extraForChannel[u32(leftIndex)] = 1u;
                remaining -= 1u;
            }
            if (remaining == 0u) { break; }
            if (rightIndex < i32(channelCount) && remaining > 0u) {
                extraForChannel[u32(rightIndex)] = 1u;
                remaining -= 1u;
            }
            offset += 1u;
        }
    }

    var y = gid.y;
    var accumulated = 0u;
    var channelIndex = 0u;

    loop {
        let h = baseHeight + extraForChannel[channelIndex];
        if (y < accumulated + h) {
            break;
        }
        accumulated += h;
        channelIndex += 1u;
        if (channelIndex >= channelCount) {
            break;
        }
    }

    channelIndex = channelCount - 1u - channelIndex;

    let localY_pixel = y - accumulated;
    let channelHeight = baseHeight + extraForChannel[channelIndex];

    let samplesPerChannel = arrayLength(&waveform) / channelCount;

    // Compute sample range per pixel
    let samplesPerPixel = f32(samplesPerChannel) / f32(width);
    let startSample = u32(floor(f32(gid.x) * samplesPerPixel));
    let endSample = u32(floor(f32(gid.x + 1u) * samplesPerPixel));

    var maxValue = 0.0;
    for (var i = startSample; i < endSample && i < samplesPerChannel; i = i + 1u) {
        let index = i * channelCount + channelIndex;
        let rawValue = waveform[index];
        let normalizedValue = rawValue / params.firstChannelPeak;
        if (normalizedValue > maxValue) {
            maxValue = normalizedValue;
        }
    }

    let waveformIndex = startSample * channelCount + channelIndex;
    let rawPeakValue = peaks[waveformIndex];
    let peakValue = (rawPeakValue / params.firstChannelPeak) * params.boost + params.offset;

    var waveformColor = vec2<f32>(0.0, 0.0);
    let lineCenter_pixel = channelHeight / 2u;
    let halfHeight_pixel = u32(maxValue * 0.5 * f32(channelHeight) + 0.5);
    let y_down = i32(lineCenter_pixel) - i32(halfHeight_pixel);
    let y_up   = i32(lineCenter_pixel) + i32(halfHeight_pixel);
    let localY = i32(localY_pixel);

    // Center line
    if (localY >= y_down && localY <= y_up) {
        waveformColor = vec2<f32>(1.0, 1.0);
    }

    let maskColor = vec2<f32>(peakValue, 1.0);
    let brightness = f32(min(waveformColor.x, maskColor.y));
    let alpha = waveformColor.y;

    let index = gid.y * width + gid.x;
    computeOutput[index] = vec2<f32>(brightness, alpha);
}
