struct Params {
    firstChannelPeak: f32,
    boost: f32,
    offset: f32,
    channelCount: f32,
    canvasWidth: f32,
    canvasHeight: f32,
    groupMask: vec3<f32>,  // groups of flac packets 1-4, 5-9 and 10-16 with their corresponding mask [0;1]
    _pad: f32,
};

// We need to correct for rounding errors
struct ChannelLayout {
    // 64 channels packed into 16 vec4s (16-byte aligned)
    channelOffset: array<vec4<f32>, 16>,
    channelHeight: array<vec4<f32>, 16>,
};

@group(0) @binding(0) var<storage, read> waveform: array<f32>;
@group(0) @binding(1) var<storage, read> peaks: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<uniform> channelLayout: ChannelLayout;
@group(0) @binding(4) var<storage, read_write> computeOutput: array<vec2<f32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let canvasWidth = u32(params.canvasWidth);
    let canvasHeight = u32(params.canvasHeight);
    if (gid.x >= canvasWidth || gid.y >= canvasHeight) {
        return;
    }

    let channelCount = u32(params.channelCount);

    // Determine which channel this pixel belongs to
    var channelIndex = 0u;
    let y = f32(gid.y);
    for (var i = 0u; i < channelCount; i = i + 1u) {
        let group = i / 4u;
        let sub = i % 4u;
        let offset = channelLayout.channelOffset[group][sub];
        let chHeight = channelLayout.channelHeight[group][sub];
        if (y >= offset && y < offset + chHeight) {
            channelIndex = i;
            break;
        }
    }

    // Get offset and height for the channel
    let group = channelIndex / 4u;
    let sub = channelIndex % 4u;
    let channelOffset = channelLayout.channelOffset[group][sub];
    let channelHeight = channelLayout.channelHeight[group][sub];

    // Local Y relative to channel
    let localY_pixel = y - channelOffset;
    let samplesPerChannel = arrayLength(&waveform) / channelCount;

    // Compute sample range for this pixel (we need to iterate over them, in order to retain full quality)
    let samplesPerPixel = f32(samplesPerChannel) / f32(canvasWidth);
    let startSample = u32(floor(f32(gid.x) * samplesPerPixel));
    let endSample = u32(floor(f32(gid.x + 1u) * samplesPerPixel));

    // Compute max waveform value for this pixel
    var maxValue = 0.0;
    for (var i = startSample; i < endSample && i < samplesPerChannel; i = i + 1u) {
        let index = i * channelCount + channelIndex;
        let rawValue = waveform[index];
        let normalizedValue = rawValue / params.firstChannelPeak;
        if (normalizedValue > maxValue) {
            maxValue = normalizedValue;
        }
    }

    // Get peak value for brightness
    let waveformIndex = startSample * channelCount + channelIndex;
    let rawPeakValue = peaks[waveformIndex];
    let peakValue = (rawPeakValue / params.firstChannelPeak) * params.boost + params.offset;

    // Compute waveform and center line for each channel
    let lineCenter_pixel = channelHeight / 2.0;
    let halfHeight_pixel = maxValue * 0.5 * channelHeight;
    let y_down = lineCenter_pixel - halfHeight_pixel + 0.0;
    let y_up   = lineCenter_pixel + halfHeight_pixel + 1.0;

    var waveformColor = vec2<f32>(0.0, 0.0);
    if (localY_pixel >= y_down && localY_pixel <= y_up) {
        waveformColor = vec2<f32>(1.0, 1.0);
    }

    var maskGroupIndex: u32 = 0u;

    // Ambisonics masking groups: channels 1-4 → group 0, 5-9 → group 1, 10-16 → group 2
    if (channelIndex <= 3u) {
        maskGroupIndex = 0u;
    } else if (channelIndex <= 8u) {
        maskGroupIndex = 1u;
    } else {
        maskGroupIndex = 2u;
    }

    // Use peakValue as brightness for mask
    let maskFraction = params.groupMask[maskGroupIndex]; // 0..1
    let xFraction = f32(gid.x) / f32(canvasWidth);       // horizontal fraction of pixel

    var maskedPeak = peakValue;
    if (xFraction > maskFraction) {
        maskedPeak = 1.0; // plain white
    }

    let brightness = min(waveformColor.x, maskedPeak);
    let alpha = waveformColor.y;

    let index = gid.y * canvasWidth + gid.x;
    computeOutput[index] = vec2<f32>(brightness, alpha);
}