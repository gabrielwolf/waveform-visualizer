@group(0) @binding(0) var outImage: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> time: f32;
@group(0) @binding(2) var<storage, read> waveform: array<f32>;
@group(0) @binding(3) var<storage, read> peaks: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outImage);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let channelCount = 16u;
    let channelHeight = dims.y / channelCount;
    let channelIndex = channelCount - 1u - (gid.y / channelHeight);
    let localY_pixel = gid.y - (channelCount - 1u - channelIndex) * channelHeight;

    let samplesPerChannel = arrayLength(&waveform) / channelCount;

    // Compute sample range per pixel
    let samplesPerPixel = f32(samplesPerChannel) / f32(dims.x);
    let startSample = u32(floor(f32(gid.x) * samplesPerPixel));
    let endSample = u32(floor(f32(gid.x + 1u) * samplesPerPixel));

    var maxValue = 0.0;
    for (var i = startSample; i < endSample && i < samplesPerChannel; i = i + 1u) {
        let idx = i * channelCount + channelIndex;
        let v = waveform[idx];
        if (v > maxValue) {
            maxValue = v;
        }
    }

    let waveformIndex = startSample * channelCount + channelIndex;
    let peakValue = peaks[waveformIndex] * 0.8 + 0.5;

    var waveformColor = vec4f(0.0);
    let lineCenter_pixel = channelHeight / 2u;
    let halfHeight_pixel = u32(maxValue * 0.5 * f32(channelHeight) + 0.5);
    let y_down = i32(lineCenter_pixel) - i32(halfHeight_pixel);
    let y_up   = i32(lineCenter_pixel) + i32(halfHeight_pixel);
    let localY = i32(localY_pixel);

    if (localY >= y_down && localY <= y_up) {
        waveformColor = vec4f(1.0, 1.0, 1.0, 1.0);
    }

    let maskColor = vec4f(peakValue, peakValue, peakValue, 1.0);
    let finalColor = vec4f(
        min(waveformColor.r, maskColor.r),
        min(waveformColor.g, maskColor.g),
        min(waveformColor.b, maskColor.b),
        waveformColor.a
    );

    textureStore(outImage, vec2<i32>(gid.xy), finalColor);
}
