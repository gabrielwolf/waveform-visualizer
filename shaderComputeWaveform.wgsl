@group(0) @binding(0) var outImage: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> time: f32;
@group(0) @binding(2) var<storage, read> waveform: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(outImage);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let uv = vec2<f32>(f32(gid.x) / f32(dims.x), f32(gid.y) / f32(dims.y));

    let channelCount = 16u;
    let samplesPerChannel = arrayLength(&waveform) / channelCount;
    let channelHeight = 1.0 / f32(channelCount);

    // Determine which channel this pixel is in
    let flippedY = 1.0 - uv.y;
    let channelIndex = u32(floor(flippedY / channelHeight));
    let localY = fract(flippedY / channelHeight); // y position within that band

    // Read from that channel’s data range
    let sampleIndex = u32(uv.x * f32(samplesPerChannel - 1u));
    let waveformIndex = sampleIndex * channelCount + channelIndex;
    let value = waveform[waveformIndex];

    var color = vec4f(0.0);

    // 1-pixel horizontal baseline for each channel
    let pixelHeight = (1.0 / f32(dims.y)) / channelHeight;
    let lineCenter = 0.5;
    let halfHeight = value * 0.5; // waveform scaled to half channel
    let y_start = lineCenter - pixelHeight * 0.5;
    let y_end   = lineCenter + pixelHeight * 0.5;

    if (localY >= y_start && localY <= y_end) {
        color = vec4f(1.0, 1.0, 1.0, 1.0);
    }

    // Waveform fill
    let y_down = lineCenter - halfHeight;  // waveform downward
    let y_up   = lineCenter + halfHeight;  // waveform upward
    if (localY >= y_down && localY <= lineCenter) {
        // original downward half-waveform
        color = vec4f(1.0, 1.0, 1.0, 1.0);
    }

    if (localY > lineCenter && localY <= y_up) {
        // mirrored upward half-waveform
        color = vec4f(1.0, 1.0, 1.0, 1.0);
    }

    textureStore(outImage, vec2<i32>(gid.xy), color);
}