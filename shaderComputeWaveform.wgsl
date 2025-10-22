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

// --- Compute channel and local pixel coordinates ---
let channelCount = 16u;
let channelHeight = dims.y / channelCount; // integer pixels per channel
let channelIndex = channelCount - 1u - (gid.y / channelHeight);
let localY_pixel = gid.y - (channelCount - 1u - channelIndex) * channelHeight;

// --- Read waveform sample ---
let samplesPerChannel = arrayLength(&waveform) / channelCount;
let uvX = f32(gid.x) / f32(dims.x);
let sampleIndex = u32(uvX * f32(samplesPerChannel - 1u));
let waveformIndex = sampleIndex * channelCount + channelIndex;
let value = waveform[waveformIndex];

var color = vec4f(0.0);

// --- Pixel-perfect mirrored waveform ---
let lineCenter_pixel = channelHeight / 2u; // baseline at center of channel
let halfHeight_pixel = u32(value * 0.5 * f32(channelHeight) + 0.5); // rounded half-height

let y_down = i32(lineCenter_pixel) - i32(halfHeight_pixel);
let y_up   = i32(lineCenter_pixel) + i32(halfHeight_pixel);
let localY = i32(localY_pixel);

// Downward half-waveform
if (localY >= y_down && localY <= i32(lineCenter_pixel)) {
    color = vec4f(1.0, 1.0, 1.0, 1.0);
}

// Upward mirrored half-waveform
if (localY > i32(lineCenter_pixel) && localY <= y_up) {
    color = vec4f(1.0, 1.0, 1.0, 1.0);
}

// 1-pixel baseline
if (localY == i32(lineCenter_pixel)) {
    color = vec4f(1.0, 1.0, 1.0, 1.0);
}

textureStore(outImage, vec2<i32>(gid.xy), color);
}