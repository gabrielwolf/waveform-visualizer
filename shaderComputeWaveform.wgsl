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
    let index = u32(uv.x * f32(arrayLength(&waveform) - 1u));
    let value = waveform[index]; // 0..1

    let line_thickness = 0.005;

    var color = vec4f(0.0);
    if (abs(uv.y - value) < line_thickness) {
        color = vec4f(value, 1.0 - value, 0.2, 1.0);
    }

    textureStore(outImage, vec2<i32>(gid.xy), color);
}

