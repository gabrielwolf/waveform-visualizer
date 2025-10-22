@group(0) @binding(0) var outImage: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> time: f32;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(outImage);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    // Normalize coordinates to [0,1]
    let uv = vec2<f32>(f32(gid.x) / f32(dims.x), f32(gid.y) / f32(dims.y));

    // Simple animated gradient
    let r = uv.x;
    let g = uv.y;
    let b = 0.5 + 0.5 * sin(time + uv.x * 10.0);
    textureStore(outImage, vec2<i32>(gid.xy), vec4f(r, g, b, 1.0));
}
