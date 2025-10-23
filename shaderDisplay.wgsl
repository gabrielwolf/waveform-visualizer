struct Params {
    firstChannelPeak: f32,
    boost: f32,
    offset: f32,
    _pad: f32,             // padding to align next vec2 to 16 bytes
    width: f32,
    height: f32,
};

@group(0) @binding(0) var<storage, read_write> computeOutput : array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    var pos = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
    );
    var uv = (pos[vid] + vec2f(1.0)) * 0.5;
    var out: VSOut;
    out.pos = vec4f(pos[vid], 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4f {
    let width = u32(params.width);
    let height = u32(params.height);
    let index = u32(floor(input.uv.y * f32(height))) * width + u32(floor(input.uv.x * f32(width)));
    let data = computeOutput[index];
    return vec4f(data.x, data.x, data.x, data.y);
}
