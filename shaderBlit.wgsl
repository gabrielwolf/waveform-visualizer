@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var inputSampler: sampler;

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
    return textureSample(inputTex, inputSampler, input.uv);
}