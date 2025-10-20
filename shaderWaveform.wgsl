struct VertexInput {
    @location(0) position: vec2f,
};

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) color: vec4f,
};

@vertex
fn vs_head(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4f(input.position, 0.0, 1.0);
    output.color = vec4f(1.0, 1.0, 1.0, 1.0);
    return output;
}

@fragment
fn fs_head(input: VertexOutput) -> @location(0) vec4f {
    return input.color;
}
