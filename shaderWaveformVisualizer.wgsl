struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) color : vec4f,
};

@vertex
fn vs_head(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var pos = array<vec2f, 3>(
        vec2f(0.0, 0.5),
        vec2f(-0.5, -0.5),
        vec2f(0.5, -0.5)
    );

    var colors = array<vec3f, 3>(
        vec3f(1.0, 0.0, 0.0),
        vec3f(0.0, 1.0, 0.0),
        vec3f(0.0, 0.0, 1.0)
    );

    var output : VertexOutput;
    output.position = vec4f(pos[vertexIndex], 0.0, 1.0);
    output.color = vec4f(colors[vertexIndex], 1.0);
    return output;
}

@fragment
fn fs_head(input : VertexOutput) -> @location(0) vec4f {
    return input.color;
}