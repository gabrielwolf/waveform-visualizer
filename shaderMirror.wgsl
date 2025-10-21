struct VertexInput {
    @location(0) position: vec2f,
    @location(1) uv: vec2f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

struct Uniforms {
    resolution : vec2f,
};

@group(0) @binding(0) var waveformTexture: texture_2d<f32>;
@group(0) @binding(1) var waveformSampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var backgroundTexture: texture_2d<f32>;

@vertex
fn vs_fullscreen(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4f(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

@fragment
fn fs_mirror(input: VertexOutput) -> @location(0) vec4f {
    let uv = input.uv;
    let pixelOffset = 1.0 / uniforms.resolution.y;
    let mirroredUv = vec2f(uv.x, clamp(1.0 - uv.y - pixelOffset, 0.0, 1.0));

    // Compute the mask from the waveform
    let maskColor = max(
        textureSampleLevel(waveformTexture, waveformSampler, uv, 0.0),
        textureSampleLevel(waveformTexture, waveformSampler, mirroredUv, 0.0)
    );

    // Sample greyscale value from background texture (peak.bin)
    let peakSample = textureSampleLevel(backgroundTexture, waveformSampler, vec2f(uv.x, 0.5), 0.0).g;

    // Scale down for safety (optional)
    let greyValue = peakSample * maskColor.a * 0.45;

    // Return greyscale
    return vec4f(greyValue, greyValue, greyValue, 1.0);
}