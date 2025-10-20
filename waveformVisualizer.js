import GpuContextManager from "./gpuContextManager.js";

/**
 * Singleton class for rendering a real-time WaveformVisualizer
 *
 * Usage:
 *    import WaveformVisualizer from './waveformVisualizer.js';
 *    const waveformVisualizer = WaveformVisualizer.init(document.querySelector('canvas'), gpuDevice);
 */

/** @typedef {import('webgpu-types').GPUBufferUsage} GPUBufferUsage */

// import shaderWaveformVisualizerUrl from '@/waveformVisualizer.wgsl?raw';

class WaveformVisualizer {
    static #instance = null;

    /** @type {HTMLCanvasElement} */
    #canvas;
    /** @type {GPUTextureFormat | null} */
    #canvasFormat;
    /** @type {PredefinedColorSpace | null} */
    #canvasColorSpace;
    /** @type {GPUTextureFormat | null} */
    #internalFormat;

    /** @type {GPUDevice | null} */
    #gpuDevice;
    /** @type {GPUCanvasContext | null} */
    #context;
    /** @type {string | null} */
    #shaderWaveformVisualizerCode;
    /** @type {GPUShaderModule | null} */
    #shaderWaveformVisualizerModule;

    /** @type {GPUBuffer | null} */
    #uniformBuffer;

    /** @type {GPUTexture | null} */
    #displayTextureMSAA;

    /** @type {GPURenderPipeline | null} */
    #pipeline;
    /** @type {GPUPipelineLayout | null} Pipeline layout combining uniform and energy bind group layouts. */
    #pipelineLayout;
    /** @type {GPUBindGroup | null} */
    #uniformBindGroup;
    /** @type {GPUBindGroupLayout | null} */
    #uniformBindGroupLayout;

    /** @type {number | null} */
    #boost;
    /** @type {number | null} */
    #gamma;
    /** @type {number | null} */
    #smoothingFactor;

    /** @type {GPUBindGroup | null} */
    #waveformBindGroup;
    /** @type {GPUBindGroupLayout | null} */
    #waveformBindGroupLayout;
    #waveformData;
    #waveformBuffer;

    static async loadShader(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load shader: ${url}`);
        return await response.text();
    }

    static async loadBinaryFile(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load binary file: ${url}`);
        const arrayBuffer = await response.arrayBuffer();
        const raw = new Uint16Array(arrayBuffer);
        const normalized = new Float32Array(raw.length);
        for (let i = 0; i < raw.length; i++) {
            normalized[i] = (raw[i] / 4095) * 1; // 12-bit -> [0,1]
        }
        return normalized;
    }

    /**
     * Initializes the singleton instance of WaveformVisualizer with the provided canvas and event bus.
     * If already initialized, returns the existing instance.
     *
     * @param {HTMLCanvasElement|null} canvasElement - The canvas element to render onto.
     * @param {GpuContextManager} gpuDevice - WebGPU device used for rendering.
     * @returns {WaveformVisualizer} The singleton instance.
     */
    static init(canvasElement = null, gpuDevice = null) {
        if (!WaveformVisualizer.#instance) {
            WaveformVisualizer.#instance = new WaveformVisualizer(canvasElement, gpuDevice);
            console.log("WaveformVisualizer initialized");
        }
        return WaveformVisualizer.#instance;
    }

    /**
     * Constructs the DummyHead renderer and initializes Babylon.js scene.
     * @param {Document | HTMLElement} [root=document] - DOM root to look for the canvas.
     * @throws {Error} If canvas element is not found.
     */
    constructor(root = document) {
        /** @type {HTMLCanvasElement} */
        const canvas = /** @type {HTMLCanvasElement} */ (root.querySelector('[data-role="track__waveform__canvas"]'));
        if (!canvas) {
            throw new Error("Waveform Visualizer: canvas not found.");
        }
        this.#canvas = /** @type {HTMLCanvasElement} */ (canvas);
        this.#canvas.width = Math.max(1, this.#canvas.clientWidth);
        this.#canvas.height = Math.max(1, this.#canvas.clientHeight);
        this.#canvasFormat = null;
        this.#canvasColorSpace = null;
        this.#shaderWaveformVisualizerCode = null;

        this.#boost = 1.5;
        this.#gamma = 0.45;
        this.#smoothingFactor = 0.20;

        const gpuContextManager = GpuContextManager.init();
        this.#context = gpuContextManager.configureCanvas(this.#canvas);
        this.#gpuDevice = gpuContextManager.context.device;
        this.#canvasFormat = gpuContextManager.context.format;
        this.#canvasColorSpace = gpuContextManager.context.colorSpace;
        this.#internalFormat = gpuContextManager.context.internalFormat;

        gpuContextManager.onReconfigure((context) => {
            console.log("SphericalHarmonicsAnimation: HDR/SDR change detected, rebuilding pipeline...");
            this.#canvasFormat = context.format;
            this.#canvasColorSpace = context.colorSpace;
            this.#internalFormat = context.internalFormat;

            this.#displayTextureMSAA?.destroy();

            this.#setupPipeline();
            this.#resizeTextures();
            this.#setupBindGroups();
        });

        if (!this.#gpuDevice) {
            console.warn("WebGPU device not provided");
            return;
        }

        (async () => {
            this.#shaderWaveformVisualizerCode = await WaveformVisualizer.loadShader('./shaderWaveformVisualizer.wgsl');
            this.#waveformData = await WaveformVisualizer.loadBinaryFile('./mean.bin');

            this.#setupPipeline();
            this.updateUniformBuffer();

            this.#resizeTextures();
            this.#setupBindGroups();
            this.#renderLoop();

            const resizeObserver = new ResizeObserver(entries => {
                for (const entry of entries) {
                    if (entry.target === this.#canvas) {
                        this.#resizeTextures();
                    }
                }
            });
            resizeObserver.observe(this.#canvas);
        })();
    }

    /**
     * Helper for creating a 2D GPU texture.
     * @param {number} width - Width of the texture in pixels.
     * @param {number} height - Height of the texture in pixels.
     * @param {GPUTextureFormat} format - Format of the texture (e.g., "rgba8unorm").
     * @param {GPUTextureUsageFlags} usage - Usage flags for the texture.
     * @returns {GPUTexture} The created GPU texture.
     */
    #createTexture(width, height, format, usage) {
        return this.#gpuDevice.createTexture({
            size: [width, height],
            format: format,
            usage: usage
        });
    }

    /**
     * (Re)creates textures and updates bind groups on resize or format change.
     * @private
     */
    #resizeTextures() {
        // Guard against race condition with render loop
        if (!this.#gpuDevice || !this.#pipeline) return;

        // --- Resize canvas to match client size ---
        this.#canvas.width = Math.max(1, this.#canvas.clientWidth);
        this.#canvas.height = Math.max(1, this.#canvas.clientHeight);

        // Destroy old textures if present
        this.#displayTextureMSAA?.destroy();

        // --- Create main render target texture ---
        // this was present here in the dumme head and space visualizer

        // --- Optionally: create multisampled texture for antialiasing ---
        this.#displayTextureMSAA = this.#gpuDevice.createTexture({
            size: [this.#canvas.width, this.#canvas.height],
            sampleCount: 1, // make it 4 for multisampling
            format: this.#canvasFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        // Recreate bind groups if needed
        this.#setupBindGroups();
    }

    /**
     * Creates shader modules, buffers, and initializes the render pipeline.
     * @private
     */
    #setupPipeline() {
        this.#shaderWaveformVisualizerModule = this.#gpuDevice.createShaderModule({
            code: this.#shaderWaveformVisualizerCode
        });
        const shaderWaveformVisualizerModule = this.#shaderWaveformVisualizerModule;

        const uniformData = new Float32Array([this.#boost, this.#gamma, 0, 0]);
        this.#uniformBuffer = this.#gpuDevice.createBuffer({
            size: uniformData.byteLength, // 16 bytes (4 floats)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.#gpuDevice.queue.writeBuffer(this.#uniformBuffer, 0, uniformData.buffer, uniformData.byteOffset, uniformData.byteLength);

        // Create uniform bind group layout for uniform buffer and model matrix
        this.#uniformBindGroupLayout = this.#gpuDevice.createBindGroupLayout({
            entries: [
                {binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}},
            ],
        });

        // Create pipeline layout using uniform and energy bind group layouts (waveformBindGroupLayout added later)
        // We'll patch pipeline layout after waveformBindGroupLayout is created in #setupBindGroups
        this.#pipelineLayout = null; // Will be set after bind group layouts created

        // vertex buffer: float32x3
        this.#shaderWaveformVisualizerModule = shaderWaveformVisualizerModule;
    }

    /**
     * Sets up bind groups linking uniform buffers/model matrix and energy/LUT data for lighting.
     */
    #setupBindGroups() {
        if (!this.#uniformBuffer || !this.#uniformBindGroupLayout) return;

        // Waveform bind group layout: energies, directions, LUT, sampler, params
        this.#waveformBindGroupLayout = this.#gpuDevice.createBindGroupLayout({
            entries: [],
        });
        this.#waveformBindGroup = this.#gpuDevice.createBindGroup({
            layout: this.#waveformBindGroupLayout,
            entries: [],
        });

        // Uniform bind group
        this.#uniformBindGroup = this.#gpuDevice.createBindGroup({
            layout: this.#uniformBindGroupLayout,
            entries: [
                {binding: 0, resource: {buffer: this.#uniformBuffer}},
            ],
        });

        if (this.#waveformData) {
            const sampleCount = this.#waveformData.length;
            const vertexData = new Float32Array(sampleCount * 2);
            for (let i = 0; i < sampleCount; i++) {
                vertexData[i * 2] = (i / (sampleCount - 1)) * 4 - 1; // x: -1 → 1
                vertexData[i * 2 + 1] = this.#waveformData[i];       // y: audio sample
            }
            this.#waveformBuffer = this.#gpuDevice.createBuffer({
                size: vertexData.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            new Float32Array(this.#waveformBuffer.getMappedRange()).set(vertexData);
            this.#waveformBuffer.unmap();
        }

        // Now create pipeline layout and pipeline if not already created
        if (!this.#pipeline) {
            this.#pipelineLayout = this.#gpuDevice.createPipelineLayout({
                bindGroupLayouts: [this.#uniformBindGroupLayout, this.#waveformBindGroupLayout],
            });
            // Main mesh pipeline
            this.#pipeline = this.#gpuDevice.createRenderPipeline({
                layout: this.#pipelineLayout,
                vertex: {
                    module: this.#shaderWaveformVisualizerModule,
                    entryPoint: "vs_head",
                    buffers: [
                        {
                            arrayStride: 8, // two floats per vertex (2 * 4 bytes)
                            attributes: [
                                {
                                    shaderLocation: 0, // matches @location(0)
                                    offset: 0,
                                    format: "float32x2"
                                }
                            ]
                        }
                    ]
                },
                fragment: {
                    module: this.#shaderWaveformVisualizerModule,
                    entryPoint: "fs_head",
                    targets: [{
                        format: this.#canvasFormat,
                        blend: {
                            color: {
                                srcFactor: 'src-alpha',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                            alpha: {
                                srcFactor: 'one',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            }
                        }
                    }],
                },
                primitive: {topology: "line-strip"},
                multisample: {count: 1}, // make it 4 for multisampling
            });
        }
    }

    /**
     * Main render loop.
     */
    #renderLoop() {
        const frame = () => {
            if (!this.#displayTextureMSAA) {
                requestAnimationFrame(frame);
                return;
            }

            // --- Begin encoding commands for this frame ---
            const encoder = this.#gpuDevice.createCommandEncoder();

            // --- Pass 1: Render Pass ---
            const renderPass = encoder.beginRenderPass({
                colorAttachments: [{
                    // For multisampling use these lines
                    // view: this.#displayTextureMSAA.createView(),
                    // resolveTarget: this.#context.getCurrentTexture().createView(),

                    view: this.#context.getCurrentTexture().createView(),
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: {r: 0, g: 0, b: 0, a: 0},
                }],
            });

            // Draw head mesh
            renderPass.setPipeline(this.#pipeline);
            renderPass.setBindGroup(0, this.#uniformBindGroup);
            renderPass.setBindGroup(1, this.#waveformBindGroup);
            renderPass.setVertexBuffer(0, this.#waveformBuffer);
            renderPass.draw(this.#waveformData.length / 2, 1, 0, 0);

            renderPass.end();

            this.#gpuDevice.queue.submit([encoder.finish()]);
            requestAnimationFrame(frame);
        };
        requestAnimationFrame(frame);
    }

    /**
     * Update uniform buffer for aspect ratio.
     */
    updateUniformBuffer() {
        if (!this.#gpuDevice || !this.#uniformBuffer) return;
        const uniformData = new Float32Array([this.#boost, this.#gamma, this.#smoothingFactor, 0]);
        this.#gpuDevice.queue.writeBuffer(this.#uniformBuffer, 0, uniformData.buffer, uniformData.byteOffset, uniformData.byteLength);
    }
}

export default WaveformVisualizer;
