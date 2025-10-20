import GpuContextManager from "./gpuContextManager.js";

/**
 * Singleton class for rendering a real-time WaveformVisualizer
 *
 * Usage:
 *    import WaveformVisualizer from './waveformVisualizer.js';
 *    const waveformVisualizer = WaveformVisualizer.init(document.querySelector('canvas'), gpuDevice);
 */

/** @typedef {import('webgpu-types').GPUBufferUsage} GPUBufferUsage */

// import shaderWaveformUrl from '@/waveformVisualizer.wgsl?raw';

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
    #shaderWaveformCode;
    /** @type {GPUShaderModule | null} */
    #shaderWaveformModule;

    /** @type {GPUBuffer | null} */
    #uniformBuffer;

    /** @type {GPUTexture | null} */
    #displayTextureMSAA;
    /** @type {GPUTexture | null} */
    #waveformTexture;
    /** @type {GPUTextureView | null} */
    #waveformTextureView;

    /** @type {GPURenderPipeline | null} */
    #pipeline;
    /** @type {GPURenderPipeline | null} */
    #mirrorPipeline;
    /** @type {GPUShaderModule | null} */
    #mirrorShaderModule;
    /** @type {GPUBuffer | null} */
    #fullscreenQuadBuffer;
    /** @type {GPUBindGroupLayout | null} */
    #mirrorBindGroupLayout;
    /** @type {GPUBindGroup | null} */
    #mirrorBindGroup;
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

    /** @type {Float32Array} */
    #waveformData;
    /** @type {GPUBuffer} */
    #waveformBuffer;
    /** @type {string | null} */
    #shaderMirrorCode;
    /** @type {GPUShaderModule | null} */
    #shaderMirrorModule;


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
        this.#shaderWaveformCode = null;

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
            this.#shaderWaveformCode = await WaveformVisualizer.loadShader('./shaderWaveform.wgsl');
            this.#shaderMirrorCode = await WaveformVisualizer.loadShader('./shaderMirror.wgsl');
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
        this.#waveformTexture?.destroy();

        // --- Create main render target texture ---
        // this was present here in the dumme head and space visualizer

        // --- Optionally: create multisampled texture for antialiasing ---
        this.#displayTextureMSAA = this.#gpuDevice.createTexture({
            size: [this.#canvas.width, this.#canvas.height],
            sampleCount: 1, // make it 4 for multisampling
            format: this.#canvasFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        // Create waveform texture for first pass (offscreen)
        this.#waveformTexture = this.#gpuDevice.createTexture({
            size: [this.#canvas.width, this.#canvas.height],
            format: this.#canvasFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.#waveformTextureView = this.#waveformTexture.createView();

        // Recreate bind groups if needed
        this.#setupBindGroups();
        this.#setupMirrorBindGroup();
    }

    /**
     * Creates shader modules, buffers, and initializes the render pipeline.
     * @private
     */
    #setupPipeline() {
        this.#shaderWaveformModule = this.#gpuDevice.createShaderModule({
            code: this.#shaderWaveformCode
        });

        this.#shaderMirrorModule = this.#gpuDevice.createShaderModule({
            code: this.#shaderMirrorCode
        });

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

        // Create pipeline layout using uniform bind group layouts. We'll patch pipeline in #setupBindGroups
        this.#pipelineLayout = null;


        // --- Setup mirror pipeline and fullscreen quad ---
        // Create shader module for mirror pass

        this.#mirrorShaderModule = this.#gpuDevice.createShaderModule({
            code: this.#shaderMirrorCode,
        });

        // Fullscreen quad vertex buffer: 2D position and UV
        // 2 triangles (triangle strip): positions [-1,1] and [0,0]-[1,1]
        const quadVertices = new Float32Array([
            //  position   uv
            -1, -1, 0, 0,
            1, -1, 1, 0,
            -1, 1, 0, 1,
            1, 1, 1, 1,
        ]);
        this.#fullscreenQuadBuffer = this.#gpuDevice.createBuffer({
            size: quadVertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.#fullscreenQuadBuffer.getMappedRange()).set(quadVertices);
        this.#fullscreenQuadBuffer.unmap();

        // Mirror bind group layout
        this.#mirrorBindGroupLayout = this.#gpuDevice.createBindGroupLayout({
            entries: [
                {binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {}},
                {binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {}},
            ]
        });

        // Mirror pipeline layout
        const mirrorPipelineLayout = this.#gpuDevice.createPipelineLayout({
            bindGroupLayouts: [this.#mirrorBindGroupLayout]
        });
        this.#mirrorPipeline = this.#gpuDevice.createRenderPipeline({
            layout: mirrorPipelineLayout,
            vertex: {
                module: this.#mirrorShaderModule,
                entryPoint: "vs_fullscreen",
                buffers: [
                    {
                        arrayStride: 16,
                        attributes: [
                            {shaderLocation: 0, offset: 0, format: "float32x2"}, // position
                            {shaderLocation: 1, offset: 8, format: "float32x2"}, // uv
                        ]
                    }
                ]
            },
            fragment: {
                module: this.#mirrorShaderModule,
                entryPoint: "fs_mirror",
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
                }]
            },
            primitive: {topology: "triangle-strip"},
            multisample: {count: 1}
        });
    }

    #setupMirrorBindGroup() {
        if (!this.#waveformTextureView || !this.#mirrorBindGroupLayout) return;
        // Create a sampler for use in the mirror pass
        const sampler = this.#gpuDevice.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });
        this.#mirrorBindGroup = this.#gpuDevice.createBindGroup({
            layout: this.#mirrorBindGroupLayout,
            entries: [
                {binding: 0, resource: this.#waveformTextureView},
                {binding: 1, resource: sampler}
            ]
        });
    }

    /**
     * Sets up bind groups linking uniform buffers/model matrix and energy/LUT data for lighting.
     */
    #setupBindGroups() {
        if (!this.#uniformBuffer || !this.#uniformBindGroupLayout) return;

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
                bindGroupLayouts: [this.#uniformBindGroupLayout]
            });
            // Main mesh pipeline
            this.#pipeline = this.#gpuDevice.createRenderPipeline({
                layout: this.#pipelineLayout,
                vertex: {
                    module: this.#shaderWaveformModule,
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
                    module: this.#shaderWaveformModule,
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
     * Pass 1: Draw waveform to offscreen texture.
     * Pass 2: Draw mirrored quad using that texture to swap chain.
     */
    #renderLoop() {
        const frame = () => {
            if (!this.#waveformTexture || !this.#mirrorPipeline || !this.#fullscreenQuadBuffer || !this.#mirrorBindGroup || !this.#waveformBuffer) {
                requestAnimationFrame(frame);
                return;
            }

            // --- Begin encoding commands for this frame ---
            const encoder = this.#gpuDevice.createCommandEncoder();

            // --- Pass 1: Render waveform into offscreen texture ---
            const pass1 = encoder.beginRenderPass({
                colorAttachments: [{
                    view: this.#waveformTextureView,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: {r: 0, g: 0, b: 0, a: 0},
                }],
            });
            pass1.setPipeline(this.#pipeline);
            pass1.setBindGroup(0, this.#uniformBindGroup);
            pass1.setVertexBuffer(0, this.#waveformBuffer);
            pass1.draw(this.#waveformData.length / 2, 1, 0, 0);
            pass1.end();

            // --- Pass 2: Draw mirrored quad to swap chain ---
            const pass2 = encoder.beginRenderPass({
                colorAttachments: [{
                    view: this.#context.getCurrentTexture().createView(),
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: {r: 0, g: 0, b: 0, a: 0},
                }],
            });
            pass2.setPipeline(this.#mirrorPipeline);
            pass2.setBindGroup(0, this.#mirrorBindGroup);
            pass2.setVertexBuffer(0, this.#fullscreenQuadBuffer);
            pass2.draw(4, 1, 0, 0);
            pass2.end();

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
