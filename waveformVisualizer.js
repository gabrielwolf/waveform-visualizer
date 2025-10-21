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

    /** @type {HTMLCanvasElement} Canvas element for waveform rendering */
    #canvas;
    /** @type {GPUTextureFormat | null} Canvas texture format */
    #canvasFormat;
    /** @type {PredefinedColorSpace | null} Canvas color space */
    #canvasColorSpace;
    /** @type {GPUTextureFormat | null} Internal format for GPU textures */
    #internalFormat;
    /** @type {GPUDevice | null} WebGPU device used for compute & render */
    #gpuDevice;
    /** @type {GPUCanvasContext | null} WebGPU canvas context */
    #context;

    /** @type {string | null} WGSL shader code for compute waveform */
    #shaderComputeWaveformCode;
    /** @type {GPUShaderModule | null} Compiled compute shader module */
    #shaderComputeWaveformModule;
    /** @type {string | null} WGSL shader code for displaying output */
    #shaderDisplayCode;
    /** @type {GPUShaderModule | null} Compiled display shader module */
    #shaderDisplayModule;



    /** @type {GPUBuffer | null} Uniform buffer for parameters (time, boost, gamma, etc.) */
    #uniformBuffer;
    /** @type {GPUTexture | null} MSAA texture for display */
    #displayTextureMSAA;

    /** @type {number | null} Visualization intensity */
    #boost;
    /** @type {number | null} Gamma correction */
    #gamma;
    /** @type {number | null} Smoothing factor for waveform */
    #smoothingFactor;

    /** @type {number | null} Time parameter for compute shader */
    #timeBuffer;
    /** @type {GPUTexture | null} Compute texture for compute shader results */
    #computeTexture;
    /** @type {GPUTextureView | null} View of the output texture */
    #computeTextureView;
    /** @type {GPUComputePipeline | null} Compute pipeline for waveform generation */
    #computePipeline;
    /** @type {GPUBindGroupLayout | null} Compute shader bind group layout */
    #computeBindGroupLayout;
    /** @type {GPUBindGroup | null} Bind group linking buffers & textures for compute */
    #computeBindGroup;

    /** @type {GPUPipelineLayout | null} Pipeline layout for display */
    #displayPipelineLayout;
    /** @type {GPURenderPipeline | null} Render pipeline for display pass */
    #displayPipeline;
    /** @type {GPUBindGroupLayout | null} Bind group layout for display pipeline */
    #displayBindGroupLayout;
    /** @type {GPUBindGroup | null} Bind group for display pipeline */
    #displayBindGroup;
    /** @type {GPUSampler | null} Sampler used for display output texture */
    #displaySampler;

    /** @type {Float32Array | null} Normalized waveform data from binary file */
    #waveformData;
    /** @type {Float32Array | null} Normalized background or peak data */
    #backgroundData;

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

        this.#shaderComputeWaveformCode = null;

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
            this.#shaderComputeWaveformCode = await WaveformVisualizer.loadShader('./shaderComputeWaveform.wgsl');

            this.#waveformData = await WaveformVisualizer.loadBinaryFile('./mean.bin');
            this.#backgroundData = await WaveformVisualizer.loadBinaryFile('./peak.bin');

            await this.#setupPipeline();
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
     * (Re)creates textures and updates bind groups on resize or format change.
     */
    #resizeTextures() {
        // Guard against race condition with render loop
        if (!this.#gpuDevice) return;

        // --- Resize canvas to match client size ---
        this.#canvas.width = Math.max(1, this.#canvas.clientWidth);
        this.#canvas.height = Math.max(1, this.#canvas.clientHeight);

        // Destroy old textures if present
        // this.#waveformTexture?.destroy();


        // Placeholder for future compute texture setup
        // (In new compute-based approach, setup output textures here if needed)
        // TODO: Create/resize output textures for compute pass here if needed

        // Recreate bind groups if needed
        this.#setupBindGroups();
    }

    /**
     * Sets up compute and rendering pipeline
     */
    async #setupPipeline() {
        this.#shaderComputeWaveformCode = await WaveformVisualizer.loadShader('./shaderComputeWaveform.wgsl');
        this.#shaderComputeWaveformModule = this.#gpuDevice.createShaderModule({
            code: this.#shaderComputeWaveformCode,
        });

        // Create uniform buffer (for time)
        this.#timeBuffer = this.#gpuDevice.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Create output texture (storage + sampled)
        this.#computeTexture = this.#gpuDevice.createTexture({
            size: [this.#canvas.width, this.#canvas.height],
            format: this.#canvasFormat,
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.COPY_SRC,
        });

        this.#computeTextureView = this.#computeTexture.createView();

        // Compute bind group layout
        this.#computeBindGroupLayout = this.#gpuDevice.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {access: "write-only", format: this.#canvasFormat}
                },
                {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
            ]
        });

        // Compute pipeline
        this.#computePipeline = this.#gpuDevice.createComputePipeline({
            layout: this.#gpuDevice.createPipelineLayout({
                bindGroupLayouts: [this.#computeBindGroupLayout]
            }),
            compute: {
                module: this.#shaderComputeWaveformModule,
                entryPoint: "main",
            },
        });

        this.#computeBindGroup = this.#gpuDevice.createBindGroup({
            layout: this.#computeBindGroupLayout,
            entries: [
                {binding: 0, resource: this.#computeTextureView},
                {binding: 1, resource: {buffer: this.#timeBuffer}},
            ],
        });

        // ------------ 2nd pass ------------

        this.#shaderDisplayCode = await WaveformVisualizer.loadShader('./shaderDisplay.wgsl');
        this.#shaderDisplayModule = this.#gpuDevice.createShaderModule({
            code: this.#shaderDisplayCode,
        });

        // Create sampler for sampling compute texture
        this.#displaySampler = this.#gpuDevice.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });

        // Create bind group layout for texture + sampler
        this.#displayBindGroupLayout = this.#gpuDevice.createBindGroupLayout({
            entries: [
                {binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {}},
                {binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {}},
            ],
        });

        // Create pipeline layout
        this.#displayPipelineLayout = this.#gpuDevice.createPipelineLayout({
            bindGroupLayouts: [this.#displayBindGroupLayout],
        });

        // Create render pipeline
        this.#displayPipeline = this.#gpuDevice.createRenderPipeline({
            layout: this.#displayPipelineLayout,
            vertex: {
                module: this.#shaderDisplayModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: this.#shaderDisplayModule,
                entryPoint: 'fs_main',
                targets: [
                    {format: this.#canvasFormat},
                ],
            },
            primitive: {topology: 'triangle-list'},
        });

        // Create bind group linking compute texture + sampler
        this.#displayBindGroup = this.#gpuDevice.createBindGroup({
            layout: this.#displayBindGroupLayout,
            entries: [
                {binding: 0, resource: this.#computeTextureView},
                {binding: 1, resource: this.#displaySampler},
            ],
        });
    }

    /**
     * Sets up bind groups (stub for compute-based approach).
     */
    #setupBindGroups() {
        // In compute-based approach, setup compute and render bind groups here.
        // This is a stub placeholder.
    }

    /**
     * Main render loop (stub for compute-based approach).
     */
    #renderLoop() {
        let time = 0;
        const frame = () => {
            if (!this.#gpuDevice || !this.#computePipeline || !this.#displayPipeline || !this.#computeTexture) {
                requestAnimationFrame(frame);
                return;
            }

            time += 0.016;
            this.#gpuDevice.queue.writeBuffer(this.#timeBuffer, 0, new Float32Array([time]));

            const encoder = this.#gpuDevice.createCommandEncoder();

            // --- Compute pass ---
            const computePass = encoder.beginComputePass();
            computePass.setPipeline(this.#computePipeline);
            computePass.setBindGroup(0, this.#computeBindGroup);
            const workgroupsX = Math.ceil(this.#canvas.width / 8);
            const workgroupsY = Math.ceil(this.#canvas.height / 8);
            computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
            computePass.end();

            // --- Render pass (display to canvas) ---
            const renderPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: this.#context.getCurrentTexture().createView(),
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: {r: 0, g: 0, b: 0, a: 1},
                }]
            });
            renderPass.setPipeline(this.#displayPipeline);
            renderPass.setBindGroup(0, this.#displayBindGroup);
            renderPass.draw(6, 1, 0, 0);
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
