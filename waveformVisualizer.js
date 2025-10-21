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

    /** @type {number | null} */
    #boost;
    /** @type {number | null} */
    #gamma;
    /** @type {number | null} */
    #smoothingFactor;
    #timeBuffer;
    #outputTexture;
    #outputTextureView;
    #computeBindGroupLayout;
    #computePipeline;
    #computeBindGroup;
    #blitPipeline;
    #blitBindGroup;


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
            this.#shaderWaveformCode = await WaveformVisualizer.loadShader('./shaderComputeWaveform.wgsl');

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
        this.#shaderWaveformCode = await WaveformVisualizer.loadShader('./shaderComputeWaveform.wgsl');
        this.#shaderWaveformModule = this.#gpuDevice.createShaderModule({
            code: this.#shaderWaveformCode,
        });

        // Create uniform buffer (for time)
        this.#timeBuffer = this.#gpuDevice.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Create output texture (storage + sampled)
        this.#outputTexture = this.#gpuDevice.createTexture({
            size: [this.#canvas.width, this.#canvas.height],
            format: this.#canvasFormat,
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.COPY_SRC,
        });

        this.#outputTextureView = this.#outputTexture.createView();

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
                module: this.#shaderWaveformModule,
                entryPoint: "main",
            },
        });

        this.#computeBindGroup = this.#gpuDevice.createBindGroup({
            layout: this.#computeBindGroupLayout,
            entries: [
                {binding: 0, resource: this.#outputTextureView},
                {binding: 1, resource: {buffer: this.#timeBuffer}},
            ],
        });

        // ------------ 2nd pass ------------

        const blitShaderCode = await WaveformVisualizer.loadShader('./shaderBlit.wgsl');
        const blitShaderModule = this.#gpuDevice.createShaderModule({
            code: blitShaderCode,
        });

        // Create sampler for sampling compute texture
        const blitSampler = this.#gpuDevice.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });

        // Create bind group layout for texture + sampler
        const blitBindGroupLayout = this.#gpuDevice.createBindGroupLayout({
            entries: [
                {binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {}},
                {binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {}},
            ],
        });

        // Create pipeline layout
        const blitPipelineLayout = this.#gpuDevice.createPipelineLayout({
            bindGroupLayouts: [blitBindGroupLayout],
        });

        // Create render pipeline
        this.#blitPipeline = this.#gpuDevice.createRenderPipeline({
            layout: blitPipelineLayout,
            vertex: {
                module: blitShaderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: blitShaderModule,
                entryPoint: 'fs_main',
                targets: [
                    {format: this.#canvasFormat},
                ],
            },
            primitive: {topology: 'triangle-list'},
        });

        // Create bind group linking compute texture + sampler
        this.#blitBindGroup = this.#gpuDevice.createBindGroup({
            layout: blitBindGroupLayout,
            entries: [
                {binding: 0, resource: this.#outputTextureView},
                {binding: 1, resource: blitSampler},
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
            if (!this.#gpuDevice || !this.#computePipeline || !this.#blitPipeline || !this.#outputTexture) {
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

            // --- Render pass (blit to canvas) ---
            const renderPass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: this.#context.getCurrentTexture().createView(),
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: {r: 0, g: 0, b: 0, a: 1},
                }]
            });
            renderPass.setPipeline(this.#blitPipeline);
            renderPass.setBindGroup(0, this.#blitBindGroup);
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
