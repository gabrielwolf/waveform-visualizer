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

    /** @type {number | null} On HiDpi monitors 2 or 3, like Apple Retina displays */
    #devicePixelRatio;
    /** @type {HTMLCanvasElement} Canvas element for waveform rendering */
    #canvas;
    /** @type {GPUTextureFormat | null} Canvas texture format */
    #canvasFormat;
    /** @type {PredefinedColorSpace | null} Canvas color space */
    #canvasColorSpace;
    /** @type {GPUTextureFormat | null} Internal format for GPU textures */
    #internalFormat;
    /** @type {GPUCanvasContext | null} WebGPU canvas context */
    #context;
    /** @type {GPUDevice | null} WebGPU device used for compute & render */
    #gpuDevice;

    #metaData;

    /** @type {string | null} WGSL shader code for compute waveform */
    #shaderComputeWaveformCode;
    /** @type {GPUShaderModule | null} Compiled compute shader module */
    #shaderComputeWaveformModule;
    /** @type {string | null} WGSL shader code for displaying output */
    #shaderDisplayCode;
    /** @type {GPUShaderModule | null} Compiled display shader module */
    #shaderDisplayModule;

    /** @type {GPUBuffer | null} Uniform buffer for parameters (boost, offset, etc.) */
    #paramsBuffer;
    /** @type {GPUBuffer | null} Channel height and offset buffer */
    #channelLayoutBuffer;
    /** @type {GPUBuffer | null} Uniform buffer for firstChannelMax */
    #firstChannelMaximumBuffer;
    /** @type {GPUTexture | null} MSAA texture for display */
    #displayTextureMSAA;

    /** @type {number | null} Peak of first channel */
    #firstChannelPeak;
    /** @type {number | null} Visualization intensity */
    #boost;
    /** @type {number | null} Brightness offset */
    #offset;
    /** @type {number | null} Ambisonics channel count */
    #channelCount;

    /** @type {GPUComputePipeline | null} Compute pipeline for waveform generation */
    #computePipeline;
    /** @type {GPUBindGroupLayout | null} Compute shader bind group layout */
    #computeBindGroupLayout;
    /** @type {GPUBindGroup | null} Bind group linking buffers & textures for compute */
    #computeBindGroup;
    /** @type {GPUBuffer | null} Two dimensional array containing greyscale values on x and alpha on y as buffer */
    #computeOutputBuffer;

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
    /** @type {GPUBuffer | null} */
    #waveformDataBuffer;
    /** @type {Float32Array | null} Normalized background or peak data */
    #backgroundData;
    /** @type {GPUBuffer | null} Normalized background or peak data */
    #backgroundDataBuffer;
    /** @type {array<number> | null} Download progress for the flac packages */
    #groupMask;

    static async loadShader(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load shader: ${url}`);
        return await response.text();
    }

    static async loadBinaryFile(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load binary file: ${url}`);
        const arrayBuffer = await response.arrayBuffer();
        return new Float32Array(arrayBuffer);
    }

    static async loadMeta(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load meta data: ${url}`);
        return await response.json();
    }


    static computeChannelLayout(height, channelCount) {
        const baseHeight = Math.floor(height / channelCount);
        const remainder = height % channelCount;
        const extraForChannel = new Array(channelCount).fill(0);
        let remaining = remainder;
        let offset = 0;
        while (remaining > 0) {
            const leftIndex = Math.floor(channelCount / 2) - 1 - offset;
            const rightIndex = Math.floor(channelCount / 2) + offset;
            if (leftIndex >= 0 && remaining > 0) {
                extraForChannel[leftIndex] = 1;
                remaining--;
            }
            if (remaining === 0) break;
            if (rightIndex < channelCount && remaining > 0) {
                extraForChannel[rightIndex] = 1;
                remaining--;
            }
            offset++;
        }

        const offsets = new Array(channelCount).fill(0);
        const heights = new Array(channelCount).fill(0);
        let acc = 0;
        for (let i = 0; i < channelCount; i++) {
            const h = baseHeight + extraForChannel[i];
            offsets[i] = acc;
            heights[i] = h;
            acc += h;
        }
        return {offsets, heights};
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
        this.#devicePixelRatio = window.devicePixelRatio || 1;
        this.#canvas = /** @type {HTMLCanvasElement} */ (canvas);
        this.#canvas.width = Math.max(1, Math.floor(this.#canvas.clientWidth * this.#devicePixelRatio));
        this.#canvas.height = Math.max(1, Math.floor(this.#canvas.clientHeight * this.#devicePixelRatio));
        this.#canvasFormat = null;
        this.#canvasColorSpace = null;
        this.#internalFormat = null;
        this.#shaderComputeWaveformCode = null;
        this.#shaderComputeWaveformModule = null;
        this.#shaderDisplayCode = null;
        this.#shaderDisplayModule = null;
        this.#paramsBuffer = null;
        this.#channelLayoutBuffer = null;
        this.#displayTextureMSAA = null;
        this.#boost = 1.5;
        this.#offset = 0.1;
        this.#channelCount = null;
        this.#firstChannelMaximumBuffer = null;
        this.#computePipeline = null;
        this.#computeBindGroupLayout = null;
        this.#computeBindGroup = null;
        this.#displayPipelineLayout = null;
        this.#displayPipeline = null;
        this.#displayBindGroupLayout = null;
        this.#displayBindGroup = null;
        this.#displaySampler = null;
        this.#waveformData = null;
        this.#waveformDataBuffer = null;
        this.#backgroundData = null;
        this.#backgroundDataBuffer = null;
        this.#groupMask = [0.25, 0.75, 0.5];

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

            this.#setupPipeline();
            this.#resizeTextures();
            this.#setupBindGroups();
        });

        if (!this.#gpuDevice) {
            console.warn("WebGPU device not provided");
            return;
        }

        (async () => {
            this.#shaderComputeWaveformCode = await WaveformVisualizer.loadShader('../wgsl/shaderComputeWaveform.wgsl');

            // Layout of the json: {input_file: 'input.caf', channel_count: 4, sample_count: 11811656, image_width: 4096}
            this.#metaData = await WaveformVisualizer.loadMeta('../binaries/waveform.json');
            this.#channelCount = this.#metaData.channel_count;
            this.#waveformData = await WaveformVisualizer.loadBinaryFile('../binaries/mean.bin');
            this.#backgroundData = await WaveformVisualizer.loadBinaryFile('./binaries/peak.bin');

            await this.#setupPipeline();
            this.updateParamsBuffer();

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

    set groupMask({index, value}) {
        if (isNaN(index) || index < 0 || index > 2) {
            throw new Error(`Index must be 0, 1, or 2`);
        }
        if (isNaN(value) || value < 0 || value > 1) {
            throw new Error(`Value must be between 0 and 1`);
        }
        this.#groupMask[index] = value;
        this.updateParamsBuffer();
    }

    /**
     * (Re)creates textures and updates bind groups on resize or format change.
     */
    #resizeTextures() {
        // Guard against race condition with render loop
        if (!this.#gpuDevice) return;

        this.#canvas.width = Math.max(1, Math.floor(this.#canvas.clientWidth * this.#devicePixelRatio));
        this.#canvas.height = Math.max(1, Math.floor(this.#canvas.clientHeight * this.#devicePixelRatio));

        this.#displayTextureMSAA?.destroy();
        this.#computeOutputBuffer?.destroy();
        this.#channelLayoutBuffer?.destroy();

        this.#displayTextureMSAA = this.#gpuDevice.createTexture({
            size: [this.#canvas.width, this.#canvas.height],
            sampleCount: 1,
            format: this.#canvasFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        const outputElementCount = this.#canvas.width * this.#canvas.height;
        const outputBufferSize = outputElementCount * 2 * 4; // vec2<f32> per pixel

        this.#computeOutputBuffer = this.#gpuDevice.createBuffer({
            size: outputBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        const paramsData = new Float32Array([
            this.#firstChannelPeak,
            this.#boost,
            this.#offset,
            this.#channelCount,
            this.#canvas.width,
            this.#canvas.height,
            this.#groupMask[0],
            this.#groupMask[1],
            this.#groupMask[2],
            0.0 // padding
        ]);
        this.#gpuDevice.queue.writeBuffer(this.#paramsBuffer, 0, paramsData);

        const {offsets, heights} = WaveformVisualizer.computeChannelLayout(this.#canvas.height, this.#channelCount);
        const vec4sPerArray = 16;
        const layoutData = new Float32Array(vec4sPerArray * 4 * 2); // offsets + heights
        for (let i = 0; i < this.#channelCount; i++) {
            const group = Math.floor(i / 4); // vec4 index
            const sub = i % 4;               // component within vec4
            const offsetBase = group * 4;    // vec4 index * 4 floats
            const heightBase = vec4sPerArray * 4 + group * 4;

            layoutData[offsetBase + sub] = offsets[i];  // x/y/z/w depends on sub
            layoutData[heightBase + sub] = heights[i];
        }

        this.#channelLayoutBuffer = this.#gpuDevice.createBuffer({
            size: layoutData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.#gpuDevice.queue.writeBuffer(this.#channelLayoutBuffer, 0, layoutData);

        this.#setupBindGroups();
    }

    /**
     * Sets up compute and rendering pipeline
     */
    async #setupPipeline() {
        this.#shaderComputeWaveformCode = await WaveformVisualizer.loadShader('../wgsl/shaderComputeWaveform.wgsl');
        this.#shaderComputeWaveformModule = this.#gpuDevice.createShaderModule({
            code: this.#shaderComputeWaveformCode,
        });

        this.#firstChannelPeak = Math.max(...this.#waveformData.filter((_, i) => i % this.#channelCount === 0));

        const paramsData = new Float32Array([
            this.#firstChannelPeak,
            this.#boost,
            this.#offset,
            this.#channelCount,
            this.#canvas.width,
            this.#canvas.height,
            this.#groupMask[0],
            this.#groupMask[1],
            this.#groupMask[2],
            0.0 // padding
        ]);

        // Create a uniform buffer for Params
        this.#paramsBuffer = this.#gpuDevice.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.#gpuDevice.queue.writeBuffer(this.#paramsBuffer, 0, paramsData);

        // Uniform buffer for firstChannelMax (1 float)
        const firstChannelMax = Math.max(...this.#waveformData.filter((_, i) => i % 16 === 0)); // channel 0
        const firstChannelMaxValue = new Float32Array([firstChannelMax]);
        this.#firstChannelMaximumBuffer = this.#gpuDevice.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.#gpuDevice.queue.writeBuffer(this.#firstChannelMaximumBuffer, 0, firstChannelMaxValue);

        const {offsets, heights} = WaveformVisualizer.computeChannelLayout(this.#canvas.height, this.#channelCount);
        const vec4sPerArray = 16;
        const layoutData = new Float32Array(vec4sPerArray * 4 * 2); // offsets + heights
        for (let i = 0; i < this.#channelCount; i++) {
            const group = Math.floor(i / 4); // vec4 index
            const sub = i % 4;               // component within vec4
            const offsetBase = group * 4;    // vec4 index * 4 floats
            const heightBase = vec4sPerArray * 4 + group * 4;

            layoutData[offsetBase + sub] = offsets[i];  // x/y/z/w depends on sub
            layoutData[heightBase + sub] = heights[i];
        }

        this.#channelLayoutBuffer = this.#gpuDevice.createBuffer({
            size: layoutData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.#gpuDevice.queue.writeBuffer(this.#channelLayoutBuffer, 0, layoutData);

        this.#waveformDataBuffer = this.#gpuDevice.createBuffer({
            size: this.#waveformData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.#gpuDevice.queue.writeBuffer(this.#waveformDataBuffer, 0, this.#waveformData);

        this.#backgroundDataBuffer = this.#gpuDevice.createBuffer({
            size: this.#backgroundData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.#gpuDevice.queue.writeBuffer(this.#backgroundDataBuffer, 0, this.#backgroundData);

        // Create compute output buffer (brightness + alpha pairs)
        const outputElementCount = this.#canvas.width * this.#canvas.height;
        const outputBufferSize = outputElementCount * 2 * 4; // vec2<f32> = 8 bytes per pixel

        this.#computeOutputBuffer = this.#gpuDevice.createBuffer({
            size: outputBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Compute bind group layout (add binding 4 for firstChannelMax uniform)
        this.#computeBindGroupLayout = this.#gpuDevice.createBindGroupLayout({
            entries: [
                {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}}, // mean waveform
                {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}}, // peak background
                {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}}, // Params
                {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}}, // channelLayout
                {binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}}, // computeOutput buffer
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

        // ------------ 2nd pass ------------

        this.#shaderDisplayCode = await WaveformVisualizer.loadShader('../wgsl/shaderDisplay.wgsl');
        this.#shaderDisplayModule = this.#gpuDevice.createShaderModule({
            code: this.#shaderDisplayCode,
        });

        // Create sampler for sampling compute texture
        this.#displaySampler = this.#gpuDevice.createSampler({
            magFilter: 'nearest',
            minFilter: 'nearest',
        });

        // Create bind group layout for texture + sampler
        this.#displayBindGroupLayout = this.#gpuDevice.createBindGroupLayout({
            entries: [
                {binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {type: "storage"}}, // computeOutput
                {binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}}, // params
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
            multisample: {
                count: 1,
            }
        });
    }

    /**
     * Sets up bind groups for compute and display pipelines.
     */
    #setupBindGroups() {
        if (
            !this.#gpuDevice ||
            !this.#computeBindGroupLayout ||
            !this.#displayBindGroupLayout ||
            !this.#displaySampler
        ) {
            return;
        }

        // Compute bind group (add binding 4 for firstChannelMax uniform buffer)
        this.#computeBindGroup = this.#gpuDevice.createBindGroup({
            layout: this.#computeBindGroupLayout,
            entries: [
                {binding: 0, resource: {buffer: this.#waveformDataBuffer}},
                {binding: 1, resource: {buffer: this.#backgroundDataBuffer}},
                {binding: 2, resource: {buffer: this.#paramsBuffer}},
                {binding: 3, resource: {buffer: this.#channelLayoutBuffer}},
                {binding: 4, resource: {buffer: this.#computeOutputBuffer}},
            ],
        });

        // Display bind group
        this.#displayBindGroup = this.#gpuDevice.createBindGroup({
            layout: this.#displayBindGroupLayout,
            entries: [
                {binding: 0, resource: {buffer: this.#computeOutputBuffer}},
                {binding: 1, resource: {buffer: this.#paramsBuffer}}
            ],
        });
    }

    /**
     * Main render loop (stub for compute-based approach).
     */
    #renderLoop() {
        if (!this.#displayTextureMSAA) {
            this.#resizeTextures();
        }

        const frame = () => {
            if (!this.#gpuDevice || !this.#computePipeline || !this.#computeBindGroup || !this.#displayPipeline
                || !this.#displayBindGroup || !this.#displayTextureMSAA || !this.#paramsBuffer) {
                requestAnimationFrame(frame);
                return;
            }

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
    updateParamsBuffer() {
        if (!this.#gpuDevice || !this.#paramsBuffer) return;
        const paramsData = new Float32Array(
            [
                this.#firstChannelPeak,
                this.#boost,
                this.#offset,
                this.#channelCount,
                this.#canvas.width,
                this.#canvas.height,
                this.#groupMask[0],
                this.#groupMask[1],
                this.#groupMask[2],
                0.0 // padding
            ]);
        this.#gpuDevice.queue.writeBuffer(this.#paramsBuffer, 0, paramsData.buffer, paramsData.byteOffset, paramsData.byteLength);
    }
}

export default WaveformVisualizer;
