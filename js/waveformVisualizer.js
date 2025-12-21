/**
 * Class for rendering a real-time WaveformVisualizer using WebGPU
 * Handles:
 * - Loading waveform binaries
 * - Setting up compute and display pipelines
 * - Rendering multiple audio channels with configurable intensity and offsets
 *
 * Usage:
 *    import WaveformVisualizer from './waveformVisualizer.js';
 *    const waveformVisualizer = new WaveformVisualizer(uuid, channels);
 */

// import GpuContextManager from "./gpuContextManager.js";

/** @typedef {import('webgpu-types').GPUTextureFormat} GPUTextureFormat */
/** @typedef {import('webgpu-types').GPUCanvasContext} GPUCanvasContext */
/** @typedef {import('webgpu-types').GPUDevice} GPUDevice */
/** @typedef {import('webgpu-types').GPUShaderModule} GPUShaderModule */
/** @typedef {import('webgpu-types').GPUBuffer} GPUBuffer */
/** @typedef {import('webgpu-types').GPUBufferUsage} GPUBufferUsage */
/** @typedef {import('webgpu-types').GPUComputePipeline} GPUComputePipeline */
/** @typedef {import('webgpu-types').GPUBindGroupLayout} GPUBindGroupLayout */
/** @typedef {import('webgpu-types').GPUBindGroup} GPUBindGroup */
/** @typedef {import('webgpu-types').GPUPipelineLayout} GPUPipelineLayout */
/** @typedef {import('webgpu-types').GPURenderPipeline} GPURenderPipeline */
/** @typedef {import('webgpu-types').GPUSampler} GPUSampler */

/**
 * @typedef {Object} FrameOrchestratorLike
 * @property {(participant: {id:string, preferredHz:number, wantsFrame:(now:number)=>boolean, encode:(encoder:GPUCommandEncoder, now:number)=>void})=>void} register
 * @property {(id: string)=>void} [unregister]
 */

// do it like this if using ViteJS (replace the static variable with import)
// import shaderComputeWaveformUrl from '@/shaderComputeWaveform.wgsl?raw';

class WaveformVisualizer {
    /* defaults for standalone version */
    static #shaderComputeWaveformCodeUrl = '../wgsl/shaderComputeWaveform.wgsl';
    static #shaderDisplayCodeUrl = '../wgsl/shaderDisplay.wgsl';
    static #waveformDataMeanUrl = '../binaries/mean.bin';
    static #backgroundDataPeakUrl = '../binaries/peak.bin';

    /** @type {number | null} Visualization intensity, static because all visualizer instances should look the same */
    static BOOST = 1.2;
    /** @type {number | null} Brightness offset, static because all visualizer instances should look the same */
    static OFFSET = 0.15;

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
    /** @type {number | null} Peak of first channel */
    #firstChannelPeak;
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
    #maskGroup;
    /** @type {ResizeObserver} Track canvas size changes */
    #resizeObserver;
    /** @type {number | null} Request ID returned by `requestAnimationFrame()` used for canceling the render loop */
    #animationFrameId;
    /** @type {boolean} Flag that helps killing the render loop */
    #disposed;
    /** @type {boolean} */
    #needsRender;
    /** @type {boolean} */
    #isRegistered;
    /** @type {FrameOrchestratorLike | null} */
    #orchestrator;
    /** @type {string | null} */
    #orchestratorParticipantId;
    /** @type {number | null} */
    #standaloneRafId;
    /** @type {GpuContextManager} Singleton managing the GPU device and canvas configurations globally */
    #gpuContextManager;
    /** @type {((context: any) => void) | null} Stored reconfigure callback for proper unsubscription */
    #onReconfigureCallback;
    /** @type {String} Relative path where all the media files are stored (e.g. /media */
    #mediaDirectory;
    /** @type {string} Track UUID used for log labeling */
    #uuid;

    /**
     * Loads a shader.wgsl file and returns its contents as text.
     *
     * @param {string} url - URL of the shader file.
     * @returns {Promise<string>} The shader as string.
     */
    static async loadShader(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load shader: ${url}`);
        return await response.text();
    }

    /**
     * Loads a binary file and returns its contents as a Float32Array.
     *
     * @param {string} url - URL of the binary file.
     * @returns {Promise<Float32Array>} The binary data as Float32Array.
     */
    static async loadBinaryFile(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load binary file: ${url}`);
        const arrayBuffer = await response.arrayBuffer();
        return new Float32Array(arrayBuffer);
    }

    /**
     * Computes per-channel vertical layout information for the waveform display.
     *
     * Each audio channel is assigned a vertical segment of the canvas. This method
     * evenly divides the total height among channels, distributing any leftover
     * pixels symmetrically around the center channel(s) to maintain visual balance.
     *
     * @param {number} height - Total canvas height in pixels.
     * @param {number} channelCount - Number of audio channels for calculating purposes (binary files).
     * @returns {{offsets: number[], heights: number[]}}
     * An object containing:
     * - `offsets`: The starting Y offset (in pixels) for each channel.
     * - `heights`: The computed height (in pixels) for each channel.
     */
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
     * Constructs the waveform visualizer.
     * @param gpuContextManager A reference to the GpuContextManager class (hack, we have to separate a copy in main repo)
     * @param uuid The UUID of the selected track
     * @param channelCount
     * @param shaderComputeWaveform
     * @param shaderDisplay
     * @param mediaDirectory
     * @param {FrameOrchestratorLike|null} [orchestrator=null] - Optional orchestrator instance for centralized frame control.
     * @throws {Error} If canvas element is not found.
     */
    constructor(gpuContextManager, uuid, channelCount, shaderComputeWaveform, shaderDisplay, mediaDirectory, orchestrator = null) {
        const root = document.querySelector(`[data-uuid="${uuid}"]`);
        /** @type {HTMLCanvasElement} */
        const canvas = /** @type {HTMLCanvasElement} */ (root.querySelector('[data-role="track__waveform-visualizer-canvas"]'));
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
        this.#shaderComputeWaveformCode = shaderComputeWaveform;
        this.#shaderComputeWaveformModule = null;
        this.#shaderDisplayCode = shaderDisplay;
        this.#shaderDisplayModule = null;
        this.#paramsBuffer = null;
        this.#channelLayoutBuffer = null;
        this.#channelCount = channelCount;
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
        this.#maskGroup = [0.0, 0.0, 0.0];
        this.#resizeObserver = null;
        this.#animationFrameId = 0;
        this.#disposed = false;
        this.#needsRender = true;
        this.#isRegistered = false;
        this.#orchestrator = orchestrator;
        this.#orchestratorParticipantId = orchestrator ? `waveform-${uuid}` : null;
        this.#standaloneRafId = null;
        this.#mediaDirectory = mediaDirectory;
        this.#uuid = uuid;

        this.#onReconfigureCallback = null;
        this.#gpuContextManager = gpuContextManager.init();
        this.#context = this.#gpuContextManager.configureCanvas(this.#canvas);
        this.#gpuDevice = this.#gpuContextManager.context.device;
        this.#canvasFormat = this.#gpuContextManager.context.format;
        this.#canvasColorSpace = this.#gpuContextManager.context.colorSpace;
        this.#internalFormat = this.#gpuContextManager.context.internalFormat;

        this.#onReconfigureCallback = (context) => {
            console.log("WaveformVisualizer: HDR/SDR change detected, rebuilding pipeline...");
            this.#canvasFormat = context.format;
            this.#canvasColorSpace = context.colorSpace;
            this.#internalFormat = context.internalFormat;

            this.#setupPipeline();
            this.#resizeTextures();
            this.#setupBindGroups();
        };
        this.#gpuContextManager.onReconfigure(this.#onReconfigureCallback);

        if (!this.#gpuDevice) {
            console.warn("WebGPU device not provided");
            return;
        }

        (async () => {
            this.#shaderComputeWaveformCode = this.#shaderComputeWaveformCode ?? await WaveformVisualizer.loadShader(WaveformVisualizer.#shaderComputeWaveformCodeUrl);
            this.#shaderDisplayCode = this.#shaderDisplayCode ?? await WaveformVisualizer.loadShader(WaveformVisualizer.#shaderDisplayCodeUrl);
            try {
                this.#waveformData = await WaveformVisualizer.loadBinaryFile(`${this.#mediaDirectory}/${uuid}/mean.bin`);
            } catch (error) {
                console.log("WaveformVisualizer is probably running self contained. Loading mean.bin from different location.");
                this.#waveformData = await WaveformVisualizer.loadBinaryFile(WaveformVisualizer.#waveformDataMeanUrl);
            }
            try {
                this.#backgroundData = await WaveformVisualizer.loadBinaryFile(`${this.#mediaDirectory}/${uuid}/peak.bin`);
            } catch (error) {
                console.log("WaveformVisualizer is probably running self contained. Loading peak.bin from different location.");
                this.#backgroundData = await WaveformVisualizer.loadBinaryFile(WaveformVisualizer.#backgroundDataPeakUrl);
            }

            await this.#setupPipeline();
            this.writeParamsBuffer();
            this.#resizeTextures();
            this.#setupBindGroups();
            this.#markDirty();
            this.#registerWithOrchestrator();

            let resizeTimeout = null;
            this.#resizeObserver = new ResizeObserver(entries => {
                for (const entry of entries) {
                    if (entry.target === this.#canvas) {
                        clearTimeout(resizeTimeout);
                        resizeTimeout = setTimeout(() => {
                            this.#resizeTextures();
                            this.#markDirty();
                        }, 100);
                    }
                }
            });
            this.#resizeObserver.observe(this.#canvas);
        })();
    }

    get maskGroup() {
        return this.#maskGroup;
    }

    /* right now we a bound to a maximum of third order ambisonics (for obvious reasons: bandwidth cost vs. quality) */
    set maskGroup({index, value}) {
        // Ensure index is exactly 0, 1, or 2
        if (typeof index !== 'number' || !Number.isInteger(index) || ![0, 1, 2].includes(index)) {
            throw new Error(`Index must be one of 0, 1, or 2 (got ${index})`);
        }
        // Ensure value is numeric and between 0 and 1 inclusive
        if (typeof value !== 'number' || Number.isNaN(value) || value < 0 || value > 1) {
            throw new Error(`Value must be a number between 0 and 1 (got ${value})`);
        }
        this.#maskGroup[index] = value;
        this.writeParamsBuffer();
        this.#markDirty();
    }

    set channelCount(value) {
        if (typeof value !== 'number' || !Number.isInteger(value) || value < 1 || value > 16) {
            throw new Error(`Value must be integer between 1 and 16 (got ${value})`);
        }
        this.#channelCount = value;
        // Rebuild pipeline & buffers for the new channel count
        this.#setupPipeline();    // async but generally fine to call here
        this.writeParamsBuffer();
        this.#resizeTextures();
        this.#setupBindGroups();
        this.#markDirty();
    }

    /**
     * Writes the current visualization parameters to the GPU uniform buffer.
     * Includes: first channel peak, boost, offset, channel count, canvas size, and mask group values.
     */
    writeParamsBuffer() {
        if (!this.#gpuDevice || !this.#paramsBuffer) return;

        const paramsData = new Float32Array([
            this.#firstChannelPeak,
            WaveformVisualizer.BOOST,
            WaveformVisualizer.OFFSET,
            this.#channelCount,
            this.#canvas.width,
            this.#canvas.height,
            0.0, // padding
            0.0, // padding
            this.#maskGroup[0],
            this.#maskGroup[1],
            this.#maskGroup[2],
            0.0, // pad to vec4<f32> (maskGroup.w currently unused)
        ]);
        this.#gpuDevice.queue.writeBuffer(this.#paramsBuffer, 0, paramsData);
        this.#markDirty();
    }

    /**
     * (Re)creates textures and updates bind groups on resize or format change.
     */
    #resizeTextures() {
        // Guard against race condition with render loop
        if (!this.#gpuDevice) return;

        this.#canvas.width = Math.max(1, Math.floor(this.#canvas.clientWidth * this.#devicePixelRatio));
        this.#canvas.height = Math.max(1, Math.floor(this.#canvas.clientHeight * this.#devicePixelRatio));

        const bytes = this.#canvas.width * this.#canvas.height * 8; // vec2<f32>

        this.#computeOutputBuffer?.destroy();
        this.#channelLayoutBuffer?.destroy();

        const outputElementCount = this.#canvas.width * this.#canvas.height;
        const outputBufferSize = outputElementCount * 2 * 4; // vec2<f32> per pixel

        this.#computeOutputBuffer = this.#gpuDevice.createBuffer({
            size: outputBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        this.writeParamsBuffer();

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
        this.#markDirty();
    }

    /**
     * Sets up compute and rendering pipeline
     */
    async #setupPipeline() {
        this.#shaderComputeWaveformCode = this.#shaderComputeWaveformCode ?? await WaveformVisualizer.loadShader(WaveformVisualizer.#shaderComputeWaveformCodeUrl);
        this.#shaderComputeWaveformModule = this.#gpuDevice.createShaderModule({
            code: this.#shaderComputeWaveformCode,
        });

        // Compute channel-0 peak without allocations (no filter/spread)
        let max = -Infinity;
        const stride = this.#channelCount || 1;
        for (let i = 0; i < this.#waveformData.length; i += stride) {
            const v = this.#waveformData[i];
            if (v > max) max = v;
        }
        this.#firstChannelPeak = max;

        // Create a uniform buffer for Params
        this.#paramsBuffer = this.#gpuDevice.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.writeParamsBuffer();

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

        // Compute bind group layout (add binding 4 for firstChannelPeak uniform)
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

        this.#shaderDisplayCode = this.#shaderDisplayCode ?? await WaveformVisualizer.loadShader(WaveformVisualizer.#shaderDisplayCodeUrl);
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

        // Compute bind group (add binding 4 for firstChannelPeak uniform buffer)
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

    #markDirty() {
        this.#needsRender = true;
        // Standalone mode: if no orchestrator is provided, render on next rAF (coalesces changes).
        if (!this.#orchestrator) {
            this.#scheduleStandaloneRender();
        }
    }

    #scheduleStandaloneRender() {
        if (this.#standaloneRafId) return;
        if (!this.#gpuDevice || !this.#context) return;

        this.#standaloneRafId = requestAnimationFrame(() => {
            this.#standaloneRafId = null;
            if (this.#disposed) return;
            if (!this.#needsRender) return;

            const encoder = this.#gpuDevice.createCommandEncoder();
            this.#encode(encoder);
            this.#gpuDevice.queue.submit([encoder.finish()]);
            this.#needsRender = false;
        });
    }

    #encode(encoder) {
        if (!this.#gpuDevice || !this.#computePipeline || !this.#computeBindGroup || !this.#displayPipeline
            || !this.#displayBindGroup || !this.#paramsBuffer) {
            return;
        }

        // --- Compute pass ---
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(this.#computePipeline);
        computePass.setBindGroup(0, this.#computeBindGroup);
        const workgroupsX = Math.ceil(this.#canvas.width / 8);
        const workgroupsY = Math.ceil(this.#canvas.height / 8);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
        computePass.end();

        // --- Render pass ---
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.#context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: {r: 0, g: 0, b: 0, a: 0},
            }]
        });
        renderPass.setPipeline(this.#displayPipeline);
        renderPass.setBindGroup(0, this.#displayBindGroup);
        renderPass.draw(6, 1, 0, 0);
        renderPass.end();
    }

    #registerWithOrchestrator() {
        if (this.#isRegistered) return;
        if (!this.#gpuDevice) return;

        // Standalone mode (no orchestrator) is considered "registered".
        if (!this.#orchestrator) {
            this.#isRegistered = true;
            return;
        }

        const id = this.#orchestratorParticipantId ?? `waveform-${this.#uuid}`;
        this.#orchestratorParticipantId = id;

        // If the orchestrator de-dupes by id, make sure old instances are removed first.
        try {
            if (typeof this.#orchestrator.unregister === 'function') {
                this.#orchestrator.unregister(id);
            }
        } catch (e) {
            // Ignore unregister failures; we'll still attempt to register.
            console.warn(`[WaveformVisualizer] Failed to unregister '${id}' before re-registering:`, e);
        }

        this.#orchestrator.register({
            id,
            // Event-driven; cap high so we are not the limiter when dirty.
            preferredHz: 120,
            wantsFrame: () => !!this.#needsRender,
            encode: (encoder) => {
                if (this.#disposed) return;
                if (!this.#context || !this.#computePipeline || !this.#computeBindGroup || !this.#displayPipeline || !this.#displayBindGroup) return;

                this.#encode(encoder);
                this.#needsRender = false;
            }
        });

        this.#isRegistered = true;
    }

    dispose() {
        this.#disposed = true;
        this.#needsRender = false;

        // Stop render loop
        if (this.#animationFrameId) {
            cancelAnimationFrame(this.#animationFrameId);
            this.#animationFrameId = null;
        }
        if (this.#standaloneRafId) {
            cancelAnimationFrame(this.#standaloneRafId);
            this.#standaloneRafId = null;
        }

        // Disconnect observers
        if (this.#resizeObserver) {
            this.#resizeObserver.disconnect();
            this.#resizeObserver = null;
        }

        // Destroy GPU buffers
        this.#paramsBuffer?.destroy();
        this.#paramsBuffer = null;
        this.#channelLayoutBuffer?.destroy();
        this.#channelLayoutBuffer = null;
        this.#waveformDataBuffer?.destroy();
        this.#waveformDataBuffer = null;
        this.#backgroundDataBuffer?.destroy();
        this.#backgroundDataBuffer = null;
        this.#computeOutputBuffer?.destroy();
        this.#computeOutputBuffer = null;

        // Destroy GPU pipelines and shader modules
        this.#computePipeline = null;
        this.#shaderComputeWaveformModule = null;
        this.#shaderDisplayModule = null;
        this.#displayPipeline = null;
        this.#displaySampler = null;

        // Clear references
        this.#canvas = null;
        this.#context = null;
        this.#gpuDevice = null;
        this.#waveformData = null;
        this.#backgroundData = null;
        this.#maskGroup = null;

        // Unregister from centralized frame orchestrator (important for HTMX remounts)
        if (this.#orchestrator && typeof this.#orchestrator.unregister === 'function' && this.#orchestratorParticipantId) {
            try {
                this.#orchestrator.unregister(this.#orchestratorParticipantId);
            } catch (e) {
                console.warn(`[WaveformVisualizer] Failed to unregister '${this.#orchestratorParticipantId}':`, e);
            }
        }
        this.#orchestratorParticipantId = null;
        this.#orchestrator = null;

        if (this.#gpuContextManager?.offReconfigure && this.#onReconfigureCallback) {
            this.#gpuContextManager.offReconfigure(this.#onReconfigureCallback);
        }
        this.#onReconfigureCallback = null;
        this.#gpuContextManager = null;

        console.info("WaveformVisualizer disposed cleanly.");
    }
}

export default WaveformVisualizer;
