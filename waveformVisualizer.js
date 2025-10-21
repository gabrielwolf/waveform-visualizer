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
     * (Re)creates textures and updates bind groups on resize or format change.
     * @private
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
     * Sets up compute and rendering pipeline (stub for compute-based approach).
     * @private
     */
    #setupPipeline() {
        // In compute-based approach, setup compute pipeline and related buffers here.
        // This is a stub placeholder.
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
        // In compute-based approach, perform compute pass and present result here.
        // This is a stub placeholder.
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
