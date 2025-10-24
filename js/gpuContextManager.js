/**
 * Holds a WebGPU device and canvas configuration for visualizers.
 */
class GpuContext {
    /** @type {GPUDevice} */
    #device;
    /** @type {GPUTextureFormat|null} */
    #format;
    /** @type {PredefinedColorSpace|null} */
    #colorSpace;
    /** @type {GPUTextureFormat} */
    #internalFormat
        /** @type {boolean} */;
    #isHighDynamicRange;

    /**
     * @param {GPUDevice} device - The WebGPU device to use.
     * @param {GPUTextureFormat|null} [format=null] - Chosen canvas texture format.
     * @param {PredefinedColorSpace|null} [colorSpace=null] - Chosen color space.
     * @param {GPUTextureFormat} [internalFormat="rgba16float"] - Internal format for computations.
     * @param {boolean} [isHighDynamicRange=false] - true = HDR, false = SDR
     */
    constructor(device, format, colorSpace, internalFormat, isHighDynamicRange) {
        this.#device = device;
        this.#format = format;
        this.#colorSpace = colorSpace;
        this.#internalFormat = internalFormat;
        this.#isHighDynamicRange = isHighDynamicRange;
    }

    get device() {
        return this.#device;
    }

    set device(value) {
        this.#device = value;
    }

    get format() {
        return this.#format;
    }

    set format(value) {
        this.#format = value;
    }

    get colorSpace() {
        return this.#colorSpace;
    }

    set colorSpace(value) {
        this.#colorSpace = value;
    }

    get internalFormat() {
        return this.#internalFormat;
    }

    set internalFormat(value) {
        this.#internalFormat = value;
    }

    get isHighDynamicRange() {
        return this.#isHighDynamicRange;
    }

    set isHighDynamicRange(value) {
        this.#isHighDynamicRange = value;
    }
}

/**
 * Singleton managing a WebGPU device and providing consistent canvas configuration.
 * Use `GpuContextManager.init()` to obtain the instance, and `configureDevice()` before accessing `.context`.
 */
class GpuContextManager {
    static #instance = null;
    #context;
    #listeners = new Set();
    #canvases = new Set();

    /**
     * Private constructor. Use GpuContextManager.init().
     */
    constructor() {
        this.#context = null;

        const dynamicRangeQuery = window.matchMedia("(dynamic-range: high)");
        dynamicRangeQuery.addEventListener("change", (event) => {
            const highDynamicRange = event.matches;
            this.#context.isHighDynamicRange = highDynamicRange;
            const dynamicRange = highDynamicRange ? "high" : "standard";
            console.log(`HDR/SDR change detected, dynamicRange: ${dynamicRange}`);

            if (this.#context) {
                for (const callback of this.#listeners) {
                    callback(this.#context);
                }

                for (const canvas of this.#canvases) {
                    try {
                        const context = GpuContextManager.#instance.configureCanvas(canvas);
                        let identifier = canvas.dataset?.role || canvas.id || canvas.className || "<unnamed canvas>";
                        console.log(
                            `Reconfig done for canvas [${identifier}]`,
                            GpuContextManager.#instance.context.format,
                            GpuContextManager.#instance.context.colorSpace,
                            dynamicRange
                        );
                    } catch (error) {
                        let identifier = canvas.dataset?.role || canvas.id || canvas.className || "<unnamed canvas>";
                        console.warn(`Reconfigure failed after HDR/SDR switch for canvas [${identifier}]:`, error);
                    }
                }
            }
        });
    }

    /**
     * Returns the singleton instance of GpuContextManager.
     * Creates it if it does not exist.
     * @returns {GpuContextManager} Singleton instance.
     */
    static init() {
        if (!GpuContextManager.#instance) {
            GpuContextManager.#instance = new GpuContextManager();
        }
        return GpuContextManager.#instance;
    }

    /**
     * Initializes the GPU adapter and device if not already initialized.
     * Sets the internal GpuContext object (format/colorSpace will be null until a canvas is configured).
     * Must be called before accessing `context`.
     * @returns {Promise<void>}
     * @throws {Error} if WebGPU is not supported or adapter/device cannot be obtained.
     */
    async configureDevice() {
        if (!navigator.gpu) throw new Error("WebGPU not supported.");
        if (this.#context) return;

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("WebGPU adapter not available.");
        const device = await adapter.requestDevice();

        // These will be set after configureCanvas, but allow null for now
        this.#context = new GpuContext(device, null, null, "rgba16float");
    }

    /**
     * Returns the GpuContext object containing the GPUDevice and canvas configuration.
     * @type {GpuContext}
     * @throws {Error} if the GPU device has not been initialized yet.
     */
    get context() {
        if (!this.#context) throw new Error("GpuContextManager not initialized yet");
        return this.#context;
    }

    /**
     * Configures a given canvas element with the best available GPU format and color space.
     * Updates the internal GpuContext.format and GpuContext.colorSpace.
     * Tracks the canvas for future reconfiguration.
     * @param {HTMLCanvasElement} canvas - The canvas element to configure.
     * @returns {GPUCanvasContext} The configured WebGPU canvas context.
     */
    configureCanvas(canvas) {
        const context = canvas.getContext("webgpu");
        const colorSpaceFallbacks = ["rec2100-hlg", "display-p3", "srgb"];
        const formatFallbacks = ["rgba16float", "rgba8unorm"];

        for (const format of formatFallbacks) {
            for (const colorSpace of colorSpaceFallbacks) {
                // Technically possible but 8-bit HDR is bad
                if (format === "rgba8unorm" && colorSpace === "rec2100-hlg") continue;
                try {
                    context.configure({
                        device: this.#context.device,
                        format,
                        colorSpace,
                        alphaMode: "premultiplied",
                    });
                    console.log(`WebGPU canvas configured: ${format}, ${colorSpace}`);
                    const changed = this.#context.format !== format || this.#context.colorSpace !== colorSpace;
                    this.#context.format = format;
                    this.#context.colorSpace = colorSpace;
                    if (changed) {
                        this.#listeners.forEach(callback => callback(this.#context));
                    }
                    this.#canvases.add(canvas);
                    return context;
                } catch (error) {
                    console.log(`Failed with ${format}, ${colorSpace}`);
                }
            }
        }

        // Fallback
        const fallbackFormat = "rgba8unorm";
        const fallbackColorSpace = "srgb";
        context.configure({
            device: this.#context.device,
            format: fallbackFormat,
            alphaMode: "premultiplied",
        });
        console.log("Fallback: rgba8unorm + srgb");
        const changed = this.#context.format !== fallbackFormat || this.#context.colorSpace !== fallbackColorSpace;
        this.#context.format = fallbackFormat;
        this.#context.colorSpace = fallbackColorSpace;
        if (changed) {
            this.#listeners.forEach(callback => callback(this.#context));
        }
        this.#canvases.add(canvas);
        return context;
    }

    onReconfigure(callback) {
        this.#listeners.add(callback);
    }

    offReconfigure(callback) {
        this.#listeners.delete(callback);
    }
}

export default GpuContextManager;
