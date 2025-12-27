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
    /** @type {boolean} */
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
    /** @type {GpuContextManager} */
    static #instance = null;
    /** @type {GpuContext} */
    #context;
    /** @type {Set} */
    #listeners;
    /** @type {Set} */
    #canvases;
    /** @type {boolean} */
    #initialHdr;

    /**
     * Private constructor. Use GpuContextManager.init().
     */
    constructor() {
        this.#context = null;
        this.#listeners = new Set();
        this.#canvases = new Set();
        this.#initialHdr = window.matchMedia("(dynamic-range: high)").matches;

        const dynamicRangeQuery = window.matchMedia("(dynamic-range: high)");
        dynamicRangeQuery.addEventListener("change", (event) => {
            const highDynamicRange = event.matches;
            if (!this.#context) return;
            this.#context.isHighDynamicRange = highDynamicRange;
            const dynamicRange = highDynamicRange ? "high" : "standard";
            console.log(`HDR/SDR change detected, dynamicRange: ${dynamicRange}`);

            if (this.#context) {
                // Re-detect best canvas configuration
                const {format, colorSpace} = this.#detectBestCanvasConfiguration(this.#context.device);
                this.#context.format = format;
                this.#context.colorSpace = colorSpace;

                for (const canvas of this.#canvases) {
                    try {
                        const context = canvas.getContext("webgpu");
                        context.configure({
                            device: this.#context.device,
                            format: this.#context.format,
                            colorSpace: this.#context.colorSpace,
                            alphaMode: "premultiplied",
                        });
                        let identifier = canvas.dataset?.role || canvas.id || canvas.className || "<unnamed canvas>";
                        console.log(
                            `Reconfig done for canvas [${identifier}]`,
                            this.#context.format,
                            this.#context.colorSpace,
                            dynamicRange
                        );
                    } catch (error) {
                        let identifier = canvas.dataset?.role || canvas.id || canvas.className || "<unnamed canvas>";
                        console.warn(`Reconfigure failed after HDR/SDR switch for canvas [${identifier}]:`, error);
                    }
                }

                for (const callback of this.#listeners) {
                    callback(this.#context);
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

        const highDynamicRange = window.matchMedia("(dynamic-range: high)").matches;

        // Create context with null format/colorSpace initially
        this.#context = new GpuContext(device, null, null, "rgba16float", highDynamicRange);

        // Detect and set best canvas configuration once (until it changes)
        const {format, colorSpace} = this.#detectBestCanvasConfiguration(device);
        this.#context.format = format;
        this.#context.colorSpace = colorSpace;
    }

    #isWebKitLike() {
        // All iOS browsers are WebKit; Safari on macOS is WebKit.
        const ua = navigator.userAgent;
        const isWebKit = /AppleWebKit\//.test(ua);
        const isChromiumLike = /(Chrome|Chromium|Edg|OPR|Brave|Vivaldi|CriOS)/.test(ua);
        return isWebKit && !isChromiumLike;
    }

    /**
     * Private method to detect the best canvas configuration for the device.
     * Creates a temporary canvas and tries opinionated format/colorSpace combinations.
     * @param {GPUDevice} device
     * @returns {{format: GPUTextureFormat, colorSpace: PredefinedColorSpace}} Best found configuration.
     */
    #detectBestCanvasConfiguration(device) {
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("webgpu");
        const preferredFormat = (navigator.gpu?.getPreferredCanvasFormat?.() ?? "bgra8unorm");

        // WebKit/Safari has had issues with transparent WebGPU canvases when using HDR color spaces
        // and/or 16-bit float swapchains. Prefer SDR + preferred format there.
        const isWebKit = this.#isWebKitLike();

        // If the OS/browser reports HDR capability, we can try HDR-first on non-WebKit browsers
        // for best quality. (We also listen to dynamic-range media query changes.)
        const wantsHdr = !!this.#context?.isHighDynamicRange;

        let colorSpaceFallbacks;
        let formatFallbacks;

        if (isWebKit) {
            // Transparency reliability > HDR on WebKit.
            colorSpaceFallbacks = ["display-p3", "srgb", "rec2100-hlg"];
            formatFallbacks = [preferredFormat, "bgra8unorm", "rgba8unorm", "rgba16float"];
        } else {
            // Best visual quality on Chromium/Firefox: try HDR + 16-bit first, then fall back.
            colorSpaceFallbacks = ["rec2100-hlg", "display-p3", "srgb"];
            formatFallbacks = ["rgba16float", preferredFormat, "bgra8unorm", "rgba8unorm"];
        }

        for (const format of formatFallbacks) {
            for (const colorSpace of colorSpaceFallbacks) {
                // Technically possible but 8-bit HDR is bad
                if (format === "rgba8unorm" && colorSpace === "rec2100-hlg") continue;
                try {
                    context.configure({
                        device,
                        format,
                        colorSpace,
                        alphaMode: "premultiplied",
                    });
                    console.log(
                        `WebGPU canvas detected supported configuration: ${format}, ${colorSpace}`,
                        isWebKit ? "(WebKit-transparency-safe)" : "(quality-first)",
                        wantsHdr ? "HDR" : "SDR"
                    );
                    return {format, colorSpace};
                } catch {
                    // ignore and try next
                }
            }
        }

        // Fallback
        console.log("Fallback canvas configuration used: rgba8unorm, srgb");
        return {format: "rgba8unorm", colorSpace: "srgb"};
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
     * Configures a given canvas element with the stored GPU format and color space.
     * Tracks the canvas for future reconfiguration.
     * @param {HTMLCanvasElement} canvas - The canvas element to configure.
     * @returns {GPUCanvasContext} The configured WebGPU canvas context.
     */
    configureCanvas(canvas) {
        const context = canvas.getContext("webgpu");
        context.configure({
            device: this.#context.device,
            format: this.#context.format,
            colorSpace: this.#context.colorSpace,
            alphaMode: "premultiplied",
        });
        this.#canvases.add(canvas);
        console.log(`WebGPU canvas configured: ${this.#context.format}, ${this.#context.colorSpace}`);
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
