import GpuContextManager from "./gpuContextManager.js";
import WaveformVisualizer from "./waveformVisualizer.js";

const gpuContextManager = GpuContextManager.init();
await gpuContextManager.configureDevice();
const waveformVisualizer = WaveformVisualizer.init(document, gpuContextManager);
