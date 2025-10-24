import WaveformVisualizer from "./waveformVisualizer.js";
import GpuContextManager from "./gpuContextManager.js";

const gpuContextManager = GpuContextManager.init();
await gpuContextManager.configureDevice();
const waveformVisualizer = WaveformVisualizer.init(document, gpuContextManager);
