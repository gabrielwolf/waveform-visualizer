import GpuContextManager from "./gpuContextManager.js";
import WaveformVisualizer from "./waveformVisualizer.js";

const gpuContextManager = GpuContextManager.init();
await gpuContextManager.configureDevice();

/*
 * We calculate the location of the necessary binary files without input data from the UUID
 * but we need the channel count as the markup gets fed into the browser (snappy loading)
 * In this demo we use ambisonic order [1;3] to calculate the channelCount
 */

const article = document.querySelector('article[data-uuid][data-order]');
if (!article) {
    throw new Error("No <article> element with waveform data found");
}

const uuid = article.getAttribute('data-uuid');
const orderString = article.getAttribute('data-order');
const channelCount = (Number(orderString) + 1) ** 2;
if (!uuid) {
    throw new Error("[Waveform Visualizer] Missing UUID");
}
if (!/^\d+$/.test(channelCount)) {
    throw new Error(`[Waveform Visualizer] Channel count must be a valid integer, got "${channelCount}"`);
}
if (!Number.isInteger(channelCount) || channelCount < 1 || channelCount > 16) {
    throw new Error("[Waveform Visualizer] Channel count must be an integer between 1 and 16 inclusive");
}


article.__waveformVisualizerInstance = new WaveformVisualizer(GpuContextManager, uuid, channelCount);

// 1. An example for animating the (e.g. loading) progress
setTimeout(() => {
    article.__waveformVisualizerInstance.maskGroup = {index: 0, value: 0.5};
}, 1000);
setTimeout(() => {
    article.__waveformVisualizerInstance.maskGroup = {index: 1, value: 0.5};
}, 2000);
setTimeout(() => {
    article.__waveformVisualizerInstance.maskGroup = {index: 2, value: 0.5};
}, 3000);


// 2. An example for controlling the brightness BOOST parameter
setTimeout(() => {
    WaveformVisualizer.BOOST = 0.2;
    article.__waveformVisualizerInstance.writeParamsBuffer();
}, 5000);


// 3. An example for controlling brightness OFFSET parameter
setTimeout(() => {
    WaveformVisualizer.OFFSET = 0.8;
    article.__waveformVisualizerInstance.writeParamsBuffer();
}, 6000);


// 4. An example animation, that ends on the default optimal brightness settings both on HDR and SDR
const waveformVisualizer = article.__waveformVisualizerInstance;
const lerp = (a, b, t) => a + (b - a) * t;
const easeInOut = (t) => t * t * (3 - 2 * t); // smoothstep

function tween({ duration = 2000, from, to, onUpdate, easing = easeInOut, onDone }) {
  const start = performance.now();
  function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    const e = easing(t);

    onUpdate((key) => lerp(from[key], to[key], e), e);

    if (t < 1) requestAnimationFrame(frame);
    else onDone?.();
  }
  requestAnimationFrame(frame);
}

setTimeout(() => {
  const from = {
    m0: waveformVisualizer.maskGroup[0], // or 0.5
    m1: waveformVisualizer.maskGroup[1],
    m2: waveformVisualizer.maskGroup[2],
    boost: WaveformVisualizer.BOOST,
    offset: WaveformVisualizer.OFFSET,
  };

  const to = { m0: 1.0, m1: 1.0, m2: 1.0, boost: 1.2, offset: 0.15 };

  tween({
    duration: 2500,
    from,
    to,
    onUpdate: (L) => {
      // "proper" usage via setter:
      waveformVisualizer.maskGroup = { index: 0, value: L("m0") };
      waveformVisualizer.maskGroup = { index: 1, value: L("m1") };
      waveformVisualizer.maskGroup = { index: 2, value: L("m2") };

      WaveformVisualizer.BOOST = L("boost");
      WaveformVisualizer.OFFSET = L("offset");
      waveformVisualizer.writeParamsBuffer();
    },
  });
}, 8000);