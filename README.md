# A WebGPU HDR audio waveform visualizer

The visualizer uses a binary format to cary information to the browser. These files contain peak and mean audio values.
The data is subset of the original audio data, enough to fill a 4K display, packed as interleaved float32 values.

1. Create an intermediate 32-bit float *.raw audio file (input bitrate doesn't matter) for now 16 channels are hardcoded:  
`$ ffmpeg -i input.caf -f f32le -ac 16 output.raw`

2. To create *peak.bin* and *mean.bin* run `python generate-bins.py` script.  

3. Open a common HTTP server in the root directory, because of security policy in browsers.

That's it. :-)
