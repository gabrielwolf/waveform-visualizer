# A WebGPU HDR audio waveform visualizer

The visualizer uses a binary format to cary information to the browser. These files contain peak or mean audio values.
It is subset of the original audio data, enough to fill a 4K display, stored as interleaved uint16 values.

To create *peak.bin* and *mean.bin* files we need *.raw audio first:

`$ ffmpeg -i input.caf -f s16le -ac 16 output.raw`

16 bit is enough as HDR up to 4000 nits is specified with 12 bit, and anyone apart from professional colorists has
a monitor that can go that bright as of 2025.

As second step we run `python generate-bins.py` script. Done.
