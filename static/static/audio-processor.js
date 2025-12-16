/**
 * Audio Processor Worklet for Real-time Voice Streaming
 * 
 * This worklet:
 * - Uses a ring buffer to avoid GC pressure
 * - Resamples audio to 16kHz
 * - Converts to PCM16 format
 * - Sends frames to the main thread via zero-copy transfers
 */

const TARGET_SR = 16000;
const FRAME_SAMPLES = 320; // 20ms @ 16kHz
const RING_BUFFER_SIZE_BITS = 16; // 2^16 = 64k samples

class VoiceProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Input ring buffer (64k samples = 4 seconds @ 48kHz)
    this._RING_LEN = 1 << RING_BUFFER_SIZE_BITS;     
    this._MASK = this._RING_LEN - 1;
    this._ring = new Float32Array(this._RING_LEN);
    this._wAbs = 0;               // total written samples (absolute)
    this._rAbs = 0;               // read position in input-sample units (float)

    // Resample ratio: input samples per output sample
    this._step = sampleRate / TARGET_SR;

    // Output frame (PCM16)
    this._out = new Int16Array(FRAME_SAMPLES);
    this._outIndex = 0;
  }

  _writeInput(ch0) {
    // Safety: If writer laps reader, jump reader forward (drop oldest)
    const maxLag = this._RING_LEN - 4;
    if ((this._wAbs - Math.floor(this._rAbs)) > maxLag) {
      this._rAbs = this._wAbs - maxLag;
    }

    for (let i = 0; i < ch0.length; i++) {
      this._ring[this._wAbs & this._MASK] = ch0[i];
      this._wAbs++;
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;

    const ch0 = input[0];
    this._writeInput(ch0);

    // Resample from Ring Buffer
    // Need at least 2 input samples ahead for interpolation
    while ((this._wAbs - Math.floor(this._rAbs)) > 1) {
      const i0 = Math.floor(this._rAbs);
      const frac = this._rAbs - i0;

      const a = this._ring[i0 & this._MASK];
      const b = this._ring[(i0 + 1) & this._MASK];
      const val = a + (b - a) * frac;

      const s = Math.max(-1, Math.min(1, val));
      this._out[this._outIndex++] = (s * 0x7fff) | 0;

      if (this._outIndex === FRAME_SAMPLES) {
        // Zero-copy transfer to main thread
        const bufferToSend = this._out.buffer;
        this.port.postMessage(bufferToSend, [bufferToSend]);
        
        // Allocate fresh buffer for next frame
        this._out = new Int16Array(FRAME_SAMPLES);
        this._outIndex = 0;
      }

      this._rAbs += this._step;
    }

    return true;
  }
}

registerProcessor("voice-processor", VoiceProcessor);
