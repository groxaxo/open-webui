# OpenWebUI Audio Conversation Implementation - Final Production Plan

**Document:** `ProductionImplementation.md`  
**Status:** **FINAL / READY FOR CODING**  
**Focus:** Reliability, Mobile Compatibility, Memory Safety  
**Date:** December 16, 2025

---

## Executive Summary

This document details the architecture for the "Realtime Voice" feature. It moves beyond drafts to address specific production constraints:
1.  **Non-blocking Backend:** Uses `asyncio.Queue` and `ThreadPoolExecutor` to prevent the WebSocket from blocking the main event loop.
2.  **Mobile Compatibility:** Handles iOS auto-play policies and network switching.
3.  **Memory Safety:** Uses a **Ring Buffer** in the AudioWorklet to prevent Garbage Collection stutters on the client.
4.  **Robust Protocol:** Enforces strict handshakes, max segment sizes, and monotonic timers to prevent edge-case failures.

---

## Part 1: Architecture Overview

### 1.1 The Protocol
**WebSocket Endpoint:** `wss://<host>/api/v1/audio/stream/transcriptions`

**Client -> Server:**
1.  **Handshake:** `{"type": "start", "sample_rate": 16000, "config": {...}}`
2.  **Binary Frames:** Raw `Int16Array` bytes (PCM, Mono, 16kHz). No headers.
3.  **Control:** `{"type": "stop"}`

**Server -> Client:**
1.  **Status:** `{"type": "ready", "session_id": "..."}`
2.  **Events:** `{"type": "speech_start"}`, `{"type": "processing"}`
3.  **Result:** `{"type": "final", "text": "Hello world"}`

### 1.2 The Pipeline
1.  **Browser:** `AudioWorklet` (Ring Buffer) -> Resamples to 16kHz -> Sends `Int16` via WS.
2.  **Server (FastAPI):**
    *   `webrtcvad` (C-extension, fast) filters silence.
    *   Buffers valid speech (max 30s).
    *   On trailing silence (>300ms), offloads last 2s to `SmartTurn` (GPU/CPU thread).
    *   If turn complete: Wraps buffer in WAV container -> Calls internal STT.
    *   Returns text.

---

## Part 2: Backend Implementation

### 2.1 Dependencies
`backend/pyproject.toml`
```toml
webrtcvad = "^2.0.10"
numpy = "^1.24.0"
torch = "^2.0.0"
transformers = "^4.35.0"
```

### 2.2 Streaming Router (`routers/audio_stream.py`)
This implementation uses a decoupled Sender Task, Monotonic time, and Flush-on-Stop logic.

```python
import asyncio
import json
import logging
import uuid
import time
import io
import wave
import numpy as np
import webrtcvad
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Service Imports
from open_webui.services.turn_detection_service import get_turn_detection_service
# You must implement this helper in your audio.py
# from open_webui.routers.audio import transcribe_bytes_wrapper 

router = APIRouter()
log = logging.getLogger(__name__)

# Dedicated thread pool for heavy ML inference (SmartTurn)
INFER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="audio_infer")

# Protocol Constants
TARGET_SR = 16000
FRAME_MS = 20
SAMPLES_PER_FRAME = 320  # 16000 * 0.02
BYTES_PER_FRAME = 640    # 320 * 2 bytes (int16)
TRAILING_SILENCE_MS = 300
SMART_TURN_WINDOW_S = 2.0
SMART_TURN_DEBOUNCE_SEC = 0.5 

# Safety Limits
MAX_SEGMENT_SEC = 30
MAX_SEGMENT_BYTES = TARGET_SR * 2 * MAX_SEGMENT_SEC

@dataclass
class StartConfig:
    sample_rate: int = TARGET_SR

def pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    """Convert Int16 bytes to Float32 [-1.0, 1.0] for ML models."""
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    return x / 32768.0

def pcm16_to_wav(pcm16: bytes) -> bytes:
    """Wrap raw PCM bytes in a valid WAV container for STT engines."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(pcm16)
    return buf.getvalue()

class AudioStreamSession:
    def __init__(self, websocket: WebSocket, cfg: StartConfig, app_state):
        self.ws = websocket
        self.cfg = cfg
        self.app_state = app_state
        self.session_id = str(uuid.uuid4())
        
        # ML Services
        self.vad = webrtcvad.Vad(2) # Mode 2: Aggressive
        self.turn = get_turn_detection_service()

        # State
        self._pending = bytearray()
        self._segment = bytearray()
        self._silence_ms = 0
        self._last_turn_check = 0.0
        self._segment_id = 0

        # Async Concurrency
        self._out_q = asyncio.Queue(maxsize=200)
        self._sender_task = None
        self._stt_lock = asyncio.Lock()

    async def start(self):
        self._sender_task = asyncio.create_task(self._sender_loop())
        await self._out_q.put({"type": "ready", "session_id": self.session_id})

    async def stop(self):
        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass

    async def _sender_loop(self):
        """Dedicated task to drain the queue to the WebSocket."""
        while True:
            msg = await self._out_q.get()
            try:
                await self.ws.send_json(msg)
            except Exception as e:
                log.error(f"WS Send Error: {e}")
                break
            finally:
                self._out_q.task_done()

    async def on_audio_bytes(self, b: bytes):
        """Accumulate bytes and process in 20ms frames."""
        self._pending.extend(b)
        
        # Process full frames
        while len(self._pending) >= BYTES_PER_FRAME:
            frame = bytes(self._pending[:BYTES_PER_FRAME])
            del self._pending[:BYTES_PER_FRAME]
            await self._process_frame(frame)

    async def _process_frame(self, frame: bytes):
        # 1. Fast VAD (C-Extension, runs in main thread safely)
        is_speech = self.vad.is_speech(frame, TARGET_SR)

        if is_speech:
            if not self._segment:
                # Notify frontend that we detected start of speech
                await self._out_q.put({"type": "speech_start"})
            
            self._segment.extend(frame)
            self._silence_ms = 0
            
            # Guard: Max segment size
            if len(self._segment) >= MAX_SEGMENT_BYTES:
                log.info("Max segment size reached, forcing finalization")
                await self._finalize_segment()
            return

        # 2. Silence Handling
        if not self._segment:
            return # Ignore silence at start

        self._segment.extend(frame)
        self._silence_ms += FRAME_MS

        if self._silence_ms < TRAILING_SILENCE_MS:
            return

        # 3. Smart Turn Check (Debounced with Monotonic Time)
        now = time.monotonic()
        if (now - self._last_turn_check) < SMART_TURN_DEBOUNCE_SEC:
            return
        
        self._last_turn_check = now

        # Extract last 2 seconds for context
        tail_bytes = int(TARGET_SR * SMART_TURN_WINDOW_S) * 2
        tail_pcm = bytes(self._segment[-tail_bytes:]) if len(self._segment) > tail_bytes else bytes(self._segment)
        tail_f32 = pcm16_to_float32(tail_pcm)

        # Offload to ThreadPool
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(INFER_POOL, self.turn.predict, tail_f32)

        if result.get("is_turn_point"):
            await self._finalize_segment()
        
        # If not turn point, we just keep buffering silence (user is thinking)

    async def _finalize_segment(self):
        """Packaging and Transcription."""
        if not self._segment:
            return
            
        self._segment_id += 1
        seg_id = self._segment_id
        audio_snapshot = bytes(self._segment) # Copy buffer
        self._segment.clear()
        self._silence_ms = 0

        await self._out_q.put({"type": "processing", "segment_id": seg_id})

        # Process STT
        async with self._stt_lock:
            try:
                wav_data = pcm16_to_wav(audio_snapshot)
                
                # CALL YOUR SHARED STT HELPER HERE
                # text = await transcribe_bytes_wrapper(
                #     self.app_state, 
                #     wav_data, 
                #     user=None
                # )
                
                # Mock response for testing
                text = "This is a simulated transcription."
                
                await self._out_q.put({"type": "final", "segment_id": seg_id, "text": text})
            except Exception as e:
                log.error(f"STT Error: {e}")
                await self._out_q.put({"type": "error", "message": "Transcription failed"})


# Note: Prefix is applied in main.py, so path is just /stream/transcriptions
@router.websocket("/stream/transcriptions")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 1. Strict Handshake
    try:
        msg = await websocket.receive_json()
        if msg.get("type") != "start" or int(msg.get("sample_rate", 0)) != TARGET_SR:
            await websocket.send_json({"type": "error", "code": "bad_config", "message": "Only 16kHz PCM16 supported"})
            await websocket.close(code=1003)
            return
    except:
        await websocket.close(code=1003)
        return

    # Use scope to get app state reliably
    app_state = websocket.scope["app"].state
    session = AudioStreamSession(websocket, StartConfig(), app_state)
    await session.start()

    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg:
                await session.on_audio_bytes(msg["bytes"])
            elif "text" in msg:
                data = json.loads(msg["text"])
                if data.get("type") == "stop":
                    # Flush remaining audio on stop
                    await session._finalize_segment()
                    await websocket.close(code=1000)
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await session.stop()
```

---

## Part 3: Frontend Implementation

### 3.1 AudioWorklet (`static/audio-processor.js`)
**Key Optimization:** Uses a **Ring Buffer** to avoid memory allocation churn (GC) in the render loop.

```javascript
const TARGET_SR = 16000;
const FRAME_SAMPLES = 320; // 20ms @ 16k

class VoiceProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Input ring buffer (64k samples)
    this._RING_LEN = 1 << 16;     
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
```

### 3.2 Svelte Component (`CallOverlay.svelte`)
**Key Fix:** Handles iOS User Gesture requirements properly and dynamically selects `wss://` vs `ws://`.

```typescript
// Svelte Component Logic
let socket: WebSocket | null = null;
let audioContext: AudioContext | null = null;

// 1. User Click Handler (Must happen first for iOS)
const onStartClick = async () => {
    // Initialize Context immediately to satisfy iOS User Gesture
    audioContext = new AudioContext();
    
    // If it's suspended (common on Safari), resume it
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }
    
    // Load Worklet
    await audioContext.audioWorklet.addModule("/static/audio-processor.js");
    
    // Now connect WS
    connectWebSocket();
};

const connectWebSocket = () => {
    // Dynamic Scheme
    const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${wsScheme}://${window.location.host}/api/v1/audio/stream/transcriptions`);
    socket.binaryType = "arraybuffer";

    socket.onopen = () => {
        socket?.send(JSON.stringify({ type: "start", sample_rate: 16000 }));
    };

    socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "ready") {
            // Server is ready, open the mic
            startStreaming();
        } else if (msg.type === "final") {
            // Handle text
            console.log("Transcribed:", msg.text);
        }
    };
};

const startStreaming = async () => {
    if (!audioContext) return;
    
    const micStream = await navigator.mediaDevices.getUserMedia({
        audio: { 
            echoCancellation: true, 
            noiseSuppression: true, 
            autoGainControl: true,
            channelCount: 1 
        }
    });

    const src = audioContext.createMediaStreamSource(micStream);
    const node = new AudioWorkletNode(audioContext, "voice-processor");

    node.port.onmessage = (e) => {
        if (socket?.readyState === WebSocket.OPEN) {
            socket.send(e.data); // Zero-copy send
        }
    };

    // Keep-alive graph
    const zero = audioContext.createGain();
    zero.gain.value = 0;
    src.connect(node);
    node.connect(zero);
    zero.connect(audioContext.destination);
};
```

---

## Part 4: Shared STT Helper

Create this function in `backend/open_webui/routers/audio.py`.

```python
async def transcribe_bytes_wrapper(app_state, wav_bytes: bytes, user):
    """
    Unified entry point for STT.
    Accepts valid WAV bytes, writes to temp file, calls existing logic.
    """
    import tempfile
    import os
    
    # Create temp file because existing `transcription_handler` likely expects a file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name
        
    try:
        # Call your existing logic here
        # result = await transcription_handler(..., file_path=tmp_path, ...)
        # return result['text']
        return "Simulated Text"
    finally:
        os.remove(tmp_path)
```

## Part 5: Integration Checklist

1.  **Route Match:** Ensure `backend/main.py` includes the router with `prefix="/api/v1/audio"`.
2.  **Static Files:** Ensure `audio-processor.js` is accessible at `/static/audio-processor.js`.
3.  **Models:** Ensure `silero_vad` (if used locally) and `smart-turn-v3` are pre-downloaded or allowed to download on first run.
4.  **Reverse Proxy:** If using Nginx/Apache, enable WebSocket Upgrades for `/api/v1/audio/stream/*`.
