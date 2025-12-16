# OpenWebUI Audio Conversation & Real-Time Streaming Implementation Plan

**Date:** December 2025  
**Project:** OpenWebUI Audio Enhancement  
**Goal:** Implement VAD (Voice Activity Detection) and real-time audio streaming with smart turn detection

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Architecture Analysis](#current-architecture-analysis)
3. [Proposed Architecture](#proposed-architecture)
4. [Implementation Details](#implementation-details)
5. [Code Snippets & Integration](#code-snippets--integration)
6. [Testing Strategy](#testing-strategy)
7. [Deployment & Configuration](#deployment--configuration)

---

## Executive Summary

### Current State
OpenWebUI implements **file-based audio processing**:
- **STT (Speech-to-Text)**: Upload audio files → transcription
- **TTS (Text-to-Speech)**: Generate and cache audio from text
- **Voice Conversations**: CallOverlay component with basic silence detection

### Proposed Enhancements
1. **Silero VAD Integration**: Replace simple silence detection with production-grade VAD
2. **Smart Turn Detection**: Use HuggingFace `pipecat-ai/smart-turn-v3` model for natural conversation flow
3. **Real-Time Streaming**: WebSocket-based bidirectional audio streaming
4. **Improved Call Experience**: Reduced latency, better interruption handling, natural turn-taking

---

## Current Architecture Analysis

### Backend: `/backend/open_webui/routers/audio.py`

#### Key Components

**1. STT Engines (Speech-to-Text)**

```python
# Supported engines:
- whisper (faster-whisper local)
- openai (API)
- deepgram (API)
- azure (API)
- mistral (API)
```

**Current STT Flow:**
```
User uploads audio file
    ↓
transcription endpoint (/transcriptions)
    ↓
File saved to cache
    ↓
Compression & splitting (if large)
    ↓
transcription_handler()
    ↓
Engine-specific processing
    ↓
Return {"text": "transcription"}
```

**2. TTS Engines (Text-to-Speech)**
```python
# Supported engines:
- openai
- elevenlabs
- azure
- transformers (local)
```

**Current TTS Flow:**
```
POST /speech with text
    ↓
Hash-based cache check
    ↓
If cached: return file
If not cached:
    ↓
Generate audio via engine
    ↓
Save to cache
    ↓
Return MP3 file
```

**3. VAD Configuration**
```python
# From config.py
WHISPER_VAD_FILTER = PersistentConfig(
    "WHISPER_VAD_FILTER",
    os.getenv("WHISPER_VAD_FILTER", "False").lower() == "true",
)

# Used in transcription_handler()
segments, info = model.transcribe(
    file_path,
    beam_size=5,
    vad_filter=request.app.state.config.WHISPER_VAD_FILTER,  # ← Current VAD
    language=languages[0],
)
```

### Frontend Audio Components

#### 1. **VoiceRecording.svelte** (Basic Recording)

**Location:** `/src/lib/components/chat/MessageInput/VoiceRecording.svelte`

**Features:**
- MediaRecorder API usage
- Basic audio visualization (RMS levels)
- Browser-based STT (Web Speech API) OR server-based
- File upload for transcription

**Current Audio Analysis:**
```typescript
const analyseAudio = (stream) => {
    const audioContext = new AudioContext();
    const audioStreamSource = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    analyser.minDecibels = MIN_DECIBELS;  // -45 dB
    
    // Calculate RMS for visualization
    const calculateRMS = (data: Uint8Array) => {
        let sumSquares = 0;
        for (let i = 0; i < data.length; i++) {
            const normalizedValue = (data[i] - 128) / 128;
            sumSquares += normalizedValue * normalizedValue;
        }
        return Math.sqrt(sumSquares / data.length);
    };
}
```

#### 2. **CallOverlay.svelte** (Voice Conversation Mode)

**Location:** `/src/lib/components/chat/MessageInput/CallOverlay.svelte`

**Current Features:**
- Continuous listening with MediaRecorder
- Simple silence detection (2 seconds)
- Audio caching for TTS responses
- Interruption support
- Camera/video support for multimodal input

**Current Turn Detection Logic:**
```typescript
const detectSound = () => {
    const processFrame = () => {
        analyser.getByteTimeDomainData(timeDomainData);
        analyser.getByteFrequencyData(domainData);
        
        const hasSound = domainData.some((value) => value > 0);
        if (hasSound) {
            if (!hasStartedSpeaking) {
                hasStartedSpeaking = true;
                stopAllAudio();  // Interrupt assistant
            }
            lastSoundTime = Date.now();
        }
        
        // Simple silence detection
        if (hasStartedSpeaking) {
            if (Date.now() - lastSoundTime > 2000) {  // 2 second silence
                confirmed = true;
                mediaRecorder.stop();  // Send for transcription
                return;
            }
        }
    };
};
```

**Conversation Flow:**
```
User starts speaking
    ↓
Sound detected → Stop assistant audio
    ↓
Record audio chunks
    ↓
2 seconds of silence
    ↓
Stop recording → Send to STT
    ↓
Get transcription
    ↓
Submit to LLM
    ↓
Stream response (text)
    ↓
Split into sentences
    ↓
Generate TTS for each sentence
    ↓
Queue and play audio
```

### Current Limitations

1. **No Production-Grade VAD**: Simple threshold-based detection
2. **Fixed Silence Timeout**: 2 seconds is not adaptive
3. **No Turn-Taking Intelligence**: Can't distinguish pauses vs. turn completion
4. **Latency Issues**: 
   - Wait for silence
   - Upload entire audio
   - Process STT
   - Wait for LLM
   - Generate TTS
   - Play audio
5. **No Real-Time Streaming**: Everything is request/response
6. **Limited Interruption**: Basic audio stopping, no graceful handling

---

## Proposed Architecture

### Enhanced Audio Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      Client (Browser)                        │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │ Microphone   │─────▶│  Silero VAD  │─────▶│ WebSocket │ │
│  │  MediaStream │      │  (ONNX/WASM) │      │  Client   │ │
│  └──────────────┘      └──────────────┘      └─────┬─────┘ │
│                                                      │       │
│                                              ┌───────▼─────┐ │
│                                              │ Audio Player│ │
│                                              │  (Streaming)│ │
│                                              └─────────────┘ │
└──────────────────────────────────────────────────────────────┘
                                │                    ▲
                                │ Audio Chunks       │ Audio Stream
                                ▼                    │
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │  WebSocket   │─────▶│ Smart Turn   │─────▶│ STT Engine│ │
│  │   Handler    │      │ Detection    │      │ (Streaming)│ │
│  └──────────────┘      │  (Pipecat)   │      └─────┬─────┘ │
│                        └──────────────┘            │       │
│                                                     ▼       │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │ Audio Output │◀─────│  TTS Engine  │◀─────│    LLM    │ │
│  │  (Streaming) │      │  (Streaming) │      │ (Streaming)│ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Three-Tier VAD Strategy

**Tier 1: Browser-Side VAD (Silero ONNX)**
- Fast, local processing
- Reduce bandwidth by not sending silence
- Immediate feedback to user

**Tier 2: Server-Side VAD (Silero PyTorch)**
- Higher accuracy
- Handle edge cases
- Verification layer

**Tier 3: Smart Turn Detection (Pipecat AI)**
- Understand conversation context
- Natural turn-taking
- Adaptive thresholds

---

## Implementation Details

### Phase 1: Silero VAD Integration

#### Backend Implementation

**New File:** `/backend/open_webui/routers/audio_stream.py`
