# Audio Chunking for Reduced Latency

## Overview

This document describes the audio chunking implementation for reduced latency in whisper audio transcription during real-time conversations with the LLM.

## Problem Statement

The previous implementation waited until the entire audio message was recorded before sending it to the server for transcription. This caused a major unnecessary delay, especially during real-time conversations using the CallOverlay feature.

## Solution

We've implemented a chunked streaming approach that:

1. **Sends audio chunks progressively** - Audio is sent to the server every 1 second during recording
2. **Provides partial transcriptions** - The backend returns transcriptions as chunks arrive
3. **Reduces perceived latency** - Users experience faster response times

## Architecture

### Backend Changes

#### New Endpoint: `/audio/transcriptions/stream`

- **Method**: POST
- **Purpose**: Handle streaming audio chunks for progressive transcription
- **Request Body**:
  ```json
  {
    "session_id": "unique-session-identifier",
    "chunk_data": "base64-encoded-audio-chunk",
    "is_final": false,
    "language": "en" // optional
  }
  ```
- **Response**:
  ```json
  {
    "text": "partial or full transcript",
    "is_final": false,
    "session_id": "unique-session-identifier",
    "partial_transcripts": ["array", "of", "transcripts"]
  }
  ```

#### Session Management

- Each recording session has a unique `session_id`
- Backend accumulates audio chunks in memory
- Transcribes when minimum chunk size (16KB) is reached
- Cleans up session when `is_final` is true

### Frontend Changes

#### New API Function: `transcribeAudioChunk()`

Located in `src/lib/apis/audio/index.ts`:

```typescript
export const transcribeAudioChunk = async (
  token: string,
  sessionId: string,
  chunkData: string,
  isFinal: boolean = false,
  language?: string
)
```

#### CallOverlay.svelte Updates

1. **Chunked Recording**:
   - MediaRecorder uses 1-second timeslices
   - `ondataavailable` event fires every second
   - Each chunk is sent to the backend immediately

2. **Session Management**:
   - New session ID generated on recording start
   - Accumulated transcript tracked in component state
   - Final transcript submitted when silence detected

3. **Feature Flag**:
   - `useChunkedTranscription` flag enables/disables chunking
   - Defaults to `true` for real-time conversations
   - Gracefully falls back to original approach if disabled

## Benefits

1. **Reduced Latency**: Transcription begins immediately as audio is recorded
2. **Better UX**: Progressive feedback instead of waiting for entire recording
3. **Backward Compatible**: Original `/transcriptions` endpoint remains unchanged
4. **Configurable**: Feature flag allows disabling if needed

## Configuration

The chunked transcription can be controlled via the `useChunkedTranscription` flag in CallOverlay.svelte.

To disable chunking:
```javascript
let useChunkedTranscription = false;
```

## Performance Considerations

1. **Network Traffic**: More frequent requests (every 1 second vs. once per recording)
2. **Server Load**: Transcription happens more frequently but with smaller chunks
3. **Memory Usage**: Backend maintains session state in memory

## Future Improvements

1. **Adaptive Chunk Size**: Adjust based on network conditions
2. **WebSocket Support**: Use WebSocket for more efficient streaming
3. **Partial Transcript Display**: Show live transcription in UI
4. **VoiceRecording.svelte**: Extend chunking to the basic voice recording component
5. **Configurable Timeslice**: Make the 1-second interval user-configurable

## Testing

To test the implementation:

1. Enable the CallOverlay feature in a chat
2. Start speaking - audio chunks will be sent every second
3. Check browser console for "Partial transcript:" logs
4. Verify final transcript is submitted after silence detection
5. Check backend logs for chunk processing

## Troubleshooting

### Chunks not being sent
- Verify `useChunkedTranscription` is `true`
- Check MediaRecorder is starting with timeslice parameter
- Verify network connectivity

### Transcription errors
- Check audio format compatibility (webm should work)
- Verify STT engine configuration
- Check server logs for transcription errors

### Session not cleaning up
- Verify `is_final: true` is sent on last chunk
- Check for JavaScript errors in browser console
- Restart the backend server to clear sessions
