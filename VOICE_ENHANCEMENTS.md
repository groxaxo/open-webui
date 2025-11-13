# Voice Call Mode Enhancements

This document describes the enhancements made to the voice call mode in Open WebUI.

## Overview

The voice call mode has been enhanced with several improvements to make it more responsive and provide better user feedback, similar to ChatGPT's voice mode.

## Features

### 1. F1 Hotkey Toggle

Press **F1** at any time to toggle the voice call overlay on/off. This provides quick access to voice mode without needing to click UI buttons.

**Implementation:**
- Added keyboard event listener in `MessageInput.svelte`
- Prevents default browser F1 behavior
- Works globally within the chat interface

### 2. Reduced Silence Detection Timeout

The pause required before audio is sent for transcription has been **reduced from 2000ms (2 seconds) to 800ms**.

**Benefits:**
- Faster response time
- Less sluggish feel
- More natural conversation flow

**Files modified:**
- `CallOverlay.svelte` (line ~403)
- `VoiceRecording.svelte` (line ~339)

### 3. Ambient Audio Feedback

A subtle ambient audio tone plays while waiting for the LLM to respond, providing better user feedback similar to ChatGPT.

**Characteristics:**
- 220Hz (A3) sine wave
- 2% volume (very subtle)
- Includes 0.5Hz LFO modulation for natural sound
- Plays during:
  - User speech transcription (loading state)
  - Waiting for TTS generation after LLM response
- Automatically stops when real audio plays

**Implementation:**
- Uses Web Audio API
- Created in `CallOverlay.svelte`
- Proper cleanup on component destroy

### 4. Silero VAD Integration (Advanced)

Integrated the Silero VAD (Voice Activity Detection) library for more accurate speech detection.

**Status:** Optional feature (disabled by default)

**To Enable:**
```javascript
localStorage.setItem('useSileroVAD', 'true');
```

Then refresh the page and start a voice call.

**Configuration:**
- Positive speech threshold: 0.5
- Negative speech threshold: 0.35
- Redemption time: 800ms (aligned with reduced silence timeout)
- Minimum speech duration: 500ms
- Pre-speech padding: 300ms

**Benefits of Silero VAD:**
- More accurate speech detection
- Better noise filtering
- Reduced false positives
- Machine learning-based approach

**Fallback:**
If VAD initialization fails, the system automatically falls back to the traditional audio analysis method.

**Files:**
- `src/lib/utils/vad.ts` - VAD utility module
- `CallOverlay.svelte` - Integration logic

## Technical Details

### Dependencies Added

- `@ricky0123/vad-web@0.0.29` - Silero VAD implementation for the browser

### Architecture

```
MessageInput.svelte
  ├── F1 key listener
  └── showCallOverlay toggle

CallOverlay.svelte
  ├── Ambient audio generation (Web Audio API)
  ├── Reduced silence timeout (800ms)
  ├── Optional Silero VAD integration
  └── Traditional audio analysis (fallback)

VoiceRecording.svelte
  └── Reduced inactivity timeout (800ms)

src/lib/utils/vad.ts
  ├── VAD configuration interface
  ├── createVAD() factory function
  └── Audio conversion utilities
```

### Audio Processing Pipeline

**Traditional Mode (Default):**
```
User speaks → MediaRecorder → Audio Analysis → 
Silence Detection (800ms) → Transcription → LLM
```

**With Silero VAD (Optional):**
```
User speaks → MediaRecorder → Silero VAD → 
Speech End Detection (800ms) → Transcription → LLM
```

### Code Quality

- All changes are minimal and surgical
- No breaking changes to existing functionality
- Type-safe with proper TypeScript interfaces
- Proper resource cleanup (audio contexts, VAD instances)
- Error handling with graceful fallbacks
- Console logging for debugging

## Usage Examples

### Enable VAD via Browser Console

```javascript
// Enable Silero VAD
localStorage.setItem('useSileroVAD', 'true');
location.reload();

// Disable Silero VAD (back to default)
localStorage.removeItem('useSileroVAD');
location.reload();
```

### Using F1 Hotkey

1. Open any chat
2. Press **F1** to start voice mode
3. Speak naturally
4. Press **F1** again to exit voice mode

## Testing

### Manual Testing Checklist

- [ ] F1 hotkey toggles voice call overlay
- [ ] Ambient audio plays when waiting for LLM response
- [ ] Ambient audio stops when TTS audio plays
- [ ] Speech is detected within 800ms of silence
- [ ] Voice input feels more responsive (less sluggish)
- [ ] VAD can be enabled via localStorage (optional test)
- [ ] System falls back gracefully if VAD fails

### Browser Compatibility

- **Ambient Audio:** All modern browsers with Web Audio API support
- **Silero VAD:** All modern browsers with WASM support
- **F1 Hotkey:** All browsers

## Future Improvements

Potential enhancements for future iterations:

1. Add UI toggle for Silero VAD (instead of localStorage)
2. Make silence timeout configurable via settings
3. Add different ambient audio options (white noise, other tones)
4. Persist VAD setting in user preferences
5. Add visual indicator when speech is detected
6. Optimize VAD model loading and caching

## Troubleshooting

### Ambient audio not playing
- Check browser's Web Audio API support
- Verify audio output device is working
- Check browser console for errors

### VAD not working
- Ensure localStorage.useSileroVAD is set to 'true'
- Refresh the page after enabling
- Check browser console for VAD initialization errors
- Verify WASM is supported by your browser

### F1 hotkey not working
- Make sure focus is on the chat window
- Check if another extension is capturing F1
- Try clicking in the chat input area first

## References

- [Silero VAD Repository](https://github.com/ricky0123/vad)
- [Web Audio API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [MediaRecorder API](https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder)
