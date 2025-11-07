#!/usr/bin/env python3
"""
Validation script for audio chunking implementation.
This script performs basic checks on the implementation.
"""

import ast
import sys
from pathlib import Path

def check_audio_router():
    """Check that the audio router has the required changes."""
    audio_py = Path("/home/runner/work/open-webui/open-webui/backend/open_webui/routers/audio.py")
    
    if not audio_py.exists():
        print(f"❌ File not found: {audio_py}")
        return False
    
    content = audio_py.read_text()
    
    checks = {
        "transcription_sessions dict": "transcription_sessions = {}" in content,
        "TranscriptionChunkForm class": "class TranscriptionChunkForm" in content,
        "/transcriptions/stream endpoint": '@router.post("/transcriptions/stream")' in content,
        "session_id parameter": "session_id: str" in content,
        "is_final parameter": "is_final: bool" in content,
        "chunk_data parameter": "chunk_data: str" in content,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} Backend check: {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def check_audio_api():
    """Check that the frontend API has the required changes."""
    audio_ts = Path("/home/runner/work/open-webui/open-webui/src/lib/apis/audio/index.ts")
    
    if not audio_ts.exists():
        print(f"❌ File not found: {audio_ts}")
        return False
    
    content = audio_ts.read_text()
    
    checks = {
        "transcribeAudioChunk export": "export const transcribeAudioChunk" in content,
        "sessionId parameter": "sessionId: string" in content,
        "chunkData parameter": "chunkData: string" in content,
        "isFinal parameter": "isFinal: boolean" in content,
        "/transcriptions/stream endpoint": '${AUDIO_API_BASE_URL}/transcriptions/stream' in content,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} Frontend API check: {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def check_call_overlay():
    """Check that the CallOverlay component has the required changes."""
    call_overlay = Path("/home/runner/work/open-webui/open-webui/src/lib/components/chat/MessageInput/CallOverlay.svelte")
    
    if not call_overlay.exists():
        print(f"❌ File not found: {call_overlay}")
        return False
    
    content = call_overlay.read_text()
    
    checks = {
        "Import transcribeAudioChunk": "transcribeAudioChunk" in content,
        "transcriptionSessionId variable": "let transcriptionSessionId" in content,
        "useChunkedTranscription flag": "let useChunkedTranscription" in content,
        "transcribeChunkHandler function": "const transcribeChunkHandler" in content,
        "Session ID generation": "transcriptionSessionId =" in content,
        "MediaRecorder timeslice": "mediaRecorder.start(timeslice)" in content or "const timeslice" in content,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} CallOverlay check: {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def main():
    """Run all validation checks."""
    print("=" * 60)
    print("Audio Chunking Implementation Validation")
    print("=" * 60)
    print()
    
    backend_ok = check_audio_router()
    print()
    
    api_ok = check_audio_api()
    print()
    
    overlay_ok = check_call_overlay()
    print()
    
    print("=" * 60)
    if backend_ok and api_ok and overlay_ok:
        print("✅ All validation checks passed!")
        print("=" * 60)
        return 0
    else:
        print("❌ Some validation checks failed!")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
