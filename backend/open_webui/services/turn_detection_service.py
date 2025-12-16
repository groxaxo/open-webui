"""
Turn Detection Service for Audio Streaming

This service uses a simple heuristic-based approach to detect turn points in conversation.
In production, you could replace this with a more sophisticated ML model.
"""

import logging
import numpy as np

log = logging.getLogger(__name__)

# Constants matching audio_stream.py
FRAME_SAMPLES = 320  # 20ms @ 16kHz


class TurnDetectionService:
    """
    Simple turn detection service that analyzes audio energy and silence patterns.
    """

    def __init__(self):
        self.energy_threshold = 0.01  # Threshold for speech energy
        self.silence_ratio_threshold = 0.7  # Ratio of silence frames to indicate turn

    def predict(self, audio_float32: np.ndarray) -> dict:
        """
        Analyze audio segment and predict if it's a turn point.
        
        Args:
            audio_float32: Float32 audio array normalized to [-1.0, 1.0]
            
        Returns:
            Dictionary with 'is_turn_point' boolean
        """
        try:
            if len(audio_float32) == 0:
                return {"is_turn_point": False}

            # Calculate RMS energy for the segment
            rms = np.sqrt(np.mean(audio_float32**2))
            
            # Calculate how many frames are below energy threshold
            num_frames = len(audio_float32) // FRAME_SAMPLES
            
            if num_frames == 0:
                return {"is_turn_point": False}
            
            silence_frames = 0
            for i in range(num_frames):
                start = i * FRAME_SAMPLES
                end = start + FRAME_SAMPLES
                frame = audio_float32[start:end]
                frame_energy = np.sqrt(np.mean(frame**2))
                
                if frame_energy < self.energy_threshold:
                    silence_frames += 1
            
            silence_ratio = silence_frames / num_frames
            
            # Consider it a turn point if there's significant silence
            # and overall low energy (indicating end of speech)
            is_turn = (silence_ratio >= self.silence_ratio_threshold and 
                      rms < self.energy_threshold * 2)
            
            log.debug(f"Turn detection - RMS: {rms:.4f}, Silence ratio: {silence_ratio:.2f}, Turn: {is_turn}")
            
            return {"is_turn_point": is_turn}
            
        except Exception as e:
            log.error(f"Error in turn detection: {e}")
            return {"is_turn_point": False}


# Global singleton instance
_turn_detection_service = None


def get_turn_detection_service() -> TurnDetectionService:
    """Get or create the global turn detection service instance."""
    global _turn_detection_service
    if _turn_detection_service is None:
        _turn_detection_service = TurnDetectionService()
        log.info("Initialized turn detection service")
    return _turn_detection_service
