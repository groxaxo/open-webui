import { MicVAD } from '@ricky0123/vad-web';

/**
 * Utility for integrating Silero VAD (Voice Activity Detection) with web audio
 * Uses the @ricky0123/vad-web library which implements Silero VAD in the browser
 */

export interface VADConfig {
	/**
	 * Threshold for voice activity detection (0-1)
	 * Lower values are more sensitive
	 * Default: 0.5
	 */
	positiveSpeechThreshold?: number;

	/**
	 * Threshold for negative (non-speech) detection (0-1)
	 * Higher values are more strict about silence
	 * Default: 0.35
	 */
	negativeSpeechThreshold?: number;

	/**
	 * Redemption time in milliseconds - time to wait after silence
	 * before triggering speech end event
	 * Default: 800
	 */
	redemptionMs?: number;

	/**
	 * Pre-speech pad time in milliseconds
	 * Time to include before speech start
	 * Default: 300
	 */
	preSpeechPadMs?: number;

	/**
	 * Minimum speech duration in milliseconds
	 * Segments shorter than this are discarded as misfires
	 * Default: 500
	 */
	minSpeechMs?: number;

	/**
	 * If true, when pausing VAD it may trigger onSpeechEnd
	 * Default: true
	 */
	submitUserSpeechOnPause?: boolean;
}

export interface VADCallbacks {
	onSpeechStart?: () => void;
	onSpeechEnd?: (audio: Float32Array) => void;
	onVADMisfire?: () => void;
}

/**
 * Creates and initializes a Silero VAD instance
 * @param config VAD configuration options
 * @param callbacks Event callbacks for speech detection
 * @returns Promise that resolves to the VAD instance
 */
export async function createVAD(
	config: VADConfig = {},
	callbacks: VADCallbacks = {}
): Promise<any> {
	try {
		const vad = await MicVAD.new({
			// Audio configuration - use millisecond-based parameters
			positiveSpeechThreshold: config.positiveSpeechThreshold ?? 0.5,
			negativeSpeechThreshold: config.negativeSpeechThreshold ?? 0.35,
			redemptionMs: config.redemptionMs ?? 800, // Wait 800ms after silence
			preSpeechPadMs: config.preSpeechPadMs ?? 300, // Include 300ms before speech
			minSpeechMs: config.minSpeechMs ?? 500, // Discard segments shorter than 500ms
			submitUserSpeechOnPause: config.submitUserSpeechOnPause ?? true,

			// Callbacks
			onSpeechStart: () => {
				console.log('VAD: Speech detected');
				callbacks.onSpeechStart?.();
			},
			onSpeechEnd: (audio: Float32Array) => {
				console.log('VAD: Speech ended');
				callbacks.onSpeechEnd?.(audio);
			},
			onVADMisfire: () => {
				console.log('VAD: Misfire detected');
				callbacks.onVADMisfire?.();
			}
		});

		return vad;
	} catch (error) {
		console.error('Failed to initialize VAD:', error);
		throw error;
	}
}

/**
 * Converts Float32Array audio from VAD to a Blob
 * @param audio Float32Array audio data from VAD
 * @param sampleRate Sample rate (default: 16000)
 * @returns Audio blob
 */
export function audioToBlob(audio: Float32Array, sampleRate: number = 16000): Blob {
	// Convert Float32Array to Int16Array for WAV format
	const int16Array = new Int16Array(audio.length);
	for (let i = 0; i < audio.length; i++) {
		const s = Math.max(-1, Math.min(1, audio[i]));
		int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
	}

	// Create WAV file
	const wavBuffer = createWavBuffer(int16Array, sampleRate);
	return new Blob([wavBuffer], { type: 'audio/wav' });
}

/**
 * Creates a WAV file buffer from audio data
 * @param samples Int16Array audio samples
 * @param sampleRate Sample rate
 * @returns ArrayBuffer containing WAV file
 */
function createWavBuffer(samples: Int16Array, sampleRate: number): ArrayBuffer {
	const buffer = new ArrayBuffer(44 + samples.length * 2);
	const view = new DataView(buffer);

	// Write WAV header
	writeString(view, 0, 'RIFF');
	view.setUint32(4, 36 + samples.length * 2, true);
	writeString(view, 8, 'WAVE');
	writeString(view, 12, 'fmt ');
	view.setUint32(16, 16, true); // fmt chunk size
	view.setUint16(20, 1, true); // PCM format
	view.setUint16(22, 1, true); // mono
	view.setUint32(24, sampleRate, true);
	view.setUint32(28, sampleRate * 2, true); // byte rate
	view.setUint16(32, 2, true); // block align
	view.setUint16(34, 16, true); // bits per sample
	writeString(view, 36, 'data');
	view.setUint32(40, samples.length * 2, true);

	// Write audio data
	for (let i = 0; i < samples.length; i++) {
		view.setInt16(44 + i * 2, samples[i], true);
	}

	return buffer;
}

function writeString(view: DataView, offset: number, string: string): void {
	for (let i = 0; i < string.length; i++) {
		view.setUint8(offset + i, string.charCodeAt(i));
	}
}
