import math
import numpy as np
import pyloudnorm as pyln
from moviepy import AudioFileClip

def get_loudness(audio_clip, sample_rate=44100):
    """
    Calculate integrated LUFS loudness of an audio clip using pyloudnorm.
    """
    samples = audio_clip.to_soundarray(fps=sample_rate)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    # Ensure sample count is compatible with pyloudnorm block processing
    # pyloudnorm uses 400ms blocks (0.4 * sample_rate samples)
    block_size = int(0.4 * sample_rate)
    if len(samples) < block_size:
        # Pad short clips to minimum block size
        samples = np.pad(samples, (0, block_size - len(samples)), mode='constant')
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(samples)
    return loudness

def _average_clip_loudness(clips):
    total_energy = 0.0
    total_duration = 0.0
    for clip in clips:
        audio = getattr(clip, "audio", None)
        duration = float(getattr(clip, "duration", 0.0) or 0.0)
        if audio is None or duration <= 0:
            continue
        try:
            # Get audio samples directly without subclipping to avoid sample count mismatches
            loudness = get_loudness(audio)
        except Exception as exc:
            print(f"[WARN] Failed to measure loudness for clip: {exc}")
            continue
        total_energy += (10 ** (loudness / 10.0)) * duration
        total_duration += duration
    if total_duration <= 0 or total_energy <= 0:
        return None
    average_power = total_energy / total_duration
    return 10.0 * math.log10(max(average_power, 1e-12))

def compute_music_adjustment(clips, background_music_path: str, total_duration: float, music_volume_multiplier: float) -> tuple[float | None, float | None]:
    music_volume_multiplier_out = None
    music_gain_db = None
    if not background_music_path or total_duration <= 0:
        return music_volume_multiplier_out, music_gain_db

    reference_loudness = _average_clip_loudness(clips)

    music_clip = None
    music_segment = None
    try:
        music_clip = AudioFileClip(background_music_path)
        music_segment = music_clip.subclipped(0, min(total_duration, float(music_clip.duration or total_duration)))
        music_loudness = get_loudness(music_segment)
    except Exception as exc:
        print(f"[WARN] Failed to measure background music loudness: {exc}")
        music_loudness = None
    finally:
        if music_segment is not None:
            try:
                music_segment.close()
            except Exception:
                pass
        if music_clip is not None:
            try:
                music_clip.close()
            except Exception:
                pass

    if reference_loudness is None or music_loudness is None:
        volume_multiplier = music_volume_multiplier
    else:
        volume_multiplier = (
            10 ** ((reference_loudness - music_loudness) / 20.0)
        ) * music_volume_multiplier

    if volume_multiplier is not None and volume_multiplier > 0:
        music_volume_multiplier_out = volume_multiplier
        music_gain_db = 20.0 * math.log10(volume_multiplier)
    else:
        music_volume_multiplier_out = None
        music_gain_db = -100.0

    return music_volume_multiplier_out, music_gain_db
