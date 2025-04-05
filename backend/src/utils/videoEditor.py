import os
import subprocess
from pydub.utils import mediainfo
from tempfile import TemporaryDirectory


def get_audio_duration(audio_path):
    info = mediainfo(audio_path)
    return float(info['duration'])


def center_trim_video(video_path, duration, output_path):
    """
    Trim the video to match the audio duration, centered. If the video is shorter than
    the audio, slow it down to match the audio length.
    """
    # Get full video duration
    info = mediainfo(video_path)
    full_duration = float(info['duration'])

    if duration <= full_duration:
        # Normal case: center trim
        start = (full_duration - duration) / 2
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(duration),
            "-i", video_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            output_path
        ], check=True)
    else:
        # Stretch video to match audio duration
        speed_factor = full_duration / duration
        print(f"[WARN] Video too short. Slowing down by factor {1/speed_factor:.2f} to match audio.")

        # Apply PTS adjustment to stretch duration
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter_complex", f"[0:v]setpts={1/speed_factor}*PTS[v]",
            "-map", "[v]",
            "-c:v", "libx264",
            "-an",  # No audio
            output_path
        ], check=True)


def replace_audio(video_path, audio_path, output_path):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path
    ], check=True)


def mix_music_with_audio(final_video_path, music_path, output_path):
    # Get final video duration
    info = mediainfo(final_video_path)
    video_duration = float(info["duration"])

    with TemporaryDirectory() as tmp:
        trimmed_music = os.path.join(tmp, "trimmed_music.m4a")

        # Trim music to video duration
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", "0", "-t", str(video_duration),
            "-i", music_path,
            "-af", "volume=0.3",
            "-c:a", "aac",
            trimmed_music
        ], check=True)

        # Final mix (narration + music)
        subprocess.run([
            "ffmpeg", "-y",
            "-i", final_video_path,
            "-i", trimmed_music,
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2",
            "-map", "0:v:0",
            "-c:v", "copy",
            "-c:a", "aac",
            output_path
        ], check=True)


async def create_final_video(
    clips,
    work_dir,
    chosen_videos_dir,
    narration_dir,
    background_music_path,
    output_path="final_recap.mp4"
):
    try:
        final_recap_dir = os.path.join(work_dir, "final_recap")
        os.makedirs(final_recap_dir, exist_ok=True)
        concat_list_file = os.path.join(final_recap_dir, "concat.txt")
        final_clips = []

        # Process each clip
        for clip in sorted(clips, key=lambda c: int(c["id"])):
            clip_id = clip["id"]
            video_path = os.path.join(chosen_videos_dir, f"{clip_id}.mp4")
            audio_path = os.path.join(narration_dir, f"{clip_id}.mp3")

            trimmed_video = os.path.join(final_recap_dir, f"{clip_id}_trimmed.mp4")
            replaced_audio = os.path.join(final_recap_dir, f"{clip_id}_narrated.mp4")

            audio_duration = get_audio_duration(audio_path)

            center_trim_video(video_path, audio_duration, trimmed_video)
            replace_audio(trimmed_video, audio_path, replaced_audio)

            final_clips.append(replaced_audio)
        
         # Write concat list
        with open(concat_list_file, "w") as f:
            for clip_path in final_clips:
                f.write(f"file '{clip_path}'\n")

            # Concatenate all narrated clips
        intermediate_output = os.path.join(final_recap_dir, "combined_narrated.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_file,
            "-c:v", "libx264",
            "-c:a", "aac",
            intermediate_output
        ], check=True)

            # Mix with background music
        mix_music_with_audio(intermediate_output, background_music_path, output_path)
        print(f"[INFO] Final recap video created: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to create final recap video: {e}")
        print(audio_path)
        print(video_path)
