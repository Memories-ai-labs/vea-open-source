import os
from io import BytesIO
from elevenlabs.client import ElevenLabs

class GenerateSubtitles:
    """
    Uses ElevenLabs speech-to-text API to transcribe audio and generate word-level subtitles.
    The __call__ method returns the transcription as a Python dict with a "words" list of dicts.
    SRT export and pretty printing are handled by separate methods.
    """

    def __init__(
        self, 
        output_dir, 
        model_id="scribe_v1", 
        language_code=None,         # Let ElevenLabs auto-detect if None
        diarize=False, 
        tag_audio_events=True
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model_id = model_id
        self.language_code = language_code
        self.diarize = diarize
        self.tag_audio_events = tag_audio_events
        self.elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    def __call__(self, audio_path, global_start_time=0.0):
        """
        Transcribes audio using ElevenLabs STT and offsets all word timestamps.
        Returns the transcription as a dict with a 'words' list of dicts.
        """
        print(f"[INFO] Transcribing {audio_path} using ElevenLabs (model: {self.model_id})...")

        try:
            with open(audio_path, "rb") as f:
                audio_data = BytesIO(f.read())

            result = self.elevenlabs.speech_to_text.convert(
                file=audio_data,
                model_id=self.model_id,
                tag_audio_events=self.tag_audio_events,
                diarize=True
            )
        except Exception as e:
            print(f"[ERROR] ElevenLabs STT failed on {audio_path}: {e}")
            raise

        words = [w.dict() for w in getattr(result, "words", [])]

        if not words:
            print("[WARN] No words returned by ElevenLabs STT.")
            return {"words": []}

        # Apply global start time offset
        if global_start_time:
            words = self.offset_words(words, global_start_time)

        return {"words": words}

    @staticmethod
    def offset_words(words, offset):
        """
        Offsets all word start/end times by a given amount.
        """
        out = []
        for w in words:
            w = dict(w)
            if "start" in w and "end" in w:
                w["start"] = float(w["start"]) + offset
                w["end"] = float(w["end"]) + offset
            out.append(w)
        return out

    @staticmethod
    def words_to_srt_entries(words, max_words=12):
        """
        Groups words into SRT subtitle entries (default: max 12 words per phrase).
        Returns: list of dicts with start, end, text.
        """
        entries = []
        phrase = []
        for w in words:
            if w["type"] == "word":
                phrase.append(w)
                if len(phrase) >= max_words:
                    entries.append(phrase)
                    phrase = []
        if phrase:
            entries.append(phrase)

        srt_entries = []
        for p in entries:
            start = p[0]["start"]
            end = p[-1]["end"]
            text = " ".join(w["text"] for w in p)
            srt_entries.append({"start": start, "end": end, "text": text})
        return srt_entries

    @staticmethod
    def write_srt(srt_entries, srt_output_path):
        """
        Writes a list of subtitle entries to an SRT file.
        """
        with open(srt_output_path, "w", encoding="utf-8") as f:
            for idx, entry in enumerate(srt_entries, 1):
                start = GenerateSubtitles._format_timestamp(entry["start"])
                end = GenerateSubtitles._format_timestamp(entry["end"])
                text = entry["text"].strip()
                f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
        print(f"[INFO] SRT subtitles written to {srt_output_path}")

    @staticmethod
    def _format_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

    @classmethod
    def pretty_print_words(cls, words):
        """
        Returns a pretty-printed word list as a single string.
        Each word has start, end, type, speaker_id, and text.
        """
        lines = []
        for w in words:
            start = cls._format_timestamp(w.get('start', 0))
            end = cls._format_timestamp(w.get('end', 0))
            line = (
                f"{w.get('text', '').ljust(16)} "
                f"start: {start}, "
                f"end: {end}, "
                f"type: {w.get('type', '').ljust(8)}, "
                f"speaker: {w.get('speaker_id', '')}"
            )
            lines.append(line)
        return "\n".join(lines)
