import os
import json
import requests
import pandas as pd
from pathlib import Path
from pydub import AudioSegment
import asyncio
from src.pipelines.common.schema import ChosenMusicResponse
from typing import List

class MusicSelection:
    def __init__(self, llm, output_dir):
        self.llm = llm
        self.soundstripe_headers = {
            "accept": "application/json",
            "Content-Type": "application/vnd.api+json",
            "Accept": "application/vnd.api+json",
            "Authorization": f"Token {os.getenv('SOUNDSTRIPE_KEY')}"
        }
        self.base_url = "https://api.soundstripe.com/v1/songs"
        self.output_dir = output_dir

    def fetch_tracks(self, page_count=10):
        all_tracks = []
        for page in range(1, page_count + 1):
            params = {
                "page[size]": 100,
                "page[number]": page,
                # Remove filter to fetch all types, not just instrumental
            }
            response = requests.get(self.base_url, headers=self.soundstripe_headers, params=params)
            if response.status_code != 200:
                print(f"[WARN] Soundstripe API returned {response.status_code}")
                continue
            data = response.json()
            tracks = data.get("data", [])
            all_tracks.extend(tracks)
        return pd.DataFrame(all_tracks)

    async def select_best_music(self, tracks_df, media_indexing_json, user_prompt):
        # Extract music data for LLM
        music_data_list = []
        for item in tracks_df.to_dict("records"):
            attributes = item.get("attributes", {})
            music_data_list.append({
                "id": item["id"],
                "title": attributes.get("title", ""),
                "genre": attributes.get("genre", ""),
                "mood": attributes.get("mood", ""),
                "bpm": attributes.get("bpm", ""),
                "description": attributes.get("description", ""),
            })

        prompt = (
            "You are helping to select the best music track for a video editing project.\n\n"
            "Below is the media indexing JSON, which summarizes all available plot, character, scene, and artistic details of the video, and below that is the user's prompt.\n"
            "Choose the TOP 5 music tracks from the list that will best match the content, emotion, and energy of the media as well as the user's preferences for the final edit. "
            "Rank them in order from best to least.\n\n"
            "Media indexing JSON:\n"
            f"{json.dumps(media_indexing_json, indent=2, ensure_ascii=False)}\n\n"
            "User prompt: \n"
            f"{user_prompt}\n\n"
            "Available music tracks:\n"
            f"{json.dumps(music_data_list, indent=2, ensure_ascii=False)}\n\n"
            "Return your selection as a JSON list with the following fields for each track: id and title."
        )


        result = await asyncio.to_thread(
            self.llm.LLM_request,
            [prompt],
            ChosenMusicResponse
        )
        return result

    def download_music(self, music_id, music_title):
        url = f"https://api.soundstripe.com/v1/songs/{music_id}"

        response = requests.get(url, headers=self.soundstripe_headers)
        audio_info = response.json()

        best_link = None
        best_duration = 0

        # Try to find the longest available version (prefer instrumental if found, but not required)
        for item in audio_info.get("included", []):
            attr = item.get("attributes", {})
            duration = attr.get("duration", 0)
            link = attr.get("versions", {}).get("mp3")
            if link and duration > best_duration:
                best_duration = duration
                best_link = link

        if not best_link:
            raise ValueError("No downloadable track found.")

        safe_title = "".join(c if c.isalnum() else "_" for c in music_title)[:60]
        temp_path = os.path.join(self.output_dir, f"temp_{safe_title}.mp3")
        with requests.get(best_link, stream=True) as r:
            r.raise_for_status()
            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        original_audio = AudioSegment.from_mp3(temp_path)
        one_hour_ms = 60 * 60 * 1000
        loop_count = one_hour_ms // len(original_audio) + 1
        long_audio = (original_audio * loop_count)[:one_hour_ms]

        final_path = os.path.join(self.output_dir, f"{safe_title}_1hour_loop.mp3")
        long_audio.export(final_path, format="mp3")

        os.remove(temp_path)
        print(f"[INFO] 1-hour looped music saved to: {final_path}")
        return final_path

    async def __call__(self, media_indexing_json, user_prompt=""):
        print("[INFO] Fetching music library from Soundstripe...")
        tracks_df = self.fetch_tracks(page_count=10)
        print(f"[INFO] Retrieved {len(tracks_df)} tracks.")

        print("[INFO] Selecting the top 5 music tracks based on media indexing JSON...")
        top_tracks = await self.select_best_music(tracks_df, media_indexing_json, user_prompt)

        for i, track in enumerate(top_tracks):
            music_id, music_title = track.get("id"), track.get("title")
            try:
                print(f"[INFO] Attempting to download Track #{i+1}: {music_title} (ID: {music_id})")
                return self.download_music(music_id, music_title)
            except Exception as e:
                print(f"[WARN] Failed to download {music_title} (ID: {music_id}): {e}")
                continue

        raise RuntimeError("Failed to download any of the top 5 recommended tracks from Soundstripe.")

