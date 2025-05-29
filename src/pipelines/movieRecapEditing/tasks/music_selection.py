import os
import json
import requests
import pandas as pd
from pathlib import Path
from pydub import AudioSegment
import asyncio
from src.pipelines.movieRecapEditing.schema import ChosenMusic


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

    def fetch_instrumental_tracks(self, page_count=1):
        all_tracks = []
        for page in range(1, page_count + 1):
            params = {
                "page[size]": 100,
                "page[number]": page,
                "filter[tag]": "instrumental"
            }
            response = requests.get(self.base_url, headers=self.soundstripe_headers, params=params)
            if response.status_code != 200:
                print(f"[WARN] Soundstripe API returned {response.status_code}")
                continue
            data = response.json()
            tracks = data.get("data", [])
            all_tracks.extend(tracks)
        return pd.DataFrame(all_tracks)

    async def select_best_music(self, instrumental_tracks_df, gist_text):
        music_data_list = []
        for item in instrumental_tracks_df.to_dict("records"):
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
            "Below is the plot gist of a movie. Based on the genre, energy, and mood conveyed in the plot, "
            "select one instrumental music track from the list that best matches the overall vibe of the movie.\n\n"
            f"Movie gist:\n{gist_text}\n\n"
            "Available instrumental music tracks:\n"
            f"{json.dumps(music_data_list, indent=2)}\n\n"
            "Return your selection as the id and title of the track"
        )

        result = await asyncio.to_thread(
            self.llm.LLM_request,
            [prompt],
            {
                "response_mime_type": "application/json",
                "response_schema": ChosenMusic
            }
        )

        return result["id"], result["title"]

    def download_music(self, music_id, music_title):
        url = f"https://api.soundstripe.com/v1/songs/{music_id}"

        response = requests.get(url, headers=self.soundstripe_headers)
        audio_info = response.json()

        best_link = None
        best_duration = 0

        for item in audio_info.get("included", []):
            attr = item.get("attributes", {})
            if "instrumental" in attr.get("description", "").lower():
                duration = attr.get("duration", 0)
                if duration > best_duration:
                    best_duration = duration
                    best_link = attr.get("versions", {}).get("mp3")

        if not best_link:
            raise ValueError("No downloadable instrumental track found.")

        temp_path = os.path.join(self.output_dir, f"temp_{music_title}.mp3")
        with requests.get(best_link, stream=True) as r:
            r.raise_for_status()
            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        original_audio = AudioSegment.from_mp3(temp_path)
        one_hour_ms = 60 * 60 * 1000
        loop_count = one_hour_ms // len(original_audio) + 1
        long_audio = (original_audio * loop_count)[:one_hour_ms]

        final_path = os.path.join(self.output_dir, f"{music_title}_1hour_loop.mp3")
        long_audio.export(final_path, format="mp3")

        os.remove(temp_path)
        print(f"[INFO] 1-hour looped music saved to: {final_path}")
        return final_path
    

    async def __call__(self, plot):
        print("[INFO] Fetching instrumental music library...")
        tracks_df = self.fetch_instrumental_tracks(page_count=4)
        print(f"[INFO] Retrieved {len(tracks_df)} tracks.")

        print("[INFO] Selecting the best music based on the movie gist...")
        music_id, music_title = await self.select_best_music(tracks_df, plot)

        print(f"[INFO] Chosen Track: {music_title} (ID: {music_id})")
        return self.download_music(music_id, music_title)
