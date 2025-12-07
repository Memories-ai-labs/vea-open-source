"""
Memories.ai API client for video understanding.

This module provides a client for interacting with the Memories.ai API,
which handles video upload, indexing, search, and chat capabilities.
"""

import asyncio
import aiohttp
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from lib.utils.metrics_collector import metrics_collector


@dataclass
class MemoriesReference:
    """A timestamped reference from a Memories.ai Chat response."""
    video_no: str
    video_name: str
    timestamp: str  # HH:MM:SS format
    annotation_type: str
    description: str
    duration: Optional[float] = None


@dataclass
class ChatResponse:
    """Parsed response from Memories.ai Chat API."""
    text: str
    references: List[MemoriesReference] = field(default_factory=list)
    ref_items: List[Dict] = field(default_factory=list)  # Raw refItems from thinkings[].refs
    session_id: Optional[str] = None
    thinking: Optional[str] = None
    raw_response: Optional[Dict] = None


@dataclass
class VideoStatus:
    """Status of an uploaded video."""
    video_no: str
    video_name: str
    status: str  # PENDING, PARSE, FAILED, etc.
    duration: Optional[float] = None
    raw_response: Optional[Dict] = None


class MemoriesAiManager:
    """
    Client for Memories.ai video understanding APIs.

    Provides methods for:
    - Uploading videos (by URL or file)
    - Checking video processing status
    - Searching indexed videos
    - Chatting with video content

    Example:
        manager = MemoriesAiManager(api_key="your-key")
        video_no = await manager.upload_video_url("https://example.com/video.mp4")
        await manager.wait_for_ready(video_no)
        response = await manager.chat([video_no], "Summarize this video")
    """

    API_HOST = "https://api.memories.ai"
    CAPTION_HOST = "https://security.memories.ai"

    def __init__(self, api_key: Optional[str] = None, debug: bool = False):
        """
        Initialize the Memories.ai client.

        Args:
            api_key: Memories.ai API key. If not provided, reads from
                     MEMORIES_API_KEY environment variable.
            debug: If True, logs full raw API responses.
        """
        self.api_key = api_key or os.environ.get("MEMORIES_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Memories.ai API key required. Pass api_key parameter or set MEMORIES_API_KEY env var."
            )
        self.debug = debug
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": self.api_key,
                    "Content-Type": "application/json",
                }
            )
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        host: Optional[str] = None,
    ) -> Dict:
        """Make an API request."""
        session = await self._get_session()
        url = f"{host or self.API_HOST}{endpoint}"

        async with session.request(method, url, json=data) as response:
            result = await response.json()

            if self.debug:
                print(f"[MEMORIES DEBUG] {method} {endpoint}")
                print(f"[MEMORIES DEBUG] Request: {json.dumps(data, indent=2)}")
                print(f"[MEMORIES DEBUG] Response: {json.dumps(result, indent=2)}")

            return result

    # -------------------------------------------------------------------------
    # Upload API
    # -------------------------------------------------------------------------

    async def upload_video_url(
        self,
        video_url: str,
        callback_url: Optional[str] = None,
        unique_id: str = "default",
    ) -> str:
        """
        Upload a video by URL for indexing.

        Args:
            video_url: Public URL of the video to upload.
            callback_url: Optional webhook URL for completion notification.
            unique_id: Tenant/group identifier for multi-tenant setups.

        Returns:
            videoNo identifier for the uploaded video.
        """
        print(f"[MEMORIES] Uploading video from URL: {video_url[:80]}...")

        data = {
            "url": video_url,
            "unique_id": unique_id,
        }
        if callback_url:
            data["callback"] = callback_url

        with metrics_collector.track_step("memories_upload"):
            response = await self._request("POST", "/serve/api/v1/upload_url", data)

        if not response.get("success"):
            raise RuntimeError(f"[MEMORIES] Upload failed: {response.get('msg', 'Unknown error')}")

        video_no = response["data"]["videoNo"]
        video_name = response["data"].get("videoName", "unknown")
        print(f"[MEMORIES] Upload successful - videoNo: {video_no}, name: {video_name}")

        return video_no

    async def upload_video_file(
        self,
        file_path: str,
        callback_url: Optional[str] = None,
        unique_id: str = "default",
    ) -> str:
        """
        Upload a video file for indexing.

        Args:
            file_path: Local path to the video file.
            callback_url: Optional webhook URL for completion notification.
            unique_id: Tenant/group identifier.

        Returns:
            videoNo identifier for the uploaded video.
        """
        print(f"[MEMORIES] Uploading video file: {file_path}")

        url = f"{self.API_HOST}/serve/api/v1/upload"

        # Determine content type based on file extension
        ext = Path(file_path).suffix.lower()
        content_types = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
        }
        content_type = content_types.get(ext, "video/mp4")

        # Get file size for timeout calculation
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"[MEMORIES] File size: {file_size_mb:.1f} MB, content-type: {content_type}")

        # Use longer timeout for large files (5 min base + 1 min per 100MB)
        timeout_seconds = 300 + int(file_size_mb / 100) * 60
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        # Only Authorization header - let aiohttp set Content-Type for multipart
        headers = {"Authorization": self.api_key}

        # Use context manager to ensure file handle is closed
        with open(file_path, "rb") as video_file:
            # Build FormData matching official docs format:
            # files = {"file": (filename, file_handle, content_type)}
            # data = {"unique_id": ..., "callback": ...}
            data = aiohttp.FormData()
            data.add_field("unique_id", unique_id)
            if callback_url:
                data.add_field("callback", callback_url)

            # Add file field with streaming (don't read entire file into memory)
            data.add_field(
                "file",
                video_file,
                filename=Path(file_path).name,
                content_type=content_type,
            )

            with metrics_collector.track_step("memories_upload_file"):
                async with aiohttp.ClientSession() as upload_session:
                    async with upload_session.post(url, data=data, headers=headers, timeout=timeout) as response:
                        response_text = await response.text()
                        if self.debug:
                            print(f"[MEMORIES DEBUG] Upload response status: {response.status}")
                            print(f"[MEMORIES DEBUG] Upload response: {response_text}")

                        try:
                            result = json.loads(response_text)
                        except json.JSONDecodeError:
                            raise RuntimeError(f"[MEMORIES] Upload failed - invalid JSON response: {response_text[:500]}")

        if not result.get("success"):
            raise RuntimeError(f"[MEMORIES] Upload failed: {result.get('msg', 'Unknown error')}")

        video_no = result["data"]["videoNo"]
        video_name = result["data"].get("videoName", "unknown")
        print(f"[MEMORIES] Upload successful - videoNo: {video_no}, name: {video_name}")

        return video_no

    # -------------------------------------------------------------------------
    # Status API
    # -------------------------------------------------------------------------

    async def find_video_by_name(self, video_name: str) -> Optional[VideoStatus]:
        """
        Check if a video with this name already exists.

        Args:
            video_name: Name to search for (exact match).

        Returns:
            VideoStatus if found, None otherwise.
        """
        response = await self._request(
            "POST",
            "/serve/api/v1/list_videos",
            {"video_name": video_name},
        )

        if not response.get("success"):
            return None

        videos = response.get("data", {}).get("videos", [])
        if not videos:
            return None

        # Return the first match
        video = videos[0]
        # API returns snake_case (video_no) not camelCase (videoNo)
        status = VideoStatus(
            video_no=video.get("video_no", "") or video.get("videoNo", ""),
            video_name=video.get("video_name", "") or video.get("videoName", ""),
            status=video.get("status", "UNKNOWN"),
            duration=video.get("duration"),
            raw_response=response if self.debug else None,
        )

        print(f"[MEMORIES] Found existing video: {video_name} -> {status.video_no} (status: {status.status})")
        return status

    async def get_video_status(self, video_no: str) -> VideoStatus:
        """
        Check the processing status of an uploaded video.

        Args:
            video_no: The videoNo identifier from upload.

        Returns:
            VideoStatus with current processing state.
        """
        response = await self._request(
            "POST",
            "/serve/api/v1/list_videos",
            {"video_no": video_no},
        )

        if not response.get("success"):
            raise RuntimeError(f"[MEMORIES] Status check failed: {response.get('msg')}")

        videos = response.get("data", {}).get("videos", [])
        if not videos:
            return VideoStatus(
                video_no=video_no,
                video_name="unknown",
                status="NOT_FOUND",
                raw_response=response,
            )

        video = videos[0]
        # API returns snake_case (video_no) not camelCase (videoNo)
        status = VideoStatus(
            video_no=video.get("video_no", "") or video.get("videoNo", "") or video_no,
            video_name=video.get("video_name", "") or video.get("videoName", "") or "unknown",
            status=video.get("status", "UNKNOWN"),
            duration=video.get("duration"),
            raw_response=response if self.debug else None,
        )

        print(f"[MEMORIES] Video {video_no} status: {status.status}")
        return status

    async def wait_for_ready(
        self,
        video_no: str,
        poll_interval: float = 30.0,  # Increased to avoid rate limits
        timeout: float = 600.0,
    ) -> VideoStatus:
        """
        Wait for a video to finish processing.

        Args:
            video_no: The videoNo identifier.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait.

        Returns:
            VideoStatus when ready.

        Raises:
            TimeoutError: If video doesn't become ready within timeout.
            RuntimeError: If video processing fails.
        """
        print(f"[MEMORIES] Waiting for video {video_no} to be ready...")
        start_time = time.time()
        last_status = None
        check_count = 0
        consecutive_errors = 0

        while True:
            # Temporarily disable debug for status polling (too noisy)
            old_debug = self.debug
            self.debug = False

            try:
                status = await self.get_video_status(video_no)
                consecutive_errors = 0  # Reset on success
            except RuntimeError as e:
                self.debug = old_debug
                consecutive_errors += 1
                error_msg = str(e)

                # Handle rate limits with exponential backoff
                if "exceeded the limit" in error_msg.lower() or "rate" in error_msg.lower():
                    backoff = min(60, poll_interval * (2 ** consecutive_errors))
                    print(f"[MEMORIES] Rate limited, backing off for {backoff:.0f}s...")
                    await asyncio.sleep(backoff)
                    continue
                elif consecutive_errors < 3:
                    print(f"[MEMORIES] Status check error ({consecutive_errors}/3): {error_msg}")
                    await asyncio.sleep(poll_interval)
                    continue
                else:
                    raise
            finally:
                self.debug = old_debug

            check_count += 1

            if status.status == "PARSE":
                print(f"[MEMORIES] Video {video_no} is ready! (took {int(time.time() - start_time)}s)")
                return status
            elif status.status == "FAILED":
                raise RuntimeError(f"[MEMORIES] Video processing failed: {video_no}")
            elif status.status == "NOT_FOUND":
                raise RuntimeError(f"[MEMORIES] Video not found: {video_no}")

            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"[MEMORIES] Timeout waiting for video {video_no} after {timeout}s"
                )

            # Only log on status change or every 60 seconds (2 checks at 30s interval)
            if status.status != last_status or check_count % 2 == 0:
                print(f"[MEMORIES] Status: {status.status}, elapsed: {int(elapsed)}s...")
                last_status = status.status

            await asyncio.sleep(poll_interval)

    # -------------------------------------------------------------------------
    # Search API
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        search_type: str = "BY_VIDEO",
        video_nos: Optional[List[str]] = None,
        unique_id: str = "default",
    ) -> Dict:
        """
        Search indexed videos.

        Args:
            query: Search query text.
            search_type: One of BY_VIDEO, BY_AUDIO, BY_CLIP.
            video_nos: Optional list of video IDs to search within.
            unique_id: Tenant/group identifier.

        Returns:
            Search results with timestamps and relevance scores.
        """
        print(f"[MEMORIES] Searching: '{query[:50]}...' (type={search_type})")

        data = {
            "search_param": query,
            "search_type": search_type,
            "unique_id": unique_id,
        }
        if video_nos:
            data["video_nos"] = video_nos

        with metrics_collector.track_step("memories_search"):
            response = await self._request("POST", "/serve/api/v1/search", data)

        if not response.get("success"):
            raise RuntimeError(f"[MEMORIES] Search failed: {response.get('msg')}")

        results = response.get("data", {})
        result_count = len(results.get("items", []))
        print(f"[MEMORIES] Search returned {result_count} results")

        if self.debug:
            print(f"[MEMORIES DEBUG] Search response: {json.dumps(response, indent=2)}")

        return results

    # -------------------------------------------------------------------------
    # Chat API
    # -------------------------------------------------------------------------

    async def chat(
        self,
        video_nos: List[str],
        prompt: str,
        session_id: Optional[str] = None,
        unique_id: str = "default",
    ) -> ChatResponse:
        """
        Chat with indexed videos.

        Args:
            video_nos: List of videoNo identifiers to query.
            prompt: Natural language prompt/question.
            session_id: Optional session ID for conversation continuity.
            unique_id: Tenant/group identifier.

        Returns:
            ChatResponse with text, references, and metadata.
        """
        print(f"[MEMORIES] Chat request - videos: {video_nos}")
        print(f"[MEMORIES] Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        data = {
            "video_nos": video_nos,
            "prompt": prompt,
            "unique_id": unique_id,
        }
        if session_id:
            data["session_id"] = session_id

        with metrics_collector.track_step("memories_chat"):
            response = await self._request("POST", "/serve/api/v1/chat", data)

        if self.debug:
            print(f"[MEMORIES DEBUG] Chat response: {json.dumps(response, indent=2)}")

        # Check for errors (response has "success": true/false and "code": "0000" for success)
        if not response.get("success", False):
            error_msg = response.get("msg", "Unknown error")
            raise RuntimeError(f"[MEMORIES] Chat failed: {error_msg}")

        # Parse the response
        parsed = self._parse_chat_response(response)

        text_preview = parsed.text[:200] if parsed.text else "(no text)"
        print(f"[MEMORIES] Chat response: {text_preview}...")
        print(f"[MEMORIES] References: {len(parsed.references)} timestamps, {len(parsed.ref_items)} ref_items found")

        return parsed

    async def chat_stream(
        self,
        video_nos: List[str],
        prompt: str,
        session_id: Optional[str] = None,
        unique_id: str = "default",
    ):
        """
        Stream chat responses from indexed videos.

        Yields chunks of the response as they arrive.

        Args:
            video_nos: List of videoNo identifiers.
            prompt: Natural language prompt.
            session_id: Optional session ID.
            unique_id: Tenant/group identifier.

        Yields:
            Dict chunks from the streaming response.
        """
        print(f"[MEMORIES] Chat stream request - videos: {video_nos}")

        session = await self._get_session()
        url = f"{self.API_HOST}/serve/api/v1/chat_stream"

        data = {
            "video_nos": video_nos,
            "prompt": prompt,
            "unique_id": unique_id,
        }
        if session_id:
            data["session_id"] = session_id

        async with session.post(url, json=data) as response:
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data:"):
                    try:
                        chunk = json.loads(line[5:].strip())
                        yield chunk
                    except json.JSONDecodeError:
                        continue

    def _parse_chat_response(self, response: Dict) -> ChatResponse:
        """Parse raw chat API response into structured ChatResponse."""
        text_parts = []
        references = []
        ref_items = []  # Raw refItems from thinkings[].refs
        thinking = None
        session_id = None

        # Handle streaming-style response with multiple message types
        data = response.get("data", response)

        if isinstance(data, list):
            for item in data:
                msg_type = item.get("type")
                if msg_type == "content":
                    content = item.get("content", {})
                    if content.get("role") == "assistant":
                        text_parts.append(content.get("content", ""))
                elif msg_type == "thinking":
                    thinking = item.get("content", "")
                elif msg_type == "ref":
                    refs = item.get("refItems", [])
                    ref_items.extend(refs)  # Collect raw refItems
                    for ref in refs:
                        references.append(MemoriesReference(
                            video_no=item.get("video_no", ""),
                            video_name=item.get("video_name", ""),
                            timestamp=ref.get("timestamp", "00:00:00"),
                            annotation_type=ref.get("annotation_type", ""),
                            description=ref.get("text", ""),
                        ))
                elif msg_type == "session":
                    session_id = item.get("session_id")
        elif isinstance(data, dict):
            # Standard response format from notebook:
            # {"role": "ASSISTANT", "content": "...", "thinkings": [...], "session_id": "..."}
            text_parts.append(data.get("text", data.get("content", "")))
            session_id = data.get("session_id")

            # Parse thinkings if present - extract both text and refs
            # Structure: thinkings[].refs[].refItems[] contains timestamped data
            thinkings = data.get("thinkings", [])
            if thinkings:
                thinking_texts = []
                for t in thinkings:
                    if isinstance(t, dict):
                        title = t.get("title", "")
                        content = t.get("content", "")
                        thinking_texts.append(f"{title}: {content}" if title else content)

                        # Extract refItems from refs (the gold data!)
                        for ref in t.get("refs", []):
                            if isinstance(ref, dict):
                                items = ref.get("refItems", [])
                                ref_items.extend(items)
                    elif isinstance(t, str):
                        thinking_texts.append(t)
                thinking = "\n\n".join(thinking_texts)

            # Parse references if present
            for ref in data.get("references", []):
                references.append(MemoriesReference(
                    video_no=ref.get("video_no", ""),
                    video_name=ref.get("video_name", ""),
                    timestamp=ref.get("timestamp", "00:00:00"),
                    annotation_type=ref.get("annotation_type", ""),
                    description=ref.get("text", ref.get("description", "")),
                ))

        return ChatResponse(
            text="\n".join(text_parts),
            references=references,
            ref_items=ref_items,
            session_id=session_id,
            thinking=thinking,
            raw_response=response if self.debug else None,
        )

    # -------------------------------------------------------------------------
    # Video Caption API (async with callbacks)
    # -------------------------------------------------------------------------

    async def caption_video_url(
        self,
        video_url: str,
        user_prompt: str,
        callback_url: str,
        system_prompt: str = "You are a helpful video analyst.",
        thinking: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """
        Start async video captioning via URL.

        This is a direct LLM video analysis (like Gemini), not RAG search.
        Results are delivered asynchronously via callback URL.

        Args:
            video_url: Public URL of the video.
            user_prompt: Analysis instructions.
            callback_url: Webhook URL for results (required).
            system_prompt: System context.
            thinking: Enable reasoning mode for deeper analysis.
            reasoning_effort: Effort level 1-10 (requires thinking=True).

        Returns:
            task_id for tracking the async operation.
        """
        print(f"[MEMORIES] Caption video (URL) request: {video_url[:60]}...")

        data = {
            "video_url": video_url,
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "callback": callback_url,
        }
        if thinking:
            data["thinking"] = True
            if reasoning_effort:
                data["reasoning_effort"] = reasoning_effort

        response = await self._request(
            "POST",
            "/v1/understand/upload",
            data,
            host=self.CAPTION_HOST,
        )

        # Response code is 0 for success (not 200)
        if response.get("code") != 0:
            raise RuntimeError(f"[MEMORIES] Caption request failed: {response}")

        task_id = response.get("data", {}).get("task_id", "unknown")
        print(f"[MEMORIES] Caption task started: {task_id}")

        return task_id

    async def caption_video_file(
        self,
        file_path: str,
        user_prompt: str,
        callback_url: str,
        system_prompt: str = "You are a helpful video analyst.",
        thinking: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """
        Start async video captioning via file upload.

        This is a direct LLM video analysis (like Gemini), not RAG search.
        Results are delivered asynchronously via callback URL.

        Args:
            file_path: Local path to video file.
            user_prompt: Analysis instructions.
            callback_url: Webhook URL for results (required).
            system_prompt: System context.
            thinking: Enable reasoning mode for deeper analysis.
            reasoning_effort: Effort level 1-10 (requires thinking=True).

        Returns:
            task_id for tracking the async operation.
        """
        print(f"[MEMORIES] Caption video (file) request: {file_path}")

        url = f"{self.CAPTION_HOST}/v1/understand/uploadFile"
        headers = {"Authorization": self.api_key}

        # Build request JSON
        req_data = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "callback": callback_url,
        }
        if thinking:
            req_data["thinking"] = True
            if reasoning_effort:
                req_data["reasoning_effort"] = reasoning_effort

        # Determine content type
        ext = Path(file_path).suffix.lower()
        content_types = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
        }
        content_type = content_types.get(ext, "video/mp4")

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        timeout_seconds = 300 + int(file_size_mb / 50) * 60
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        print(f"[MEMORIES] Uploading {file_size_mb:.1f}MB for captioning...")

        # Use context manager to ensure file handle is closed
        with open(file_path, "rb") as video_file:
            # Build multipart form data per docs:
            # files = [("req", (filename, json, "application/json")), ("files", (filename, file, content_type))]
            data = aiohttp.FormData()
            data.add_field(
                "req",
                json.dumps(req_data),
                filename="req.json",
                content_type="application/json",
            )
            data.add_field(
                "files",
                video_file,
                filename=Path(file_path).name,
                content_type=content_type,
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers, timeout=timeout) as response:
                    response_text = await response.text()
                    if self.debug:
                        print(f"[MEMORIES DEBUG] Caption upload response: {response_text}")

                    try:
                        result = json.loads(response_text)
                    except json.JSONDecodeError:
                        raise RuntimeError(f"[MEMORIES] Caption upload failed - invalid JSON: {response_text[:500]}")

        if result.get("code") != 0:
            raise RuntimeError(f"[MEMORIES] Caption request failed: {result}")

        task_id = result.get("data", {}).get("task_id", "unknown")
        print(f"[MEMORIES] Caption task started: {task_id}")

        return task_id

    # Keep old method name as alias for backwards compatibility
    async def caption_video(
        self,
        video_url: str,
        user_prompt: str,
        system_prompt: str = "You are a helpful video analyst.",
        callback_url: Optional[str] = None,
        thinking: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Deprecated: Use caption_video_url instead."""
        if not callback_url:
            raise ValueError("callback_url is required for Video Caption API")
        return await self.caption_video_url(
            video_url=video_url,
            user_prompt=user_prompt,
            callback_url=callback_url,
            system_prompt=system_prompt,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
        )

    async def caption_image(
        self,
        image_url: str,
        user_prompt: str,
        system_prompt: str = "You are a helpful image analyst.",
    ) -> str:
        """
        Caption an image (synchronous).

        Args:
            image_url: Public URL of the image.
            user_prompt: Analysis instructions.
            system_prompt: System context.

        Returns:
            Caption text.
        """
        print(f"[MEMORIES] Caption image: {image_url[:60]}...")

        data = {
            "url": image_url,
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
        }

        response = await self._request(
            "POST",
            "/v1/understand/uploadImg",
            data,
            host=self.CAPTION_HOST,
        )

        if response.get("code") != 0:
            raise RuntimeError(f"[MEMORIES] Image caption failed: {response}")

        text = response.get("text", "")
        print(f"[MEMORIES] Image caption: {text[:100]}...")

        return text


# Convenience function for creating manager from environment
def create_memories_manager(debug: bool = False) -> MemoriesAiManager:
    """
    Create MemoriesAiManager from environment variables.

    Reads MEMORIES_API_KEY from environment.
    Sets debug=True if ENV=development.
    """
    debug = debug or os.environ.get("ENV") == "development"
    return MemoriesAiManager(debug=debug)
