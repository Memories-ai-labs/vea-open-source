import os
import tempfile
from pathlib import Path
from lib.llm.GeminiGenaiManager import GeminiGenaiManager
from lib.oss.gcp_oss import GoogleCloudStorage
from lib.oss.auth import credentials_from_file
from src.config import CREDENTIAL_PATH, BUCKET_NAME
from src.pipelines.qualityAnalysis.schema import QualityAssessmentResult, BrandSafetyScores


class QualityAssessmentPipeline:
    def __init__(self, cloud_storage_video_path: str, ground_truth_text: str, user_prompt: str):
        self.video_blob_path = cloud_storage_video_path
        self.media_name = os.path.basename(cloud_storage_video_path)
        self.media_base_name = os.path.splitext(self.media_name)[0]
        self.gcs_client = GoogleCloudStorage(credentials=credentials_from_file(CREDENTIAL_PATH))
        self.llm = GeminiGenaiManager(model="gemini-2.5-flash")
        self.ground_truth = ground_truth_text
        self.user_prompt = user_prompt
        self.workdir = tempfile.mkdtemp()
        print(f"[DEBUG] Initialized QualityAssessmentPipeline with media: {self.media_name}")

    def _download_video(self) -> Path:
        local_path = os.path.join(self.workdir, self.media_name)
        print(f"[DEBUG] Downloading video from GCS path: {self.video_blob_path} to {local_path}")
        self.gcs_client.download_files(BUCKET_NAME, self.video_blob_path, local_path)
        return Path(local_path)

    def _score_plot_fidelity(self, video_path: Path) -> int:
        print("[DEBUG] Starting plot fidelity scoring...")
        if not self.ground_truth or not self.ground_truth.strip():
            print("[INFO] No ground truth provided. Returning default score of 3 for plot fidelity.")
            return 3

        prompt = (
            "You are evaluating how closely an edited video aligns with a provided ground truth reference.\n"
            "The ground truth may be any factual or narrative source relevant to the video, such as:\n"
            "- An official movie or TV episode synopsis\n"
            "- A news article about an event\n"
            "- A travel itinerary or sports game record\n"
            "- Any other reliable description of what the video is expected to portray\n\n"
            "Compare the video’s content to the ground truth, and rate how accurately and completely the video reflects the information.\n"
            "Assess whether the main events, sequence, facts, and emphasis match, and whether anything important is missing, altered, or misrepresented.\n\n"
            f"Ground truth:\n{self.ground_truth}\n\n"
            "Scoring rubric (return only an integer from 1 to 5):\n"
            "- 5: Fully faithful and complete — all key elements from the ground truth are covered with high accuracy.\n"
            "- 4: Largely faithful — most important elements are present; minor omissions or minor rewording.\n"
            "- 3: Moderately faithful — about half the key points are included or some are unclear/misrepresented.\n"
            "- 2: Weak fidelity — major parts are missing or wrong; viewer may be misled.\n"
            "- 1: No fidelity — video is mostly unrelated or contradicts the ground truth.\n\n"
            "Return a single integer from 1 to 5."
        )

        score = self.llm.LLM_request([video_path, prompt], {
            "response_mime_type": "application/json",
            "response_schema": int
        })
        print(f"[DEBUG] Plot fidelity score: {score}")
        return score

    def _score_user_prompt_alignment(self, video_path: Path) -> int:
        print("[DEBUG] Starting user prompt alignment scoring...")
        prompt = (
            "A user gave the following request to generate a video:\n"
            f"{self.user_prompt}\n\n"
            "Evaluate how well the video satisfies the user's request.\n"
            "Rate alignment from 1 to 5 using this scale:\n"
            "- 5: Fully meets the request; all elements are well-executed.\n"
            "- 4: Covers most of the request with minor gaps.\n"
            "- 3: Partially aligned; significant elements are missing.\n"
            "- 2: Barely aligns; only a few request elements present.\n"
            "- 1: Does not address the request at all.\n\n"
            "Return only a single integer score from 1 to 5."
        )

        score = self.llm.LLM_request([video_path, prompt], {
            "response_mime_type": "application/json",
            "response_schema": int
        })
        print(f"[DEBUG] User prompt alignment score: {score}")
        return score

    def _score_video_quality(self, video_path: Path) -> int:
        print("[DEBUG] Starting video quality scoring...")
        prompt = (
            "Evaluate the overall technical and artistic quality of the video based on:\n"
            "- Visual clarity and resolution\n"
            "- Smoothness of editing and transitions\n"
            "- Coherence and aesthetic appeal\n\n"
            "Use this scale to rate quality:\n"
            "- 5: Excellent; professional-grade visuals and editing.\n"
            "- 4: Good; high quality with only minor flaws.\n"
            "- 3: Fair; acceptable but noticeable rough edges.\n"
            "- 2: Poor; significant quality or editing issues.\n"
            "- 1: Very poor; distracting problems or broken visuals.\n\n"
            "Return only a single integer score from 1 to 5."
        )

        score = self.llm.LLM_request([video_path, prompt], {
            "response_mime_type": "application/json",
            "response_schema": int
        })
        print(f"[DEBUG] Video quality score: {score}")
        return score

    def _score_brand_safety(self, video_path: Path) -> BrandSafetyScores:
        print("[DEBUG] Starting brand safety scoring...")
        prompt = (
            "Review the video and assess brand safety risks using the following categories:\n"
            "- Drug Use\n- Violence\n- Language\n- Sexual Content\n\n"
            "Score each from 1 (very safe) to 5 (high risk):\n"
            "- 5: Extremely unsafe; explicit and prominent\n"
            "- 4: Clearly present and noticeable\n"
            "- 3: Moderate risk; present but not dominant\n"
            "- 2: Low risk; mild or brief presence\n"
            "- 1: Very safe; no signs at all\n\n"
            "Also assign an 'overall' brand safety score using the same scale.\n"
            "Respond with a JSON object like this:\n"
            "{\n"
            "  \"drug_use\": 2,\n"
            "  \"violence\": 1,\n"
            "  \"language\": 3,\n"
            "  \"sexual_content\": 2,\n"
            "  \"overall\": 2\n"
            "}"
        )

        result = self.llm.LLM_request([video_path, prompt], {
            "response_mime_type": "application/json",
            "response_schema": BrandSafetyScores
        })
        print(f"[DEBUG] Brand safety scores: {result}")
        return BrandSafetyScores(**result)

    async def run(self) -> QualityAssessmentResult:
        print(f"[INFO] Running QualityAssessmentPipeline for: {self.video_blob_path}")
        video_path = self._download_video()

        fidelity_score = self._score_plot_fidelity(video_path)
        alignment_score = self._score_user_prompt_alignment(video_path)
        quality_score = self._score_video_quality(video_path)
        brand_safety_scores = self._score_brand_safety(video_path)

        result = QualityAssessmentResult(
            plot_fidelity=fidelity_score,
            user_prompt_alignment=alignment_score,
            video_quality=quality_score,
            brand_safety=brand_safety_scores
        )

        print("[SUCCESS] Quality assessment complete.")
        return result
