import os
import tempfile
import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import time

from dotenv import load_dotenv
load_dotenv()

from google import genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_json_loads(text: str) -> dict:
    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


EMPTY_CLINICAL_NOTES = {
    "patient_name": "not mentioned",
    "chief_complaint": "not mentioned",
    "history_of_present_illness": "not mentioned",
    "examination_findings": "not mentioned",
    "vital_signs": {
        "temperature": "not mentioned",
        "pulse": "not mentioned",
        "blood_pressure": "not mentioned",
        "respiratory_rate": "not mentioned",
        "oxygen_saturation": "not mentioned",
    },
    "provisional_diagnosis": "not mentioned",
    "investigations_advised": [],
    "laboratory_reports_reviewed": "not mentioned",
    "final_diagnosis": "not mentioned",
    "management_plan": "not mentioned",
    "follow_up_advice": "not mentioned",
    "summary": "not mentioned",
    "transcript": "",
}


class PatientAudioAnalyzer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")

        self.client = genai.Client(api_key=self.api_key)
        self.temp_dir = tempfile.mkdtemp()
        self.models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

    # ─────────────────────────────────────────────────────────────
    # INFRASTRUCTURE HELPERS
    # ─────────────────────────────────────────────────────────────

    def _generate_content_with_fallback(self, prompt, gemini_file=None, config=None):
        last_exception = None
        contents = [prompt, gemini_file] if gemini_file else [prompt]
        kwargs = {"model": self.models_to_try[0], "contents": contents}
        if config:
            kwargs["config"] = config

        for attempt, model in enumerate(self.models_to_try):
            try:
                kwargs["model"] = model
                response = self.client.models.generate_content(**kwargs)
                return response
            except Exception as e:
                last_exception = e
                error_str = str(e)
                if "503" in error_str or "overloaded" in error_str.lower():
                    logger.warning(f"Model {model} overloaded, trying fallback...")
                    if attempt < len(self.models_to_try) - 1:
                        time.sleep(1)
                        continue
                    else:
                        raise Exception(f"All models unavailable. Last error: {error_str}")
                else:
                    raise e

        raise last_exception

    async def download_audio_from_s3(self, s3_url: str) -> str:
        filename = f"patient_audio_{int(time.time())}.mp3"
        file_path = os.path.join(self.temp_dir, filename)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(s3_url) as response:
                    if response.status == 200:
                        with open(file_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        logger.info(f"Audio downloaded: {file_path}")
                        return file_path
                    else:
                        raise Exception(f"Failed to download audio: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Error downloading audio: {str(e)}")
            raise

    def upload_file_to_gemini(self, file_path: str) -> Any:
        try:
            myfile = self.client.files.upload(file=file_path)
            logger.info(f"File uploaded to Gemini: {myfile.name}")
            return myfile
        except Exception as e:
            logger.error(f"Error uploading to Gemini: {str(e)}")
            raise

    def cleanup_temp_file(self, file_path: str):
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temp file removed: {file_path}")
        except Exception as e:
            logger.error(f"Error removing temp file: {str(e)}")

    # ─────────────────────────────────────────────────────────────
    # STEP 1 — LANGUAGE DETECTION
    # ─────────────────────────────────────────────────────────────

    def detect_primary_language(self, gemini_file: Any) -> str:
        prompt = """
        Analyze this audio and detect the primary language being spoken.
        If multiple languages are present (e.g. Hindi + English mix), identify the dominant one.
        Return only the language name in lowercase (e.g., 'english', 'hindi', 'hindi-english').
        """
        try:
            response = self._generate_content_with_fallback(prompt, gemini_file)
            lang = response.text.strip().lower()
            logger.info(f"Primary language detected: {lang}")
            return lang
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return "english"

    # ─────────────────────────────────────────────────────────────
    # STEP 2 — TRANSCRIPT GENERATION
    # ─────────────────────────────────────────────────────────────

    def generate_transcript(self, gemini_file: Any, primary_language: str) -> Dict[str, Any]:
        prompt = f"""
        Generate a detailed, accurate transcript of this clinical audio in the exact language(s) spoken.

        CONTEXT: This is either:
        - A doctor dictating clinical notes alone (monologue)
        - A doctor-patient consultation (two-person conversation)

        Both are valid. Transcribe exactly what is spoken.

        SPEAKER IDENTIFICATION:
        - For monologue: use "Doctor" as the speaker
        - For consultation: use "Doctor" and "Patient" as speaker names
        - If actual names are mentioned in audio, use them (e.g., "Dr. Sharma", "Patient")
        - Listen for introductions or name mentions carefully

        TRANSCRIPTION RULES:
        - Word-by-word precision in original language and script
        - Preserve medical terminology exactly as spoken
        - For Hindi-English mix: preserve both scripts accurately
        - Capture: [inaudible], [background noise], [unclear] where needed
        - Timestamps at speaker changes or every 30 seconds

        TRANSCRIPT FORMAT:
        - For monologue: "Doctor: <text>"
        - For consultation: "Doctor: <text>\\nPatient: <text>"

        JSON OUTPUT REQUIREMENTS:
        - VALID parseable JSON only
        - Escape all special characters properly
        - NO trailing commas

        Required JSON structure:
        {{
            "transcript": "Full formatted transcript as plain text. For monologue: 'Doctor: ...'. For consultation: 'Doctor: ...\\nPatient: ...'",
            "segments": [
                {{
                    "speaker_name": "Doctor or Patient",
                    "text": "Exact transcription in original language",
                    "start_time": "MM:SS",
                    "language": "{primary_language}"
                }}
            ],
            "audio_type": "DOCTOR_DICTATION or DOCTOR_PATIENT_CONSULTATION"
        }}
        """
        try:
            response = self._generate_content_with_fallback(
                prompt, gemini_file,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info(f"Transcript generated: {len(result.get('segments', []))} segments, type: {result.get('audio_type')}")
            return result
        except Exception as e:
            logger.error(f"Error generating transcript: {str(e)}")
            return {"transcript": "", "segments": [], "audio_type": "unknown"}

    # ─────────────────────────────────────────────────────────────
    # STEP 3 — VALIDATION
    # ─────────────────────────────────────────────────────────────

    def validate_clinical_audio(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        transcript_text = transcript_data.get("transcript", "")
        audio_type = transcript_data.get("audio_type", "")

        prompt = f"""
        Analyze this transcript to determine if it contains valid clinical/medical content.

        Transcript: "{transcript_text}"
        Detected audio type: "{audio_type}"

        Check for:
        1. Medical or clinical content (symptoms, diagnosis, treatment, medications, vitals, etc.)
        2. Doctor dictating notes OR doctor-patient consultation
        3. Sufficient content to extract clinical information

        Classify as one of:
        - VALID_CLINICAL: Contains meaningful medical/clinical content (doctor dictation or consultation)
        - MUSIC: Primarily music or songs
        - NOISE: Background noise, static, or unclear audio
        - INSUFFICIENT_CONTENT: Too short or lacks any clinical content
        - NON_CLINICAL: Clear conversation but not medical in nature

        Return ONLY valid JSON:
        {{
            "is_valid_clinical": true/false,
            "content_type": "VALID_CLINICAL / MUSIC / NOISE / INSUFFICIENT_CONTENT / NON_CLINICAL",
            "audio_type": "DOCTOR_DICTATION / DOCTOR_PATIENT_CONSULTATION / UNKNOWN",
            "reason": "Brief reason",
            "recommendation": "PROCEED / SKIP"
        }}
        """
        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info(f"Validation: {result.get('content_type')} — {result.get('recommendation')}")
            return result
        except Exception as e:
            logger.error(f"Error validating transcript: {str(e)}")
            return {
                "is_valid_clinical": False,
                "content_type": "INSUFFICIENT_CONTENT",
                "recommendation": "SKIP",
                "error": str(e)
            }

    # ─────────────────────────────────────────────────────────────
    # STEP 4 — CLINICAL EXTRACTION (single Gemini call, all fields)
    # ─────────────────────────────────────────────────────────────

    def extract_clinical_notes(self, gemini_file: Any, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        transcript_text = transcript_data.get("transcript", "")
        audio_type = transcript_data.get("audio_type", "UNKNOWN")

        prompt = f"""
        You are a highly accurate clinical documentation assistant.

        Extract structured clinical notes from this medical audio.

        CONTEXT:
        - Audio type: {audio_type}
        - Language: {primary_language}
        - Full transcript for reference:
        "{transcript_text}"

        EXTRACTION RULES:
        - Extract information ONLY from what is explicitly mentioned in the audio
        - If a field is NOT mentioned at all, use exactly: "not mentioned"
        - For investigations_advised: return as a plain array of strings e.g. ["CBC", "X-Ray chest", "Blood sugar"]
          If none mentioned, return empty array: []
        - For vital_signs: extract numeric values with units if spoken (e.g. "120/80 mmHg", "98.6°F", "72 bpm")
          If a vital is not mentioned, use "not mentioned" for that field
        - For summary: write a concise 2-3 sentence clinical summary in English based on the audio
        - Medical terminology: preserve exact terms as spoken by the doctor
        - If audio is in Hindi or mixed language, still return all field VALUES in English

        CRITICAL: Return ONLY valid JSON. No markdown, no extra text, no backticks.

        {{
            "chief_complaint": "main complaint or not mentioned",
            "history_of_present_illness": "HPI details or not mentioned",
            "examination_findings": "findings or not mentioned",
            "vital_signs": {{
                "temperature": "value with unit or not mentioned",
                "pulse": "value with unit or not mentioned",
                "blood_pressure": "value with unit or not mentioned",
                "respiratory_rate": "value with unit or not mentioned",
                "oxygen_saturation": "value with unit or not mentioned"
            }},
            "provisional_diagnosis": "diagnosis or not mentioned",
            "investigations_advised": [],
            "laboratory_reports_reviewed": "lab findings or not mentioned",
            "final_diagnosis": "final diagnosis or not mentioned",
            "management_plan": "treatment/medications or not mentioned",
            "follow_up_advice": "follow-up instructions or not mentioned",
            "summary": "concise clinical summary in English"
        }}
        """
        try:
            response = self._generate_content_with_fallback(
                prompt, gemini_file,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Clinical notes extraction completed")
            return result
        except Exception as e:
            logger.error(f"Error extracting clinical notes: {str(e)}")
            notes = dict(EMPTY_CLINICAL_NOTES)
            notes.pop("transcript", None)
            return notes

    # ─────────────────────────────────────────────────────────────
    # MAIN ORCHESTRATOR
    # ─────────────────────────────────────────────────────────────

    async def analyze_audio(self, s3_url: str) -> Dict[str, Any]:
        start_time = datetime.now()
        file_path = None

        try:
            logger.info("Step 1: Downloading audio from URL...")
            file_path = await self.download_audio_from_s3(s3_url)

            logger.info("Step 2: Uploading to Gemini...")
            gemini_file = self.upload_file_to_gemini(file_path)

            logger.info("Step 3: Cleaning up temp file...")
            self.cleanup_temp_file(file_path)
            file_path = None

            logger.info("Step 4: Detecting language...")
            primary_language = self.detect_primary_language(gemini_file)

            logger.info("Step 5: Generating transcript...")
            transcript_data = self.generate_transcript(gemini_file, primary_language)

            logger.info("Step 6: Validating clinical content...")
            validation = self.validate_clinical_audio(transcript_data)

            transcript_text = transcript_data.get("transcript", "")

            if not validation.get("is_valid_clinical", False):
                logger.warning(f"Invalid clinical audio: {validation.get('content_type')} — returning empty notes")
                result = dict(EMPTY_CLINICAL_NOTES)
                result["transcript"] = transcript_text
                return result

            logger.info("Step 7: Extracting clinical notes...")
            clinical_notes = self.extract_clinical_notes(gemini_file, transcript_data, primary_language)

            # Build final payload — exactly the 14 keys client requires
            result = {
                "patient_name":                 clinical_notes.get("patient_name", "not mentioned"),
                "chief_complaint":              clinical_notes.get("chief_complaint", "not mentioned"),
                "history_of_present_illness":   clinical_notes.get("history_of_present_illness", "not mentioned"),
                "examination_findings":         clinical_notes.get("examination_findings", "not mentioned"),
                "vital_signs": {
                    "temperature":        clinical_notes.get("vital_signs", {}).get("temperature", "not mentioned"),
                    "pulse":              clinical_notes.get("vital_signs", {}).get("pulse", "not mentioned"),
                    "blood_pressure":     clinical_notes.get("vital_signs", {}).get("blood_pressure", "not mentioned"),
                    "respiratory_rate":   clinical_notes.get("vital_signs", {}).get("respiratory_rate", "not mentioned"),
                    "oxygen_saturation":  clinical_notes.get("vital_signs", {}).get("oxygen_saturation", "not mentioned"),
                },
                "provisional_diagnosis":        clinical_notes.get("provisional_diagnosis", "not mentioned"),
                "investigations_advised":       clinical_notes.get("investigations_advised", []),
                "laboratory_reports_reviewed":  clinical_notes.get("laboratory_reports_reviewed", "not mentioned"),
                "final_diagnosis":              clinical_notes.get("final_diagnosis", "not mentioned"),
                "management_plan":              clinical_notes.get("management_plan", "not mentioned"),
                "follow_up_advice":             clinical_notes.get("follow_up_advice", "not mentioned"),
                "summary":                      clinical_notes.get("summary", "not mentioned"),
                "transcript":                   transcript_text,
            }

            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ Patient audio analysis completed in {processing_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"✗ Error during analysis: {str(e)}")
            result = dict(EMPTY_CLINICAL_NOTES)
            result["transcript"] = ""
            return result

        finally:
            if file_path:
                self.cleanup_temp_file(file_path)


# ─────────────────────────────────────────────────────────────
# FLOW FUNCTION (called from server.py)
# ─────────────────────────────────────────────────────────────

async def patient_flow(url: str, patient_name: str = "not mentioned") -> Dict[str, Any]:
    try:
        analyzer = PatientAudioAnalyzer()
        result = await analyzer.analyze_audio(url)
        # Always override patient_name from request payload
        result["patient_name"] = patient_name if patient_name else "not mentioned"
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except Exception as e:
        logger.error(f"Error in patient_flow: {str(e)}")
        result = dict(EMPTY_CLINICAL_NOTES)
        result["patient_name"] = patient_name if patient_name else "not mentioned"
        result["transcript"] = ""
        return result


if __name__ == "__main__":
    test_url = "https://example-bucket.s3.amazonaws.com/patient/consultation_001.mp3"
    asyncio.run(patient_flow(test_url))