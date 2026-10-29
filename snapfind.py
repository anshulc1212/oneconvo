import contextlib
import os
import asyncio
import aiohttp
import tempfile
import time
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import subprocess

from dotenv import load_dotenv
load_dotenv()

from google import genai

try:
    from mutagen import File as MutagenFile
except ImportError:
    raise ImportError("mutagen is required. Install it with: pip install mutagen")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to usage tracking file (same directory as this script)
USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usage.json")


# ──────────────────────────────────────────────
# Usage tracking helpers
# ──────────────────────────────────────────────

def load_usage() -> dict:
    """Load usage.json from disk. Returns empty dict if file doesn't exist."""
    if not os.path.exists(USAGE_FILE):
        return {}
    try:
        with open(USAGE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read usage.json: {e}")
        return {}


def save_usage(data: dict):
    """Persist usage data to usage.json."""
    try:
        with open(USAGE_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Could not write usage.json: {e}")


def update_usage(client: str, duration_minutes: float):
    """
    Add duration_minutes to the client's total_minutes_processed in usage.json.
    Creates the client entry if it doesn't exist.
    """
    usage = load_usage()

    if client not in usage:
        usage[client] = {
            "total_minutes_processed": 0.0,
            "last_used": None
        }

    usage[client]["total_minutes_processed"] = round(
        usage[client]["total_minutes_processed"] + duration_minutes, 4
    )
    usage[client]["last_used"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    save_usage(usage)
    logger.info(f"Usage updated for client '{client}': +{duration_minutes:.4f} min "
                f"(total: {usage[client]['total_minutes_processed']:.4f} min)")


# ──────────────────────────────────────────────
# Audio duration check
# ──────────────────────────────────────────────

def get_audio_duration_minutes(file_path: str):
    """
    Use ffprobe to detect real audio duration.
    Works for mp3, wav, m4a, aac, webm, ogg, opus, etc.
    Ignores file extension.
    """
    try:
        command = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            file_path
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"ffprobe error: {result.stderr}")
            return None

        data = json.loads(result.stdout)
        duration_seconds = float(data["format"]["duration"])
        duration_minutes = duration_seconds / 60.0

        logger.info(
            f"Audio duration (ffprobe): {duration_seconds:.2f}s ({duration_minutes:.4f} min)"
        )

        return duration_minutes

    except Exception as e:
        logger.error(f"Error reading duration via ffprobe: {e}")
        return None


# ──────────────────────────────────────────────
# Main analyzer class
# ──────────────────────────────────────────────

class SnapfindInterviewAnalyzer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")

        self.client = genai.Client(api_key=self.api_key)
        self.temp_dir = tempfile.mkdtemp()

        self.primary_model = "gemini-2.5-flash"
        self.fallback_model = "gemini-2.0-flash"
        self.current_model = self.primary_model

    def _generate_content_with_fallback(self, prompt: str, gemini_file: Any = None, config: dict = None) -> Any:
        """Generate content with automatic fallback on 503 errors."""
        models_to_try = [self.primary_model, self.fallback_model]
        last_exception = None

        for attempt, model in enumerate(models_to_try):
            try:
                logger.info(f"API call with model: {model} (attempt {attempt + 1})")
                contents = [prompt]
                if gemini_file is not None:
                    contents.append(gemini_file)

                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config or {}
                )

                if self.current_model != model:
                    logger.info(f"Switched to fallback model: {model}")
                    self.current_model = model

                return response

            except Exception as e:
                error_str = str(e)
                last_exception = e

                if "503" in error_str and "overloaded" in error_str.lower():
                    logger.warning(f"Model {model} overloaded. Trying fallback...")
                    if attempt < len(models_to_try) - 1:
                        time.sleep(1)
                        continue
                    else:
                        raise Exception(f"All models unavailable. Last error: {error_str}")
                else:
                    raise e

        raise last_exception

    def safe_json_loads(self, text: str) -> dict:
        try:
            data = json.loads(text)
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            if isinstance(data, dict):
                return data
            return {}
        except Exception:
            return {}

    async def download_audio(self, url: str) -> str:
        """Download audio from URL and save to temp directory."""
        filename = f"audio_{int(time.time())}{os.path.splitext(url.split('?')[0])[-1] or '.mp3'}"
        file_path = os.path.join(self.temp_dir, filename)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download audio: HTTP {response.status}")
                    with open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
            logger.info(f"Audio downloaded: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            raise

    def upload_to_gemini(self, file_path: str) -> Any:
        """Upload audio file to Gemini Files API."""
        try:
            gemini_file = self.client.files.upload(file=file_path)
            logger.info(f"File uploaded to Gemini: {gemini_file.name}")
            return gemini_file
        except Exception as e:
            logger.error(f"Error uploading to Gemini: {e}")
            raise

    def cleanup(self, file_path: str):
        """Delete a local temp file."""
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temp file deleted: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting temp file: {e}")

    # ── Parallel Gemini calls ──────────────────

    async def generate_transcript(self, gemini_file: Any) -> List[dict]:
        """
        Call 1: Generate transcript segments from audio.
        Returns list of {speaker_name, text, start_time}
        """
        prompt = """
        Generate an accurate transcript of this interview audio.

        SPEAKER IDENTIFICATION:
        - Extract actual names mentioned in the audio (introductions, greetings, etc.)
        - Use consistent speaker names throughout
        - Only use generic names like "Speaker 1" if absolutely no names are mentioned

        TRANSCRIPTION RULES:
        - Transcribe word-for-word in the language spoken
        - Mark timestamps at each speaker turn in MM:SS format
        - Preserve natural speech including fillers (um, uh, etc.)

        Return ONLY valid JSON — no markdown, no extra text.

        Required format:
        {
            "segments": [
                {
                    "speaker_name": "Interviewer",
                    "text": "Exact words spoken",
                    "start_time": "00:00"
                }
            ]
        }
        """
        try:
            response = self._generate_content_with_fallback(
                prompt, gemini_file,
                config={"response_mime_type": "application/json"}
            )
            result = self.safe_json_loads(response.text)
            segments = result.get("segments", [])
            logger.info(f"Transcript: {len(segments)} segments")
            return segments
        except Exception as e:
            logger.error(f"Transcript generation failed: {e}")
            return []

    async def generate_question_answers(self, gemini_file: Any, questions: List[dict]) -> List[dict]:
        """
        Call 2: For each question in payload, find the exact answer spoken
        by the candidate in the audio. Returns null if not addressed.
        """
        questions_json = json.dumps(questions, ensure_ascii=False)

        prompt = f"""
        You are analyzing an interview audio recording.

        Below is a list of questions with their IDs. For each question, find the
        EXACT words spoken by the candidate as their answer in the audio.

        RULES:
        - Match questions by semantic meaning — the interviewer may paraphrase
        - Return the candidate's EXACT spoken words, do not paraphrase or summarize
        - If a question was not asked or not answered in the audio, set answer to null
        - Do not include interviewer speech in the answer field

        Questions to map:
        {questions_json}

        Return ONLY valid JSON — no markdown, no extra text.

        Required format:
        {{
            "question_answers": [
                {{
                    "question_id": 1,
                    "question_text": "original question text from payload",
                    "answer": "exact words spoken by candidate or null"
                }}
            ]
        }}
        """
        try:
            response = self._generate_content_with_fallback(
                prompt, gemini_file,
                config={"response_mime_type": "application/json"}
            )
            result = self.safe_json_loads(response.text)
            qa = result.get("question_answers", [])
            logger.info(f"Q&A mapping: {len(qa)} questions processed")
            return qa
        except Exception as e:
            logger.error(f"Q&A mapping failed: {e}")
            return [
                {
                    "question_id": q["question_id"],
                    "question_text": q["question_text"],
                    "answer": None,
                }
                for q in questions
            ]

    async def analyze_communication_skills(self, gemini_file: Any) -> dict:
        """
        Call 3: Analyze candidate's communication skills from audio.
        """
        prompt = """
        Analyze the candidate's communication skills in this interview audio.

        Focus ONLY on the candidate's speech — not the interviewer's.

        Evaluate:
        - Clarity and articulation
        - Professional language and vocabulary
        - Confidence and tone
        - Use of fillers (um, uh, like, etc.)
        - Response structure and coherence
        - Active listening indicators

        Scoring (0-100):
        - 90-100: Exceptional — articulate, confident, professional
        - 70-89: Strong — good communication, minor gaps
        - 50-69: Adequate — understandable but needs improvement
        - 30-49: Below average — unclear or unprofessional
        - 0-29: Poor — very difficult to follow

        Return ONLY valid JSON — no markdown, no extra text.

        Required format:
        {
            "communication_skills": {
                "score": 78,
                "evidence": ["Structured responses throughout", "Professional vocabulary used"],
                "strengths": ["Clear articulation", "Good use of examples"],
                "weaknesses": ["Occasional filler words", "Some incomplete sentences"],
                "examples": ["Well explained at 00:32", "Strong closing statement at 04:21"]
            }
        }
        """
        try:
            response = self._generate_content_with_fallback(
                prompt, gemini_file,
                config={"response_mime_type": "application/json"}
            )
            result = self.safe_json_loads(response.text)
            skills = result.get("communication_skills", {})
            logger.info("Communication skills analysis complete")
            return skills
        except Exception as e:
            logger.error(f"Communication skills analysis failed: {e}")
            return {}

    async def analyze_candidate_assessment(self, gemini_file: Any) -> dict:
        """
        Call 4: Analyze cultural fit, candidate score, and confidence level from audio.
        Gemini infers everything purely from the candidate's speech.
        """
        prompt = """
        You are an expert HR analyst. Analyze this interview audio and assess the candidate.

        EVALUATE THE FOLLOWING:

        1. CULTURAL FIT (score 0-100):
           - Listen for: teamwork mindset, communication style, attitude toward feedback,
             initiative, enthusiasm, transparency, shared success vs individual focus
           - Alignment Indicators: specific phrases or behaviors demonstrating cultural alignment
             (e.g., "Values open communication", "Proactive in seeking solutions")
           - Company Research: did the candidate demonstrate knowledge of the company,
             its mission, values, or recent projects based on what they said?
             Infer PURELY from the candidate's own words — no external info is provided.
             If no company research is evident, state that clearly.

        2. CANDIDATE SCORE (0-100):
           - Overall suitability score based on:
             * Depth and relevance of answers
             * Problem-solving approach demonstrated
             * Professionalism and attitude
             * Demonstrated experience and domain knowledge

        3. CONFIDENCE LEVEL:
           Assign one of three levels based on holistic assessment:
           - "High"   → Candidate is clearly well-suited, strong across all areas
           - "Medium" → Candidate shows promise but has notable gaps
           - "Low"    → Candidate struggles significantly or is poorly suited

        SCORING GUIDE:
        - 90-100: Exceptional
        - 70-89:  Strong
        - 50-69:  Adequate
        - 30-49:  Below average
        - 0-29:   Poor

        Return ONLY valid JSON — no markdown, no extra text.

        Required format:
        {
            "cultural_fit": {
                "score": 90,
                "alignment_indicators": [
                    "Values open communication and transparency",
                    "Proactive in seeking solutions and taking initiative",
                    "Emphasized shared success over individual accolades",
                    "Positive and enthusiastic demeanor"
                ],
                "company_research": "Clearly articulated understanding of the company mission and recent projects, referencing public initiatives"
            },
            "candidate_score": 82,
            "confidence_level": "High"
        }
        """
        try:
            response = self._generate_content_with_fallback(
                prompt, gemini_file,
                config={"response_mime_type": "application/json"}
            )
            result = self.safe_json_loads(response.text)
            logger.info("Candidate assessment analysis complete")
            return result
        except Exception as e:
            logger.error(f"Candidate assessment analysis failed: {e}")
            return {
                "cultural_fit": {
                    "score": 0,
                    "alignment_indicators": [],
                    "company_research": ""
                },
                "candidate_score": 0,
                "confidence_level": "Low"
            }

    async def analyze_mti(self, gemini_file: Any) -> dict:
        """
        Call 5: Analyze Mother Tongue Influence (MTI) from audio.
        Scores pronunciation accuracy, fluency & flow, and grammar interference.
        overall_mti_score is computed on the backend as average of the three.
        """
        prompt = """
        You are an expert linguist and HR speech analyst.
        Analyze this interview audio for Mother Tongue Influence (MTI).
        Focus ONLY on the candidate's speech — not the interviewer.

        SCORE THE FOLLOWING (each 0–100):

        1. PRONUNCIATION ACCURACY
           - Phoneme errors from native language interference
             (e.g. v/w confusion, th substitution, retroflex r, consonant clusters)
           - Syllable stress errors on common English words
           - Tone or pitch patterns carried over from native language

        2. FLUENCY AND FLOW
           - Unnatural pauses mid-sentence
           - Filler word count (um, uh, basically, only, like)
           - Code-switching — slipping into native language mid-response
           - Words-per-minute estimate
           - Recovery behavior after errors or slips

        3. GRAMMAR INTERFERENCE
           - Article errors (a/an/the omission or wrong use)
           - Verb tense inconsistency
           - Subject-verb agreement errors
           - Native language sentence structure patterns (e.g. SOV)
           - Preposition misuse

        EVIDENCE RULES:
        - Log EXACTLY 6 evidence items from the audio
        - At least 4 must be negative (issues found), at least 1 must be positive (strong moment)
        - Each item must include: timestamp (MM:SS), exact words spoken by candidate,
          which dimension it belongs to, and what the issue is
        - For positive items set issue to null
        - Detect the candidate's likely native language from speech patterns

        SCORING GUIDE:
        90–100 = Minimal MTI
        70–89  = Mild MTI
        50–69  = Moderate MTI
        30–49  = Strong MTI
        0–29   = Severe MTI

        Return ONLY valid JSON — no markdown, no extra text.

        Required format:
        {
            "mti_analysis": {
                "detected_native_language": "Hindi",
                "pronunciation_accuracy": {
                    "score": 58,
                    "phoneme_errors": [
                        "v/w confusion — says 'wery' for 'very'",
                        "Retroflex r replacing English r"
                    ],
                    "stress_errors": [
                        "de-VEL-op instead of DEV-el-op"
                    ]
                },
                "fluency_and_flow": {
                    "score": 72,
                    "words_per_minute": 118,
                    "filler_count": 9,
                    "code_switching_detected": true,
                    "code_switching_count": 2,
                    "unnatural_pause_count": 5,
                    "notes": "Adequate pace, self-corrects after slips"
                },
                "grammar_interference": {
                    "score": 55,
                    "error_types": [
                        "Article omission — 'submitted report' not 'submitted the report'",
                        "Tense inconsistency — switches present/past mid-sentence",
                        "SOV pattern — 'the client the issue raised'"
                    ],
                    "error_frequency": "High"
                },
                "evidence": [
                    {
                        "timestamp": "01:14",
                        "type": "negative",
                        "dimension": "pronunciation_accuracy",
                        "spoken": "I was wery confident about the project",
                        "issue": "v/w confusion"
                    },
                    {
                        "timestamp": "02:10",
                        "type": "negative",
                        "dimension": "grammar_interference",
                        "spoken": "She don't know about the deadline",
                        "issue": "Subject-verb agreement error"
                    },
                    {
                        "timestamp": "02:38",
                        "type": "negative",
                        "dimension": "grammar_interference",
                        "spoken": "We did the needful and submitted report",
                        "issue": "Direct translation phrase + article omission"
                    },
                    {
                        "timestamp": "03:45",
                        "type": "negative",
                        "dimension": "pronunciation_accuracy",
                        "spoken": "I tink the solution was correct",
                        "issue": "th substitution — 'tink' for 'think'"
                    },
                    {
                        "timestamp": "04:02",
                        "type": "negative",
                        "dimension": "fluency_and_flow",
                        "spoken": "Haan, so basically what happened was...",
                        "issue": "Code-switching — native word used, self-corrected"
                    },
                    {
                        "timestamp": "05:50",
                        "type": "positive",
                        "dimension": "fluency_and_flow",
                        "spoken": "When the conflict arose, I first listened to both sides before responding",
                        "issue": null
                    }
                ]
            }
        }
        """
        try:
            response = self._generate_content_with_fallback(
                prompt, gemini_file,
                config={"response_mime_type": "application/json"}
            )
            result = self.safe_json_loads(response.text)
            mti = result.get("mti_analysis", {})
            logger.info("MTI analysis complete")
            return mti
        except Exception as e:
            logger.error(f"MTI analysis failed: {e}")
            return {
                "detected_native_language": None,
                "pronunciation_accuracy": {"score": 0, "phoneme_errors": [], "stress_errors": []},
                "fluency_and_flow": {
                    "score": 0, "words_per_minute": 0, "filler_count": 0,
                    "code_switching_detected": False, "code_switching_count": 0,
                    "unnatural_pause_count": 0, "notes": ""
                },
                "grammar_interference": {"score": 0, "error_types": [], "error_frequency": "Unknown"},
                "evidence": []
            }

    # ── Main entry point ──────────────────────

    async def analyze(self, url: str, questions: List[dict], client: str) -> Dict[str, Any]:
        """
        Main method:
        1. Download audio from URL
        2. Check duration (reject if < 1 min)
        3. Upload to Gemini
        4. Run 5 parallel analysis calls
        5. Merge candidate_assessment into communication_skills
        6. Compute MTI overall_mti_score + mti_severity on backend
        7. Update usage.json
        8. Return combined result
        """
        file_path = None

        try:
            # Step 1: Download audio
            logger.info("Downloading audio from URL...")
            file_path = await self.download_audio(url)

            # Step 2: Check duration before uploading
            logger.info("Checking audio duration...")
            duration_minutes = get_audio_duration_minutes(file_path)

            if duration_minutes is None:
                self.cleanup(file_path)
                file_path = None
                return {"error": "Could not determine audio duration. Please check the file format."}

            if duration_minutes < 1.0:
                duration_seconds = round(duration_minutes * 60, 1)
                self.cleanup(file_path)
                file_path = None
                return {
                    "error": f"Audio too short ({duration_seconds}s). Minimum duration is 1 minute."
                }

            logger.info(f"Duration check passed: {duration_minutes:.4f} min")

            # Step 3: Upload to Gemini
            logger.info("Uploading to Gemini...")
            gemini_file = self.upload_to_gemini(file_path)

            # Step 4: Delete local temp file
            self.cleanup(file_path)
            file_path = None

            # Step 5: Run 5 parallel Gemini calls
            logger.info("Running parallel analysis (transcript + Q&A + communication + assessment + MTI)...")
            transcript, question_answers, communication_skills, candidate_assessment, mti_raw = await asyncio.gather(
                self.generate_transcript(gemini_file),
                self.generate_question_answers(gemini_file, questions),
                self.analyze_communication_skills(gemini_file),
                self.analyze_candidate_assessment(gemini_file),
                self.analyze_mti(gemini_file),
                return_exceptions=False
            )

            # Step 6: Merge candidate_assessment into communication_skills
            comm_score      = communication_skills.get("score", 0)
            cultural_score  = candidate_assessment.get("cultural_fit", {}).get("score", 0)
            candidate_score = candidate_assessment.get("candidate_score", 0)

            overall_score = round((comm_score + cultural_score + candidate_score) / 3)

            communication_skills["overall_score"]    = overall_score
            communication_skills["confidence_level"] = candidate_assessment.get("confidence_level", "Low")
            communication_skills["cultural_fit"]     = candidate_assessment.get("cultural_fit", {})

            logger.info(
                f"Scores merged — comm: {comm_score}, cultural: {cultural_score}, "
                f"candidate: {candidate_score}, overall: {overall_score}"
            )

            # Step 7: Compute MTI overall_mti_score + mti_severity on backend
            p = mti_raw.get("pronunciation_accuracy", {}).get("score", 0)
            f = mti_raw.get("fluency_and_flow", {}).get("score", 0)
            g = mti_raw.get("grammar_interference", {}).get("score", 0)

            overall_mti_score = round((p + f + g) / 3)

            if overall_mti_score >= 90:
                mti_severity = "Minimal"
            elif overall_mti_score >= 70:
                mti_severity = "Mild"
            elif overall_mti_score >= 50:
                mti_severity = "Moderate"
            elif overall_mti_score >= 30:
                mti_severity = "Strong"
            else:
                mti_severity = "Severe"

            mti_raw["overall_mti_score"] = overall_mti_score
            mti_raw["mti_severity"]      = mti_severity

            logger.info(
                f"MTI scores — pronunciation: {p}, fluency: {f}, grammar: {g}, "
                f"overall: {overall_mti_score}, severity: {mti_severity}"
            )

            # Step 8: Update usage tracking
            update_usage(client, duration_minutes)

            logger.info("Analysis complete.")
            return {
                "transcript":          transcript,
                "question_answers":    question_answers,
                "communication_skills": communication_skills,
                "mti_analysis":        mti_raw
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": str(e)}

        finally:
            if file_path:
                self.cleanup(file_path)