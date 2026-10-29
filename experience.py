import os
import tempfile
import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import time

from experiencereport import generate_and_upload_experience_report

from dotenv import load_dotenv
load_dotenv()

from google import genai

# Configure logging
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


def convert_to_percentage(score: float) -> int:
    """Convert decimal score (0.0-1.0) to percentage (0-100)"""
    return int(round(score * 100))


def convert_scores_to_percentage(data: dict) -> dict:
    """Recursively convert all decimal scores to percentages in the data structure"""
    if isinstance(data, dict):
        converted = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                converted[key] = convert_scores_to_percentage(value)
            elif isinstance(value, float) and (key.endswith('_score') or key in ['score', 'sentiment_score', 'overall_confidence']):
                converted[key] = convert_to_percentage(value)
            else:
                converted[key] = value
        return converted
    elif isinstance(data, list):
        return [convert_scores_to_percentage(item) for item in data]
    else:
        return data


class ExperienceFeedbackAnalyzer:
    def __init__(self, api_key: str = None):
        """
        Initialize the Experience Feedback Analyzer with Google Gemini API
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")

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

    async def download_audio_from_s3(self, s3_url: str, filename: str = None) -> str:
        """Download audio file from S3 URL and save to temp directory"""
        if not filename:
            filename = f"experience_audio_{int(time.time())}.mp3"

        file_path = os.path.join(self.temp_dir, filename)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(s3_url) as response:
                    if response.status == 200:
                        with open(file_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        logger.info(f"Audio downloaded successfully: {file_path}")
                        return file_path
                    else:
                        raise Exception(f"Failed to download audio: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Error downloading audio: {str(e)}")
            raise

    def upload_file_to_gemini(self, file_path: str) -> Any:
        """Upload file to Gemini and return file object"""
        try:
            myfile = self.client.files.upload(file=file_path)
            logger.info(f"File uploaded to Gemini: {myfile.name}")
            return myfile
        except Exception as e:
            logger.error(f"Error uploading file to Gemini: {str(e)}")
            raise

    def cleanup_temp_file(self, file_path: str):
        """Clean up temporary files"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary file removed: {file_path}")
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")

    def detect_primary_language(self, gemini_file: Any) -> str:
        """Detect the primary language of the audio"""
        prompt = """
        Analyze this audio and detect the primary language being spoken.
        If multiple languages are present, identify the dominant one.
        Return only the language name in lowercase (e.g., 'english', 'hindi', 'spanish').
        """
        try:
            response = self._generate_content_with_fallback(prompt, gemini_file)
            primary_language = response.text.strip().lower()
            logger.info(f"Primary language detected: {primary_language}")
            return primary_language
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return 'english'

    def generate_transcript(self, gemini_file: Any, primary_language: str) -> Dict[str, Any]:
        """Generate detailed transcript with timestamps and speaker identification"""
        prompt = f"""
        Generate a detailed, accurate transcript of this EMPLOYEE EXPERIENCE FEEDBACK CALL audio in the exact language(s) spoken.

        CONTEXT: This is an HR check-in call to a new employee about their first-week experience at the company.

        SPEAKER IDENTIFICATION (CRITICAL):
        - PRIMARY: Extract actual names mentioned in the audio (e.g., "Suhani", "Priya", "Rahul")
        - Listen for:
        * Direct introductions: "This is Suhani calling from...", "I'm speaking with..."
        * Name references: "Thank you, Suhani", "As you mentioned, John..."
        - Use FIRST identified name consistently as speaker_name
        - ONLY use generic identifiers (HR Agent, Employee) if absolutely no names found
        - Typical speakers: HR Agent / Caller, New Employee

        TRANSCRIPTION ACCURACY (CRITICAL):
        - Word-by-word precision in original language and script
        - PROPER NOUNS: Company names, role titles, person names, department names
        - Preserve natural speech, fillers (um, uh, like, you know), incomplete sentences
        - Capture: [laughs], [sighs], [pauses], [background noise], [inaudible]

        TIMESTAMP PRECISION:
        - Format: MM:SS (e.g., "00:00", "02:45")
        - Mark at speaker changes and topic shifts

        MULTILINGUAL HANDLING:
        - Label language per segment
        - Preserve original script

        JSON OUTPUT REQUIREMENTS (CRITICAL):
        - VALID, parseable JSON only
        - Escape all special characters properly
        - NO trailing commas

        Required JSON structure:
        {{
            "transcript": "Complete formatted transcript with [MM:SS] Speaker Name: text",
            "segments": [
                {{
                    "speaker_name": "Actual name from audio or HR Agent/Employee if no names found",
                    "text": "Exact transcription in original language",
                    "start_time": "MM:SS format timestamp",
                    "language": "Language code or name"
                }}
            ]
        }}

        QUALITY CHECKLIST:
        ✓ All speaker names are real names from audio (or explicitly generic if none found)
        ✓ Timestamps are sequential and logical
        ✓ Language tags are accurate per segment
        ✓ ALL PROPER NOUNS VERIFIED FOR CORRECT SPELLING
        ✓ JSON is valid (no syntax errors, trailing commas, unescaped quotes)
        """

        try:
            response = self._generate_content_with_fallback(
                prompt,
                gemini_file,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info(f"Transcript generated: {len(result.get('segments', []))} segments")
            return result
        except Exception as e:
            logger.error(f"Error generating transcript: {str(e)}")
            return {"transcript": "", "segments": []}

    def validate_experience_feedback_call(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if transcript contains a valid first-week experience feedback call"""
        transcript_text = transcript_data.get("transcript", "")

        prompt = f"""
        Analyze this transcript to determine if it contains a meaningful employee first-week or early-tenure experience feedback call.

        Transcript: "{transcript_text}"

        Check for:
        1. HR agent checking in with a new employee about their early experience
        2. Discussion of first days/week, team, manager, role, work environment
        3. Two-way conversation — HR asking, employee responding with experience
        4. Content type classification

        Classify as:
        - VALID_EXPERIENCE_FEEDBACK: Contains meaningful first-week or early-experience check-in dialogue
        - MUSIC: Primarily music or songs
        - NOISE: Primarily background noise, static, or unclear audio
        - MONOLOGUE: Single person speaking without interaction
        - INSUFFICIENT_CONTENT: Too short or lacks substantial content

        Format as JSON:
        {{
            "is_valid_experience_feedback": true/false,
            "content_type": "VALID_EXPERIENCE_FEEDBACK/MUSIC/NOISE/MONOLOGUE/INSUFFICIENT_CONTENT",
            "validation_details": {{
                "speaker_count_estimate": 2,
                "has_conversation": true,
                "content_duration_sufficient": true
            }},
            "reason": "Clear first-week experience check-in conversation",
            "recommendation": "PROCEED/SKIP"
        }}
        """

        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            validation_result = safe_json_loads(response.text)
            logger.info(f"Validation: {validation_result.get('content_type')} - {'PASS' if validation_result.get('is_valid_experience_feedback') else 'FAIL'}")
            return validation_result
        except Exception as e:
            logger.error(f"Error validating transcript: {str(e)}")
            return {
                "is_valid_experience_feedback": False,
                "content_type": "INSUFFICIENT_CONTENT",
                "recommendation": "SKIP",
                "error": str(e)
            }

    # ─────────────────────────────────────────────────────────────
    # PARALLEL ANALYSIS METHODS
    # ─────────────────────────────────────────────────────────────

    async def analyze_scores_and_sentiment_from_audio(self, gemini_file: Any, primary_language: str) -> Dict[str, Any]:
        """Analyze quality scores and overall sentiment from audio (parallel call 1)"""
        prompt = f"""
        You are an expert HR analyst specializing in new employee experience calls.
        Analyze this first-week experience feedback call audio for quality scores and sentiment.

        LANGUAGE: Primary language is {primary_language}.

        ## SENTIMENT ANALYSIS:
        Sentiment reflects the NEW EMPLOYEE's overall first-week experience and satisfaction.

        **overall_sentiment**: "positive" / "negative" / "neutral"
        **sentiment_score** (0-100):
        - 90-100: Extremely positive — employee thrilled, no concerns, highly engaged
        - 70-89: Positive — generally happy with minor observations
        - 50-69: Mixed — some positives and some concerns
        - 30-49: Negative — noticeable issues, dissatisfaction present
        - 0-29: Very negative — serious concerns, disengaged, or overwhelmed

        **sentiment_distribution**: % breakdown of positive / neutral / negative tone
        **emotional_tones**: detected emotions (e.g., "enthusiastic", "satisfied", "uncertain", "optimistic", "concerned")

        ## QUALITY SCORES:
        - **audio_quality_score** (0-100): Technical audio clarity
        - **speech_clarity_score** (0-100): Speaker articulation
        - **transcription_confidence** (0-100): Confidence in transcript accuracy
        - **analysis_reliability** (0-100): Overall AI analysis confidence
        - **noise_level**: "low" / "medium" / "high"
        - **overall_score** (0-100): Composite quality score
        - **confidence_scores_by_speaker**: Array — speaker name + score (0-100)
        - **overall_confidence** (0-100): Combined confidence

        CRITICAL: Use EXACT speaker names from the audio. Do not truncate or modify.

        Return ONLY valid JSON — no markdown, no extra text.

        {{
            "sentiment_analysis": {{
                "overall_sentiment": "positive",
                "sentiment_score": 82,
                "sentiment_distribution": {{
                    "positive": 75,
                    "negative": 5,
                    "neutral": 20
                }},
                "emotional_tones": ["enthusiastic", "satisfied", "optimistic"]
            }},
            "quality_scores": {{
                "audio_quality_score": 80,
                "speech_clarity_score": 78,
                "transcription_confidence": 85,
                "analysis_reliability": 82,
                "noise_level": "low",
                "overall_score": 81,
                "confidence_scores_by_speaker": [
                    {{"speaker": "Suhani", "score": 88}},
                    {{"speaker": "User", "score": 80}}
                ],
                "overall_confidence": 84
            }}
        }}
        """

        try:
            response = self._generate_content_with_fallback(
                prompt,
                gemini_file,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Audio-based scores and sentiment analysis")
            return result
        except Exception as e:
            logger.error(f"Error in audio scores and sentiment: {str(e)}")
            return {
                "sentiment_analysis": {"overall_sentiment": "neutral", "sentiment_score": 0},
                "quality_scores": {"overall_score": 0, "overall_confidence": 0},
                "error": str(e)
            }

    async def generate_summary_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """Generate call summary from transcript (parallel call 2)"""
        transcript_text = transcript_data.get("transcript", "")

        prompt = f"""
        You are an HR analyst. Provide a comprehensive summary of this first-week experience feedback call in {primary_language}.

        Transcript: "{transcript_text}"

        Include:
        - executive_summary: 2-3 sentence overview of the call
        - call_purpose: purpose of the call (e.g., "First-week experience check-in")
        - employee_name: name of the employee if mentioned, else "Unknown"
        - agent_name: name of the HR agent if mentioned, else "Unknown"
        - organization: company/organization name if mentioned, else "Unknown"
        - employee_role: employee's job role/title if mentioned, else "Unknown"
        - tenure_period: what period is being reviewed (e.g., "First week", "First month")
        - key_takeaways: list of 3-5 most important points from the call
        - overall_experience_rating: "excellent" / "good" / "mixed" / "poor" — based on employee's words
        - call_outcome: "positive_experience" / "follow_up_required" / "escalation_needed" / "incomplete"
        - confidence_score (0-100): confidence in this summary

        Format as JSON:
        {{
            "executive_summary": "...",
            "call_purpose": "First-week experience check-in",
            "employee_name": "...",
            "agent_name": "...",
            "organization": "...",
            "employee_role": "...",
            "tenure_period": "First week",
            "key_takeaways": ["...", "..."],
            "overall_experience_rating": "good",
            "call_outcome": "positive_experience",
            "confidence_score": 88
        }}
        """

        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Summary generation from transcript")
            return result
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {"executive_summary": "", "key_takeaways": [], "error": str(e)}

    async def analyze_week_experience(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """Analyze the employee's overall first-week experience (parallel call 3)"""
        transcript_text = transcript_data.get("transcript", "")

        prompt = f"""
        You are an expert HR analyst. Analyze the employee's first-week experience from this feedback call transcript.

        Transcript: "{transcript_text}"

        ## EXPERIENCE DIMENSIONS (each scored 0-100):
        1. **overall_experience_score**: Composite first-week experience score
        2. **team_integration_score**: How well the employee has integrated with the team
        3. **manager_support_score**: Quality of support received from the manager
        4. **role_clarity_score**: How clear and well-defined the employee's role is
        5. **work_environment_score**: Satisfaction with the physical/cultural work environment
        6. **onboarding_quality_score**: Quality of the onboarding/induction process received

        ## EXPERIENCE HIGHLIGHTS:
        - positives: list of specific things the employee appreciated or enjoyed
        - concerns: list of anything the employee flagged as a challenge or issue
        - suggestions: any improvements the employee mentioned or implied

        ## EMPLOYEE ENGAGEMENT:
        - engagement_level: "high" / "medium" / "low"
        - enthusiasm_for_role: "high" / "medium" / "low"
        - likelihood_to_stay: "high" / "medium" / "low" — inferred from tone and content
        - initial_impression: employee's overall initial impression of the company

        Format as JSON:
        {{
            "experience_scores": {{
                "overall_experience_score": 85,
                "team_integration_score": 90,
                "manager_support_score": 88,
                "role_clarity_score": 82,
                "work_environment_score": 87,
                "onboarding_quality_score": 80
            }},
            "experience_highlights": {{
                "positives": ["..."],
                "concerns": ["..."],
                "suggestions": ["..."]
            }},
            "employee_engagement": {{
                "engagement_level": "high",
                "enthusiasm_for_role": "high",
                "likelihood_to_stay": "high",
                "initial_impression": "..."
            }}
        }}
        """

        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Week experience analysis")
            return result
        except Exception as e:
            logger.error(f"Error analyzing week experience: {str(e)}")
            return {"experience_scores": {}, "experience_highlights": {}, "error": str(e)}

    async def analyze_role_alignment(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """Analyze role alignment and job expectation match (parallel call 4)"""
        transcript_text = transcript_data.get("transcript", "")

        prompt = f"""
        You are an expert HR analyst. Analyze whether the employee's actual role aligns with what was communicated during hiring.

        Transcript: "{transcript_text}"

        ## ROLE ALIGNMENT:
        - **alignment_score** (0-100): How well the actual role matches what was described during hiring
          * 90-100: Perfect match — exactly as described
          * 70-89: Good match — mostly aligned with minor differences
          * 50-69: Partial match — some discrepancies found
          * 30-49: Poor match — significant gaps between expectation and reality
          * 0-29: No match — very different from what was communicated

        - **alignment_status**: "fully_aligned" / "mostly_aligned" / "partially_aligned" / "misaligned"
        - **employee_statement**: direct quote or paraphrase of what the employee said about role alignment
        - **kra_clarity**: "clear" / "partially_clear" / "unclear" — how clear the KRAs were communicated
        - **responsibilities_match**: true/false — do the actual duties match the JD?
        - **additional_opportunities**: any new responsibilities or growth areas the employee mentioned
        - **concerns**: any discrepancies or concerns about role scope

        ## EXPECTATION VS REALITY:
        - **hiring_promise**: what was communicated during hiring (inferred from employee's words)
        - **actual_experience**: what the employee is actually experiencing
        - **gap_identified**: true/false
        - **gap_description**: description of any gap found

        Format as JSON:
        {{
            "role_alignment": {{
                "alignment_score": 90,
                "alignment_status": "fully_aligned",
                "employee_statement": "...",
                "kra_clarity": "clear",
                "responsibilities_match": true,
                "additional_opportunities": ["Data analysis tasks added based on employee's interest"],
                "concerns": []
            }},
            "expectation_vs_reality": {{
                "hiring_promise": "...",
                "actual_experience": "...",
                "gap_identified": false,
                "gap_description": ""
            }}
        }}
        """

        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Role alignment analysis")
            return result
        except Exception as e:
            logger.error(f"Error analyzing role alignment: {str(e)}")
            return {"role_alignment": {}, "expectation_vs_reality": {}, "error": str(e)}

    async def analyze_challenges_and_support(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """Analyze challenges faced and support needs (parallel call 5)"""
        transcript_text = transcript_data.get("transcript", "")

        prompt = f"""
        You are an expert HR analyst. Analyze the challenges and support needs expressed by the new employee.

        Transcript: "{transcript_text}"

        ## CHALLENGES:
        For each challenge identified:
        - challenge_title: short descriptive title
        - description: what the challenge is
        - category: "policy" / "workload" / "tools" / "team" / "manager" / "role" / "personal" / "process" / "other"
        - severity: "high" / "medium" / "low"
        - employee_attitude: "accepting" / "concerned" / "frustrated" / "seeking_help"
        - mentioned_explicitly: true/false — did employee bring it up directly?

        ## SUPPORT NEEDS:
        For each support need identified:
        - need: what support is needed
        - category: "hr_support" / "manager_support" / "team_support" / "tools" / "training" / "policy_clarification" / "other"
        - urgency: "immediate" / "soon" / "no_rush"
        - currently_addressed: true/false — is this already being handled?

        ## OVERALL SUPPORT ASSESSMENT:
        - support_adequacy: "well_supported" / "adequately_supported" / "needs_more_support" / "unsupported"
        - hr_action_required: true/false
        - action_items: list of specific actions HR/manager should take

        Format as JSON:
        {{
            "challenges": [
                {{
                    "challenge_title": "Leave during probation",
                    "description": "Employee wanted leave but understands it is not allowed in first month",
                    "category": "policy",
                    "severity": "low",
                    "employee_attitude": "accepting",
                    "mentioned_explicitly": true
                }}
            ],
            "support_needs": [
                {{
                    "need": "...",
                    "category": "...",
                    "urgency": "...",
                    "currently_addressed": false
                }}
            ],
            "overall_support_assessment": {{
                "support_adequacy": "well_supported",
                "hr_action_required": false,
                "action_items": ["..."]
            }}
        }}
        """

        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Challenges and support needs analysis")
            return result
        except Exception as e:
            logger.error(f"Error analyzing challenges: {str(e)}")
            return {"challenges": [], "support_needs": [], "error": str(e)}

    async def extract_topics_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """Extract key topics discussed from transcript (parallel call 6)"""
        transcript_text = transcript_data.get("transcript", "")

        prompt = f"""
        Identify and analyze the key topics discussed in this first-week experience feedback call.

        Transcript: "{transcript_text}"

        For each topic:
        - topic: topic name
        - importance_score (0-100): how significant this topic was in the call
        - mentions: approximate number of times it came up
        - context: brief context about how it was discussed
        - sentiment: "positive" / "negative" / "neutral" — employee's sentiment about this topic

        Also provide:
        - topic_categories: broad category labels (e.g., "team", "manager", "role", "culture", "policy", "workload")
        - most_discussed_topic: name of the topic with most discussion

        Format as JSON:
        {{
            "key_topics": [
                {{
                    "topic": "Team Integration",
                    "importance_score": 85,
                    "mentions": 3,
                    "context": "Employee spoke positively about meeting team members and team activities",
                    "sentiment": "positive"
                }}
            ],
            "topic_categories": ["team", "manager", "role"],
            "most_discussed_topic": "Team Integration"
        }}
        """

        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Topic extraction from transcript")
            return result
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return {"key_topics": [], "topic_categories": [], "error": str(e)}

    # ─────────────────────────────────────────────────────────────
    # MAIN ANALYSIS ORCHESTRATOR
    # ─────────────────────────────────────────────────────────────

    async def analyze_audio(self, s3_url: str) -> Dict[str, Any]:
        """Main method to perform comprehensive experience feedback call analysis"""
        start_time = datetime.now()
        file_path = None

        try:
            logger.info("Step 1: Downloading audio from S3...")
            file_path = await self.download_audio_from_s3(s3_url)

            logger.info("Step 2: Uploading file to Gemini...")
            gemini_file = self.upload_file_to_gemini(file_path)

            logger.info("Step 3: Cleaning up temporary file...")
            self.cleanup_temp_file(file_path)
            file_path = None

            logger.info("Step 4: Detecting primary language...")
            primary_language = self.detect_primary_language(gemini_file)

            logger.info("Step 5: Generating transcript...")
            transcript_data = self.generate_transcript(gemini_file, primary_language)

            logger.info("Step 6: Validating experience feedback call content...")
            validation_result = self.validate_experience_feedback_call(transcript_data)

            results = {"transcript": transcript_data}

            if validation_result.get("is_valid_experience_feedback", False):
                logger.info("✓ Validation passed — proceeding with full analysis...")
                logger.info("Step 7: Running parallel analysis tasks...")

                async def run_experience_analysis():
                    tasks = [
                        self.analyze_scores_and_sentiment_from_audio(gemini_file, primary_language),  # 0 - audio
                        self.generate_summary_from_transcript(transcript_data, primary_language),      # 1 - transcript
                        self.analyze_week_experience(transcript_data, primary_language),               # 2 - transcript
                        self.analyze_role_alignment(transcript_data, primary_language),                # 3 - transcript
                        self.analyze_challenges_and_support(transcript_data, primary_language),        # 4 - transcript
                        self.extract_topics_from_transcript(transcript_data, primary_language),        # 5 - transcript
                    ]
                    analysis_results = await asyncio.gather(*tasks, return_exceptions=True)

                    return {
                        "quality":     analysis_results[0] if not isinstance(analysis_results[0], Exception) else {"error": str(analysis_results[0])},
                        "summary":     analysis_results[1] if not isinstance(analysis_results[1], Exception) else {"error": str(analysis_results[1])},
                        "experience":  analysis_results[2] if not isinstance(analysis_results[2], Exception) else {"error": str(analysis_results[2])},
                        "role":        analysis_results[3] if not isinstance(analysis_results[3], Exception) else {"error": str(analysis_results[3])},
                        "challenges":  analysis_results[4] if not isinstance(analysis_results[4], Exception) else {"error": str(analysis_results[4])},
                        "topics":      analysis_results[5] if not isinstance(analysis_results[5], Exception) else {"error": str(analysis_results[5])},
                    }

                parallel_results = await run_experience_analysis()
                quality = parallel_results.get("quality", {})

                results.update({
                    "summary":        parallel_results.get("summary", {}),
                    "sentiment":      quality.get("sentiment_analysis", {}),
                    "quality_scores": quality.get("quality_scores", {}),
                    "experience":     parallel_results.get("experience", {}),
                    "role_alignment": parallel_results.get("role", {}),
                    "challenges":     parallel_results.get("challenges", {}),
                    "topics":         parallel_results.get("topics", {}),
                })

                logger.info("✓ Full experience feedback analysis completed")

            else:
                logger.warning(f"✗ Validation failed: {validation_result.get('content_type')} — {validation_result.get('reason')}")
                logger.info("→ Returning transcript only...")

                results.update({
                    "summary":        {"executive_summary": "Analysis skipped — not a valid experience feedback call", "key_takeaways": []},
                    "sentiment":      {"overall_sentiment": "neutral", "sentiment_score": 0},
                    "quality_scores": {"overall_score": 0},
                    "experience":     {"experience_scores": {}, "experience_highlights": {}},
                    "role_alignment": {"role_alignment": {}, "expectation_vs_reality": {}},
                    "challenges":     {"challenges": [], "support_needs": []},
                    "topics":         {"key_topics": [], "topic_categories": []},
                })

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            final_results = {
                "validation": validation_result,
                "metadata": {
                    "processing_time_seconds": processing_time,
                    "timestamp": start_time.isoformat(),
                    "primary_language": primary_language,
                    "file_url": s3_url,
                    "status": "completed_full_analysis" if validation_result.get("is_valid_experience_feedback", False) else "completed_transcript_only",
                    "analysis_version": "1.0",
                    "analysis_type": "experience_feedback"
                },
                "language_info": {
                    "primary_language": primary_language
                },
                "analysis_results": results
            }

            logger.info(f"✓ Completed in {processing_time:.2f}s")
            print("final_results", final_results)
            return final_results

        except Exception as e:
            logger.error(f"✗ Error: {str(e)}")
            return {
                "error": str(e),
                "metadata": {
                    "processing_time_seconds": 0,
                    "timestamp": start_time.isoformat(),
                    "status": "failed"
                }
            }

        finally:
            if file_path and os.path.exists(file_path):
                self.cleanup_temp_file(file_path)

    async def analyze_audio_from_file(self, audio_file, filename: str = None) -> Dict[str, Any]:
        """Analyze audio directly from uploaded file / form data"""
        start_time = datetime.now()
        file_path = None

        try:
            logger.info("Step 1: Saving uploaded audio file...")

            if not filename:
                filename = f"experience_audio_{int(time.time())}.mp3"

            file_path = os.path.join(self.temp_dir, filename)

            if hasattr(audio_file, 'read'):
                with open(file_path, 'wb') as f:
                    f.write(audio_file.read())
            elif isinstance(audio_file, bytes):
                with open(file_path, 'wb') as f:
                    f.write(audio_file)
            else:
                raise ValueError("audio_file must be a file-like object or bytes")

            logger.info(f"Audio saved: {file_path}")

            logger.info("Step 2: Uploading to Gemini...")
            gemini_file = self.upload_file_to_gemini(file_path)

            logger.info("Step 3: Cleaning up...")
            self.cleanup_temp_file(file_path)
            file_path = None

            logger.info("Step 4: Detecting language...")
            primary_language = self.detect_primary_language(gemini_file)

            logger.info("Step 5: Generating transcript...")
            transcript_data = self.generate_transcript(gemini_file, primary_language)

            logger.info("Step 6: Validating experience feedback call...")
            validation_result = self.validate_experience_feedback_call(transcript_data)

            results = {"transcript": transcript_data}

            if validation_result.get("is_valid_experience_feedback", False):
                logger.info("✓ Validation passed — running full analysis...")

                async def run_analysis():
                    tasks = [
                        self.analyze_scores_and_sentiment_from_audio(gemini_file, primary_language),
                        self.generate_summary_from_transcript(transcript_data, primary_language),
                        self.analyze_week_experience(transcript_data, primary_language),
                        self.analyze_role_alignment(transcript_data, primary_language),
                        self.analyze_challenges_and_support(transcript_data, primary_language),
                        self.extract_topics_from_transcript(transcript_data, primary_language),
                    ]
                    analysis_results = await asyncio.gather(*tasks, return_exceptions=True)

                    return {
                        "quality":    analysis_results[0] if not isinstance(analysis_results[0], Exception) else {"error": str(analysis_results[0])},
                        "summary":    analysis_results[1] if not isinstance(analysis_results[1], Exception) else {"error": str(analysis_results[1])},
                        "experience": analysis_results[2] if not isinstance(analysis_results[2], Exception) else {"error": str(analysis_results[2])},
                        "role":       analysis_results[3] if not isinstance(analysis_results[3], Exception) else {"error": str(analysis_results[3])},
                        "challenges": analysis_results[4] if not isinstance(analysis_results[4], Exception) else {"error": str(analysis_results[4])},
                        "topics":     analysis_results[5] if not isinstance(analysis_results[5], Exception) else {"error": str(analysis_results[5])},
                    }

                parallel_results = await run_analysis()
                quality = parallel_results.get("quality", {})

                results.update({
                    "summary":        parallel_results.get("summary", {}),
                    "sentiment":      quality.get("sentiment_analysis", {}),
                    "quality_scores": quality.get("quality_scores", {}),
                    "experience":     parallel_results.get("experience", {}),
                    "role_alignment": parallel_results.get("role", {}),
                    "challenges":     parallel_results.get("challenges", {}),
                    "topics":         parallel_results.get("topics", {}),
                })

                logger.info("✓ Full analysis completed")

            else:
                logger.warning(f"✗ Validation failed: {validation_result.get('content_type')}")
                results.update({
                    "summary":        {"executive_summary": "Analysis skipped", "key_takeaways": []},
                    "sentiment":      {"overall_sentiment": "neutral", "sentiment_score": 0},
                    "quality_scores": {"overall_score": 0},
                    "experience":     {"experience_scores": {}, "experience_highlights": {}},
                    "role_alignment": {"role_alignment": {}, "expectation_vs_reality": {}},
                    "challenges":     {"challenges": [], "support_needs": []},
                    "topics":         {"key_topics": [], "topic_categories": []},
                })

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            final_results = {
                "validation": validation_result,
                "metadata": {
                    "processing_time_seconds": processing_time,
                    "timestamp": start_time.isoformat(),
                    "primary_language": primary_language,
                    "file_name": filename,
                    "status": "completed_full_analysis" if validation_result.get("is_valid_experience_feedback", False) else "completed_transcript_only",
                    "analysis_version": "1.0",
                    "analysis_type": "experience_feedback"
                },
                "language_info": {
                    "primary_language": primary_language
                },
                "analysis_results": results
            }

            logger.info(f"✓ Completed in {processing_time:.2f}s")
            print("final_results", final_results)
            return final_results

        except Exception as e:
            logger.error(f"✗ Error: {str(e)}")
            return {
                "error": str(e),
                "metadata": {
                    "processing_time_seconds": 0,
                    "timestamp": start_time.isoformat(),
                    "status": "failed"
                }
            }

        finally:
            if file_path and os.path.exists(file_path):
                self.cleanup_temp_file(file_path)


def filter_json(data: dict) -> dict:
    """Filter to keep only metadata and transcript"""
    filtered = {}
    if "metadata" in data:
        filtered["metadata"] = data["metadata"]
    if "analysis_results" in data and "transcript" in data["analysis_results"]:
        filtered["analysis_results"] = {
            "transcript": data["analysis_results"]["transcript"]
        }
    return filtered


async def experience_flow(url: str, client: str = "unknown"):
    """Main experience feedback call analysis flow"""
    try:
        analyzer = ExperienceFeedbackAnalyzer()
        result = await analyzer.analyze_audio(url)

        analysis_status = result.get("metadata", {}).get("status")

        if analysis_status == "completed_full_analysis":
            print("Step 1: Full analysis completed — generating comprehensive experience report...")
            report_response = generate_and_upload_experience_report(result)
            print("Step 2: Experience report generated")
        elif analysis_status == "completed_transcript_only":
            print("Step 1: Transcript-only — generating transcript report...")
            refactored_result = filter_json(result)
            report_response = generate_and_upload_experience_report(refactored_result)
            print("Step 2: Transcript report generated")
            print(f"Content type: {result.get('validation', {}).get('content_type', 'Unknown')}")
        else:
            print("Step 1: Analysis failed — no report generated")
            report_response = None

        if report_response:
            result["report_url"] = report_response

        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    except Exception as e:
        print("Error during analysis:", e)
        return {"error": str(e)}


async def experience_flow_from_file(audio_file, filename: str = None):
    """Experience flow for analyzing audio from uploaded file (form data)"""
    try:
        analyzer = ExperienceFeedbackAnalyzer()
        result = await analyzer.analyze_audio_from_file(audio_file, filename)

        analysis_status = result.get("metadata", {}).get("status")

        if analysis_status == "completed_full_analysis":
            print("Step 1: Full analysis completed — generating comprehensive experience report...")
            report_response = generate_and_upload_experience_report(result)
            print("Step 2: Experience report generated")
        elif analysis_status == "completed_transcript_only":
            print("Step 1: Transcript-only — generating transcript report...")
            refactored_result = filter_json(result)
            report_response = generate_and_upload_experience_report(refactored_result)
            print("Step 2: Transcript report generated")
        else:
            print("Step 1: Analysis failed — no report generated")
            report_response = None

        if report_response:
            result["report_url"] = report_response

        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    except Exception as e:
        print("Error during analysis:", e)
        return {"error": str(e)}


if __name__ == "__main__":
    s3_audio_url = "https://drive.google.com/uc?export=download&id=1OzbtNWyxA86sBLVTSUMm7WgELg-bo5IH"
    asyncio.run(experience_flow(s3_audio_url))