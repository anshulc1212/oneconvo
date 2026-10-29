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

from feedbackreport import generate_and_upload_feedback_report

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


class FeedbackCallAnalyzer:
    def __init__(self, api_key: str = None):
        """
        Initialize the Feedback Call Analyzer with Google Gemini API
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
        """
        Download audio file from S3 URL and save to temp directory
        """
        if not filename:
            filename = f"feedback_audio_{int(time.time())}.mp3"

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
        """
        Upload file to Gemini and return file object
        """
        try:
            myfile = self.client.files.upload(file=file_path)
            logger.info(f"File uploaded to Gemini: {myfile.name}")
            return myfile
        except Exception as e:
            logger.error(f"Error uploading file to Gemini: {str(e)}")
            raise

    def cleanup_temp_file(self, file_path: str):
        """
        Clean up temporary files
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary file removed: {file_path}")
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")

    def detect_primary_language(self, gemini_file: Any) -> str:
        """
        Detect the primary language of the audio
        """
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
        """
        Generate detailed transcript with timestamps and speaker identification
        """
        prompt = f"""
        Generate a detailed, accurate transcript of this FEEDBACK CALL audio in the exact language(s) spoken.

        SPEAKER IDENTIFICATION (CRITICAL):
        - PRIMARY: Extract actual names mentioned in the audio (e.g., "Suhani", "John", "Priya")
        - Listen for:
        * Direct introductions: "This is Suhani calling from...", "I'm speaking with...", "My name is..."
        * Name mentions during conversation: "Suhani, I have a question", "Thank you, John"
        * Third-party introductions: "We're speaking with..."
        - Use FIRST identified name consistently as speaker_name
        - ONLY use generic identifiers (HR Agent, Employee, User) if absolutely no names found
        - Typical speakers in feedback calls: HR Agent / Caller, New Employee / User

        TRANSCRIPTION ACCURACY (CRITICAL):
        - LISTEN CAREFULLY: Exact pronunciation and phonetics
        - Word-by-word precision in original language and script
        - PROPER NOUNS: Company names, person names, product names, locations
        - Context-aware spelling for homophones and similar-sounding words
        - Preserve natural speech patterns, fillers (um, uh, etc.), incomplete sentences
        - Capture emotional markers: [laughs], [sighs], [pauses], [background noise]
        - Capture overlapping or interrupted speech: [call drops], [inaudible]

        TIMESTAMP PRECISION:
        - Format: MM:SS (e.g., "00:00", "02:45", "05:30")
        - Mark at natural breaks (speaker changes, topic shifts, pauses > 2 seconds)

        MULTILINGUAL HANDLING:
        - Label language for EACH segment individually
        - Note code-switching if languages mix
        - Preserve original script (Devanagari, Latin, Arabic, etc.)

        JSON OUTPUT REQUIREMENTS (CRITICAL):
        - Generate VALID, parseable JSON only
        - Escape all special characters: quotes (\"), backslashes (\\\\), newlines (\\n)
        - NO trailing commas in arrays or objects

        Required JSON structure:
        {{
            "transcript": "Complete formatted transcript with [MM:SS] Speaker Name: text for each segment",
            "segments": [
                {{
                    "speaker_name": "Actual name from audio or HR Agent/User only if no names found",
                    "text": "Exact transcription in original language",
                    "start_time": "MM:SS format timestamp",
                    "language": "Language code or name (e.g., 'English', 'Hindi', 'Hindi-English mix')"
                }}
            ]
        }}

        QUALITY CHECKLIST BEFORE OUTPUT:
        ✓ All speaker names are real names from audio (or explicitly generic if none found)
        ✓ Timestamps are sequential and logical
        ✓ Language tags are accurate per segment
        ✓ ALL PROPER NOUNS VERIFIED FOR CORRECT SPELLING
        ✓ JSON is valid (no syntax errors, trailing commas, unescaped quotes)
        ✓ Full transcript matches segmented version
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

    def validate_feedback_call(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate if transcript contains a valid HR feedback/onboarding call
        """
        transcript_text = transcript_data.get("transcript", "")

        prompt = f"""
        Analyze this transcript to determine if it contains a meaningful HR feedback or onboarding call.

        Transcript: "{transcript_text}"

        Check for:
        1. Conversational content between HR/agent and an employee or candidate
        2. Feedback, onboarding, check-in, or survey related dialogue
        3. Number of speakers and quality of interactions
        4. Content type classification

        Classify as:
        - VALID_FEEDBACK_CALL: Contains meaningful HR feedback, onboarding check-in, or employee survey dialogue
        - MUSIC: Primarily music or songs
        - NOISE: Primarily background noise, static, or unclear audio
        - MONOLOGUE: Single person speaking without interaction
        - INSUFFICIENT_CONTENT: Too short or lacks substantial content

        Format as JSON:
        {{
            "is_valid_feedback_call": true/false,
            "content_type": "VALID_FEEDBACK_CALL/MUSIC/NOISE/MONOLOGUE/INSUFFICIENT_CONTENT",
            "validation_details": {{
                "speaker_count_estimate": 2,
                "has_conversation": true,
                "content_duration_sufficient": true
            }},
            "reason": "Clear HR feedback conversation between agent and employee",
            "recommendation": "PROCEED/SKIP"
        }}
        """

        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            validation_result = safe_json_loads(response.text)
            logger.info(f"Validation: {validation_result.get('content_type')} - {'PASS' if validation_result.get('is_valid_feedback_call') else 'FAIL'}")
            return validation_result
        except Exception as e:
            logger.error(f"Error validating transcript: {str(e)}")
            return {
                "is_valid_feedback_call": False,
                "content_type": "INSUFFICIENT_CONTENT",
                "recommendation": "SKIP",
                "error": str(e)
            }

    # ─────────────────────────────────────────────────────────────
    # PARALLEL ANALYSIS METHODS
    # ─────────────────────────────────────────────────────────────

    async def analyze_scores_and_sentiment_from_audio(self, gemini_file: Any, primary_language: str) -> Dict[str, Any]:
        """
        Analyze quality scores and sentiment from audio (parallel call 1)
        """
        prompt = f"""
        You are an expert HR quality analyst specializing in onboarding and employee feedback calls.
        Analyze this feedback call audio and provide quality scores and sentiment analysis.

        LANGUAGE: Primary language is {primary_language}.

        ## SENTIMENT ANALYSIS:
        **overall_sentiment**: "positive" / "negative" / "neutral" (reflects employee's overall experience)
        **sentiment_score** (0-100):
        - 90-100: Highly positive — employee very satisfied, enthusiastic, no issues
        - 70-89: Positive — generally satisfied with minor concerns
        - 50-69: Mixed — both positive and negative aspects present
        - 30-49: Negative — noticeable dissatisfaction or concerns
        - 0-29: Very negative — major issues, frustrated, or disengaged

        **sentiment_distribution**: percentage breakdown of positive / negative / neutral tone
        **emotional_tones**: array of detected emotions (e.g., "enthusiastic", "concerned", "confused", "satisfied")

        ## QUALITY SCORES:
        - **audio_quality_score** (0-100): Technical audio clarity
        - **speech_clarity_score** (0-100): Speaker articulation and understandability
        - **transcription_confidence** (0-100): Confidence in accuracy of transcript
        - **analysis_reliability** (0-100): Overall AI analysis confidence
        - **noise_level**: "low" / "medium" / "high"
        - **overall_score** (0-100): Composite quality score
        - **confidence_scores_by_speaker**: Array with score per speaker
          * speaker: exact name used in transcript
          * score (0-100): confidence in analyzing that speaker
        - **overall_confidence** (0-100): Combined confidence score

        CRITICAL: Use EXACT speaker names from the audio transcript. Do not truncate or modify names.

        Return ONLY valid JSON — no markdown, no extra text.

        {{
            "sentiment_analysis": {{
                "overall_sentiment": "mixed",
                "sentiment_score": 65,
                "sentiment_distribution": {{
                    "positive": 60,
                    "negative": 15,
                    "neutral": 25
                }},
                "emotional_tones": ["concerned", "cooperative", "hopeful"]
            }},
            "quality_scores": {{
                "audio_quality_score": 70,
                "speech_clarity_score": 75,
                "transcription_confidence": 80,
                "analysis_reliability": 78,
                "noise_level": "medium",
                "overall_score": 75,
                "confidence_scores_by_speaker": [
                    {{"speaker": "Suhani", "score": 85}},
                    {{"speaker": "User", "score": 78}}
                ],
                "overall_confidence": 82
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
            logger.error(f"Error in audio scores and sentiment analysis: {str(e)}")
            return {
                "sentiment_analysis": {"overall_sentiment": "neutral", "sentiment_score": 0},
                "quality_scores": {"overall_score": 0, "overall_confidence": 0},
                "error": str(e)
            }

    async def generate_summary_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Generate summary from transcript (parallel call 2)
        """
        transcript_text = transcript_data.get("transcript", "")

        prompt = f"""
        You are an HR analyst. Provide a comprehensive summary of this onboarding feedback call transcript in {primary_language}.

        Transcript: "{transcript_text}"

        Include:
        - executive_summary: 2-3 sentence overview of the call
        - call_purpose: purpose of the call (e.g., "First-day onboarding check-in")
        - employee_name: name of the employee/user if mentioned, else "Unknown"
        - agent_name: name of the HR agent if mentioned, else "Unknown"
        - organization: company/organization name if mentioned, else "Unknown"
        - key_takeaways: list of 3-5 most important points from the call
        - overall_experience_rating: "excellent" / "good" / "mixed" / "poor"
        - call_outcome: "resolved" / "follow_up_required" / "escalation_needed" / "incomplete"
        - confidence_score (0-100): confidence in this summary

        Format as JSON:
        {{
            "executive_summary": "...",
            "call_purpose": "...",
            "employee_name": "...",
            "agent_name": "...",
            "organization": "...",
            "key_takeaways": ["...", "..."],
            "overall_experience_rating": "mixed",
            "call_outcome": "follow_up_required",
            "confidence_score": 85
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

    async def analyze_issues_and_action_items(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Extract issues, concerns, and action items from transcript (parallel call 3)
        """
        transcript_text = transcript_data.get("transcript", "")

        prompt = f"""
        You are an expert HR analyst. Analyze this feedback call transcript and extract all issues, concerns, and action items.

        Transcript: "{transcript_text}"

        ## ISSUES & CONCERNS:
        For each issue found, provide:
        - issue_title: short descriptive title
        - description: detailed explanation of the issue
        - raised_by: who raised it ("employee", "agent", or speaker name)
        - priority: "high" / "medium" / "low"
        - category: e.g., "documentation", "onboarding", "scheduling", "communication", "technical", "process", "other"
        - timestamp: approximate timestamp from transcript if available, else null

        ## ACTION ITEMS:
        For each action item:
        - task: what needs to be done
        - assignee: who should do it ("HR", "employee", "manager", or specific name)
        - deadline: urgency — "immediate" / "this_week" / "this_month" / null
        - priority: "high" / "medium" / "low"

        ## CALL QUALITY ISSUES:
        Technical issues with the call itself:
        - call_dropped: true/false
        - connection_issues: true/false
        - incomplete_conversation: true/false
        - notes: brief description of any call quality issues

        Format as JSON:
        {{
            "issues": [
                {{
                    "issue_title": "...",
                    "description": "...",
                    "raised_by": "employee",
                    "priority": "high",
                    "category": "documentation",
                    "timestamp": "02:15"
                }}
            ],
            "total_issues_count": 2,
            "high_priority_count": 1,
            "action_items": [
                {{
                    "task": "...",
                    "assignee": "HR",
                    "deadline": "immediate",
                    "priority": "high"
                }}
            ],
            "call_quality_issues": {{
                "call_dropped": false,
                "connection_issues": false,
                "incomplete_conversation": false,
                "notes": ""
            }}
        }}
        """

        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Issues and action items extraction")
            return result
        except Exception as e:
            logger.error(f"Error extracting issues: {str(e)}")
            return {"issues": [], "action_items": [], "error": str(e)}

    async def analyze_onboarding_experience(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Analyze onboarding/employee experience from transcript (parallel call 4)
        """
        transcript_text = transcript_data.get("transcript", "")

        prompt = f"""
        You are an expert HR onboarding analyst. Analyze the employee's onboarding experience from this feedback call transcript.

        Transcript: "{transcript_text}"

        ## ONBOARDING EXPERIENCE DIMENSIONS (each scored 0-100):

        1. **process_clarity_score**: How clearly the onboarding process was communicated
        2. **communication_score**: Quality of communication between HR and employee
        3. **support_availability_score**: How supported the employee felt
        4. **documentation_readiness_score**: Availability and completeness of required documents
        5. **cultural_integration_score**: Employee's integration into company culture
        6. **overall_onboarding_score**: Composite onboarding experience score

        ## EXPERIENCE HIGHLIGHTS:
        - positives: list of things the employee appreciated or liked
        - concerns: list of things the employee was worried about or had issues with
        - suggestions: any improvement suggestions mentioned by the employee

        ## EMPLOYEE ENGAGEMENT:
        - engagement_level: "high" / "medium" / "low" — how engaged the employee seems
        - first_impression: employee's first impression of the company
        - likelihood_to_stay: "high" / "medium" / "low" — based on tone and content

        Format as JSON:
        {{
            "onboarding_scores": {{
                "process_clarity_score": 70,
                "communication_score": 80,
                "support_availability_score": 75,
                "documentation_readiness_score": 40,
                "cultural_integration_score": 85,
                "overall_onboarding_score": 70
            }},
            "experience_highlights": {{
                "positives": ["..."],
                "concerns": ["..."],
                "suggestions": ["..."]
            }},
            "employee_engagement": {{
                "engagement_level": "medium",
                "first_impression": "...",
                "likelihood_to_stay": "high"
            }}
        }}
        """

        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Onboarding experience analysis")
            return result
        except Exception as e:
            logger.error(f"Error analyzing onboarding experience: {str(e)}")
            return {"onboarding_scores": {}, "experience_highlights": {}, "error": str(e)}

    async def analyze_agent_performance_from_audio(self, gemini_file: Any, primary_language: str) -> Dict[str, Any]:
        """
        Analyze HR agent's call performance from audio (parallel call 5)
        """
        prompt = f"""
        You are an expert call quality analyst. Analyze the HR agent's performance in this feedback call audio.
        Language: {primary_language}

        ## AGENT PERFORMANCE DIMENSIONS (each scored 0-100):

        1. **professionalism_score**: How professional and courteous the agent was
        2. **empathy_score**: How empathetic and understanding the agent was toward the employee's concerns
        3. **clarity_score**: How clearly the agent communicated and explained things
        4. **active_listening_score**: How well the agent listened and acknowledged the employee
        5. **problem_acknowledgment_score**: How well the agent acknowledged and committed to resolving issues
        6. **call_handling_score**: Overall call management, pacing, and structure
        7. **overall_agent_score**: Composite agent performance score

        ## AGENT BEHAVIOR:
        - strengths: list of what the agent did well
        - improvement_areas: list of areas needing improvement
        - communication_style: brief description (e.g., "warm and professional", "formal and brief")
        - follow_through: "strong" / "adequate" / "weak" — whether agent committed to next steps

        Format as JSON:
        {{
            "agent_scores": {{
                "professionalism_score": 85,
                "empathy_score": 80,
                "clarity_score": 75,
                "active_listening_score": 70,
                "problem_acknowledgment_score": 75,
                "call_handling_score": 72,
                "overall_agent_score": 76
            }},
            "agent_behavior": {{
                "strengths": ["..."],
                "improvement_areas": ["..."],
                "communication_style": "...",
                "follow_through": "adequate"
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
            logger.info("Completed: Agent performance analysis from audio")
            return result
        except Exception as e:
            logger.error(f"Error analyzing agent performance: {str(e)}")
            return {"agent_scores": {}, "agent_behavior": {}, "error": str(e)}

    async def extract_topics_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Extract key topics discussed from transcript (parallel call 6)
        """
        transcript_text = transcript_data.get("transcript", "")

        prompt = f"""
        Identify and analyze the key topics discussed in this feedback call transcript.

        Transcript: "{transcript_text}"

        For each topic found, provide:
        - topic: topic name
        - importance_score (0-100): how important/significant this topic was in the call
        - mentions: approximate number of times it came up
        - context: brief context about how it was discussed
        - sentiment: "positive" / "negative" / "neutral" — employee's sentiment about this topic

        Also provide:
        - topic_categories: list of broad category labels that apply (e.g., "onboarding", "documentation", "culture", "scheduling")
        - most_discussed_topic: name of the topic with most discussion

        Format as JSON:
        {{
            "key_topics": [
                {{
                    "topic": "Document Submission",
                    "importance_score": 90,
                    "mentions": 3,
                    "context": "Employee mentioned pending transcript submission required by HR",
                    "sentiment": "negative"
                }}
            ],
            "topic_categories": ["onboarding", "documentation"],
            "most_discussed_topic": "Document Submission"
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

    async def analyze_speakers_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Analyze speaker details from transcript (parallel call 7)
        """
        transcript_text = transcript_data.get("transcript", "")
        segments = transcript_data.get("segments", [])

        prompt = f"""
        Analyze speaker information from this feedback call transcript in {primary_language}.

        Transcript: "{transcript_text}"
        Segments: {json.dumps(segments)}

        For each speaker:
        - speaker_name: exact name from transcript
        - role: "hr_agent" / "employee" / "manager" / "other"
        - speaking_time_percentage: estimated % of total speaking time
        - turn_count: number of speaking turns
        - characteristics: tone, pace, communication style description
        - engagement_level: "high" / "medium" / "low"

        Also provide:
        - speaker_count: total number of unique speakers
        - interaction_quality: brief description of the overall interaction dynamic

        Format as JSON:
        {{
            "speaker_count": 2,
            "speakers": [
                {{
                    "speaker_name": "Suhani",
                    "role": "hr_agent",
                    "speaking_time_percentage": 45,
                    "turn_count": 8,
                    "characteristics": "professional, warm, concise",
                    "engagement_level": "high"
                }},
                {{
                    "speaker_name": "User",
                    "role": "employee",
                    "speaking_time_percentage": 55,
                    "turn_count": 7,
                    "characteristics": "cooperative but hesitant at times",
                    "engagement_level": "medium"
                }}
            ],
            "interaction_quality": "Cordial and professional with some communication interruptions"
        }}
        """

        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Speaker analysis from transcript")
            return result
        except Exception as e:
            logger.error(f"Error analyzing speakers: {str(e)}")
            return {"speaker_count": 0, "speakers": [], "error": str(e)}

    async def analyze_segment_sentiment(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Segment-level sentiment timeline from transcript (parallel call 8)
        """
        transcript_text = transcript_data.get("transcript", "")
        segments = transcript_data.get("segments", [])

        prompt = f"""
        Provide segment-wise sentiment analysis for this feedback call transcript in {primary_language}.

        Transcript: "{transcript_text}"
        Segments: {json.dumps(segments)}

        For each logical segment (2-3 minute blocks), provide:
        - start_time: segment start (MM:SS)
        - end_time: segment end (MM:SS)
        - sentiment: "positive" / "negative" / "neutral"
        - sentiment_score (0-100): strength of the sentiment
        - content_summary: brief 1-line summary of what was discussed

        Also provide:
        - sentiment_timeline: narrative description of how sentiment changed across the call

        Format as JSON:
        {{
            "segments": [
                {{
                    "start_time": "0:00",
                    "end_time": "1:30",
                    "sentiment": "positive",
                    "sentiment_score": 75,
                    "content_summary": "Introduction and welcome — friendly tone established"
                }}
            ],
            "sentiment_timeline": "Started positively, became mixed when employee raised document concern, partially recovered at end"
        }}
        """

        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Segment sentiment analysis")
            return result
        except Exception as e:
            logger.error(f"Error in segment sentiment analysis: {str(e)}")
            return {"segments": [], "sentiment_timeline": "", "error": str(e)}

    # ─────────────────────────────────────────────────────────────
    # MAIN ANALYSIS ORCHESTRATOR
    # ─────────────────────────────────────────────────────────────

    async def analyze_audio(self, s3_url: str) -> Dict[str, Any]:
        """
        Main method to perform comprehensive feedback call analysis
        """
        start_time = datetime.now()
        file_path = None

        try:
            # Step 1: Download audio
            logger.info("Step 1: Downloading audio from S3...")
            file_path = await self.download_audio_from_s3(s3_url)

            # Step 2: Upload to Gemini
            logger.info("Step 2: Uploading file to Gemini...")
            gemini_file = self.upload_file_to_gemini(file_path)

            # Step 3: Cleanup temp file
            logger.info("Step 3: Cleaning up temporary file...")
            self.cleanup_temp_file(file_path)
            file_path = None

            # Step 4: Detect language
            logger.info("Step 4: Detecting primary language...")
            primary_language = self.detect_primary_language(gemini_file)

            # Step 5: Generate transcript
            logger.info("Step 5: Generating transcript...")
            transcript_data = self.generate_transcript(gemini_file, primary_language)

            # Step 6: Validate transcript
            logger.info("Step 6: Validating feedback call content...")
            validation_result = self.validate_feedback_call(transcript_data)

            results = {"transcript": transcript_data}

            if validation_result.get("is_valid_feedback_call", False):
                logger.info("✓ Validation passed — proceeding with full analysis...")
                logger.info("Step 7: Running parallel analysis tasks...")

                async def run_feedback_analysis():
                    tasks = [
                        self.analyze_scores_and_sentiment_from_audio(gemini_file, primary_language),   # 0 - audio
                        self.analyze_agent_performance_from_audio(gemini_file, primary_language),       # 1 - audio
                        self.generate_summary_from_transcript(transcript_data, primary_language),       # 2 - transcript
                        self.analyze_issues_and_action_items(transcript_data, primary_language),        # 3 - transcript
                        self.analyze_onboarding_experience(transcript_data, primary_language),          # 4 - transcript
                        self.extract_topics_from_transcript(transcript_data, primary_language),         # 5 - transcript
                        self.analyze_speakers_from_transcript(transcript_data, primary_language),       # 6 - transcript
                        self.analyze_segment_sentiment(transcript_data, primary_language),              # 7 - transcript
                    ]
                    analysis_results = await asyncio.gather(*tasks, return_exceptions=True)

                    return {
                        "quality":            analysis_results[0] if not isinstance(analysis_results[0], Exception) else {"error": str(analysis_results[0])},
                        "agent_performance":  analysis_results[1] if not isinstance(analysis_results[1], Exception) else {"error": str(analysis_results[1])},
                        "summary":            analysis_results[2] if not isinstance(analysis_results[2], Exception) else {"error": str(analysis_results[2])},
                        "issues":             analysis_results[3] if not isinstance(analysis_results[3], Exception) else {"error": str(analysis_results[3])},
                        "onboarding":         analysis_results[4] if not isinstance(analysis_results[4], Exception) else {"error": str(analysis_results[4])},
                        "topics":             analysis_results[5] if not isinstance(analysis_results[5], Exception) else {"error": str(analysis_results[5])},
                        "speakers":           analysis_results[6] if not isinstance(analysis_results[6], Exception) else {"error": str(analysis_results[6])},
                        "segment_sentiment":  analysis_results[7] if not isinstance(analysis_results[7], Exception) else {"error": str(analysis_results[7])},
                    }

                parallel_results = await run_feedback_analysis()
                quality = parallel_results.get("quality", {})

                results.update({
                    "summary":           parallel_results.get("summary", {}),
                    "sentiment":         quality.get("sentiment_analysis", {}),
                    "quality_scores":    quality.get("quality_scores", {}),
                    "agent_performance": parallel_results.get("agent_performance", {}),
                    "issues":            parallel_results.get("issues", {}),
                    "onboarding":        parallel_results.get("onboarding", {}),
                    "topics":            parallel_results.get("topics", {}),
                    "speakers":          parallel_results.get("speakers", {}),
                    "segment_sentiment": parallel_results.get("segment_sentiment", {}),
                })

                logger.info("✓ Full feedback call analysis completed")

            else:
                logger.warning(f"✗ Validation failed: {validation_result.get('content_type')} — {validation_result.get('reason')}")
                logger.info("→ Returning transcript only...")

                results.update({
                    "summary":           {"executive_summary": "Analysis skipped — not a valid feedback call", "key_takeaways": []},
                    "sentiment":         {"overall_sentiment": "neutral", "sentiment_score": 0},
                    "quality_scores":    {"overall_score": 0},
                    "agent_performance": {"agent_scores": {}, "agent_behavior": {}},
                    "issues":            {"issues": [], "action_items": []},
                    "onboarding":        {"onboarding_scores": {}, "experience_highlights": {}},
                    "topics":            {"key_topics": [], "topic_categories": []},
                    "speakers":          {"speaker_count": 0, "speakers": []},
                    "segment_sentiment": {"segments": [], "sentiment_timeline": ""},
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
                    "status": "completed_full_analysis" if validation_result.get("is_valid_feedback_call", False) else "completed_transcript_only",
                    "analysis_version": "1.0",
                    "analysis_type": "feedback_call"
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
        """
        Analyze audio directly from uploaded file / form data
        """
        start_time = datetime.now()
        file_path = None

        try:
            logger.info("Step 1: Saving uploaded audio file...")

            if not filename:
                filename = f"feedback_audio_{int(time.time())}.mp3"

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

            logger.info("Step 6: Validating feedback call...")
            validation_result = self.validate_feedback_call(transcript_data)

            results = {"transcript": transcript_data}

            if validation_result.get("is_valid_feedback_call", False):
                logger.info("✓ Validation passed — running full analysis...")

                async def run_analysis():
                    tasks = [
                        self.analyze_scores_and_sentiment_from_audio(gemini_file, primary_language),
                        self.analyze_agent_performance_from_audio(gemini_file, primary_language),
                        self.generate_summary_from_transcript(transcript_data, primary_language),
                        self.analyze_issues_and_action_items(transcript_data, primary_language),
                        self.analyze_onboarding_experience(transcript_data, primary_language),
                        self.extract_topics_from_transcript(transcript_data, primary_language),
                        self.analyze_speakers_from_transcript(transcript_data, primary_language),
                        self.analyze_segment_sentiment(transcript_data, primary_language),
                    ]
                    analysis_results = await asyncio.gather(*tasks, return_exceptions=True)

                    return {
                        "quality":            analysis_results[0] if not isinstance(analysis_results[0], Exception) else {"error": str(analysis_results[0])},
                        "agent_performance":  analysis_results[1] if not isinstance(analysis_results[1], Exception) else {"error": str(analysis_results[1])},
                        "summary":            analysis_results[2] if not isinstance(analysis_results[2], Exception) else {"error": str(analysis_results[2])},
                        "issues":             analysis_results[3] if not isinstance(analysis_results[3], Exception) else {"error": str(analysis_results[3])},
                        "onboarding":         analysis_results[4] if not isinstance(analysis_results[4], Exception) else {"error": str(analysis_results[4])},
                        "topics":             analysis_results[5] if not isinstance(analysis_results[5], Exception) else {"error": str(analysis_results[5])},
                        "speakers":           analysis_results[6] if not isinstance(analysis_results[6], Exception) else {"error": str(analysis_results[6])},
                        "segment_sentiment":  analysis_results[7] if not isinstance(analysis_results[7], Exception) else {"error": str(analysis_results[7])},
                    }

                parallel_results = await run_analysis()
                quality = parallel_results.get("quality", {})

                results.update({
                    "summary":           parallel_results.get("summary", {}),
                    "sentiment":         quality.get("sentiment_analysis", {}),
                    "quality_scores":    quality.get("quality_scores", {}),
                    "agent_performance": parallel_results.get("agent_performance", {}),
                    "issues":            parallel_results.get("issues", {}),
                    "onboarding":        parallel_results.get("onboarding", {}),
                    "topics":            parallel_results.get("topics", {}),
                    "speakers":          parallel_results.get("speakers", {}),
                    "segment_sentiment": parallel_results.get("segment_sentiment", {}),
                })

                logger.info("✓ Full analysis completed")

            else:
                logger.warning(f"✗ Validation failed: {validation_result.get('content_type')}")
                results.update({
                    "summary":           {"executive_summary": "Analysis skipped", "key_takeaways": []},
                    "sentiment":         {"overall_sentiment": "neutral", "sentiment_score": 0},
                    "quality_scores":    {"overall_score": 0},
                    "agent_performance": {"agent_scores": {}, "agent_behavior": {}},
                    "issues":            {"issues": [], "action_items": []},
                    "onboarding":        {"onboarding_scores": {}, "experience_highlights": {}},
                    "topics":            {"key_topics": [], "topic_categories": []},
                    "speakers":          {"speaker_count": 0, "speakers": []},
                    "segment_sentiment": {"segments": [], "sentiment_timeline": ""},
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
                    "status": "completed_full_analysis" if validation_result.get("is_valid_feedback_call", False) else "completed_transcript_only",
                    "analysis_version": "1.0",
                    "analysis_type": "feedback_call"
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


async def feedback_flow(url: str, client: str = "unknown"):
    """Main feedback call analysis flow"""
    try:
        analyzer = FeedbackCallAnalyzer()
        result = await analyzer.analyze_audio(url)

        analysis_status = result.get("metadata", {}).get("status")

        if analysis_status == "completed_full_analysis":
            print("Step 1: Full analysis completed — generating comprehensive feedback report...")
            report_response = generate_and_upload_feedback_report(result)
            print("Step 2: Feedback report generated")
        elif analysis_status == "completed_transcript_only":
            print("Step 1: Transcript-only — generating transcript report...")
            refactored_result = filter_json(result)
            report_response = generate_and_upload_feedback_report(refactored_result)
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


async def feedback_flow_from_file(audio_file, filename: str = None):
    """Feedback flow for analyzing audio from uploaded file (form data)"""
    try:
        analyzer = FeedbackCallAnalyzer()
        result = await analyzer.analyze_audio_from_file(audio_file, filename)

        analysis_status = result.get("metadata", {}).get("status")

        if analysis_status == "completed_full_analysis":
            print("Step 1: Full analysis completed — generating comprehensive feedback report...")
            report_response = generate_and_upload_feedback_report(result)
            print("Step 2: Feedback report generated")
        elif analysis_status == "completed_transcript_only":
            print("Step 1: Transcript-only — generating transcript report...")
            refactored_result = filter_json(result)
            report_response = generate_and_upload_feedback_report(refactored_result)
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
    s3_audio_url = "https://drive.google.com/uc?export=download&id=1p5RZ45TKi5F1cVPJ33BKSNMkPkY0_erU"
    asyncio.run(feedback_flow(s3_audio_url))