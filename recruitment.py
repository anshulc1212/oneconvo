import os
import tempfile
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from report import generate_and_upload_report
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import time

from dotenv import load_dotenv
load_dotenv()

from google import genai
from snapfind import update_usage, get_audio_duration_minutes  # ← NEW

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
            elif isinstance(value, float) and key.endswith('_score') or key in ['score', 'sentiment_score', 'overall_confidence']:
                converted[key] = convert_to_percentage(value)
            else:
                converted[key] = value
        return converted
    elif isinstance(data, list):
        return [convert_scores_to_percentage(item) for item in data]
    else:
        return data

class HRRecruitmentAudioAnalyzer:
    def __init__(self, api_key: str = None):
        """
        Initialize the HR recruitment audio analyzer with Google Gemini API
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.temp_dir = tempfile.mkdtemp()
        
        # Model configuration with fallback
        self.primary_model = "gemini-2.5-flash"
        self.fallback_model = "gemini-2.0-flash"
        self.current_model = self.primary_model
        self.max_retries = 2
        
        
        self.language_codes = {
            'english': 'en',
            'hindi': 'hi',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'chinese': 'zh',
            'japanese': 'ja',
            'korean': 'ko',
            'arabic': 'ar',
            'russian': 'ru',
            'portuguese': 'pt',
            'italian': 'it'
        }

    def _generate_content_with_fallback(self, prompt: str, gemini_file: Any = None, config: dict = None) -> Any:
        """
        Generate content with automatic fallback to secondary model on 503 errors
        """
        models_to_try = [self.primary_model, self.fallback_model]
        last_exception = None
        
        for attempt, model in enumerate(models_to_try):
            try:
                logger.info(f"Attempting API call with model: {model} (attempt {attempt + 1}/{len(models_to_try)})")
                
                contents = [prompt]
                if gemini_file is not None:
                    contents.append(gemini_file)
                
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config or {}
                )
                
                if self.current_model != model:
                    logger.info(f"Successfully switched to fallback model: {model}")
                    self.current_model = model
                
                return response
                
            except Exception as e:
                error_str = str(e)
                last_exception = e
                
                if "503" in error_str and "overloaded" in error_str.lower():
                    logger.warning(f"Model {model} is overloaded (503 error). Trying fallback...")
                    if attempt < len(models_to_try) - 1:
                        time.sleep(1)
                        continue
                    else:
                        logger.error(f"All models failed. Last error: {error_str}")
                        raise Exception(f"All models unavailable. Last error: {error_str}")
                else:
                    logger.error(f"Non-503 error encountered: {error_str}")
                    raise e
        
        raise last_exception

    async def download_audio_from_s3(self, s3_url: str, filename: str = None) -> str:
        """
        Download audio file from S3 URL and save to temp directory
        """
        if not filename:
            filename = f"audio_{int(time.time())}.mp3"
        
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
        Generate a detailed, accurate transcript of this audio in the exact language(s) spoken.

        SPEAKER IDENTIFICATION (CRITICAL):
        - PRIMARY: Extract actual names mentioned in the audio (e.g., "Ashok Kumar", "अशोक कुमार", "Dr. Smith")
        - Listen carefully for:
        * Direct introductions: "मेरा नाम राज है", "I'm Sarah", "This is Mohammad speaking"
        * Third-party introductions: "अशोक कुमार जी से बात हो रही है", "We're joined by Lisa"
        * Name mentions during conversation: "राज जी, आप क्या सोचते हैं?", "As John mentioned..."
        * Professional titles with names: "Dr. Patel", "Professor Martinez"
        - Use the FIRST identified name consistently as speaker_name throughout
        - ONLY use generic identifiers (Speaker 1, Speaker 2) if absolutely no names are mentioned after careful listening
        - Maintain consistent speaker identification across all segments

        TRANSCRIPTION ACCURACY (CRITICAL - HIGHEST PRIORITY):
        - LISTEN CAREFULLY: Pay extreme attention to exact pronunciation and phonetics
        - Word-by-word precision in the original language and script
        - BRAND NAMES & PROPER NOUNS: 
        * Listen multiple times to verify spelling
        * Consider context clues (company names, product names, technical terms)
        * Examples: "Snapfind" vs "Snapfine", "Google" not "Googol", "iPhone" not "IPhone","Onetab" not "Onetap"
        * If uncertain, prefer the more common/logical spelling based on context
        - HOMOPHONE AWARENESS:
        * Distinguish between similar-sounding words carefully
        * Use context to determine correct word: "their/there/they're", "to/too/two"
        * Verify technical terminology and jargon
        - Preserve:
        * Exact pronunciation, dialectal variations, accents
        * Fillers and natural speech patterns (um, uh, आ, उम्म)
        * Incomplete sentences, stutters, repetitions if significant
        * Code-switching within sentences
        - Capture emotional tone indicators: [laughs], [sighs], [pauses], [background noise]
        - Handle overlapping speech: [speaking simultaneously]
        - DOUBLE-CHECK: Review each segment for accuracy before finalizing
        -

        CONTEXT-AWARE TRANSCRIPTION:
        - Use surrounding context to disambiguate unclear words
        - Consider:
        * Topic of conversation (business, technical, casual)
        * Industry-specific terminology
        * Logical flow of discussion
        * Previously mentioned terms and concepts
        - When in doubt between similar-sounding words, choose the one that makes semantic sense

        TIMESTAMP PRECISION:
        - Format: MM:SS (e.g., "00:00", "01:45", "12:30")
        - Mark at natural segment breaks (speaker changes, topic shifts, pauses >2 seconds)
        - Approximate if exact timing unavailable, but maintain logical progression

        MULTILINGUAL HANDLING:
        - Detect and label language for EACH segment individually
        - Note code-switching: if a segment mixes languages, indicate primary + secondary
        - Use ISO language codes or clear names: "Hindi", "English", "Spanish", "Hindi-English mix"
        - Preserve original script (Devanagari, Latin, Arabic, etc.)

        FULL TRANSCRIPT GENERATION:
        - Compile complete, flowing transcript with speaker names and timestamps
        - Format for readability while maintaining accuracy
        - Example: "[00:15] Ashok Kumar: नमस्ते, मैं अशोक कुमार बोल रहा हूँ।"

        JSON OUTPUT REQUIREMENTS (CRITICAL):
        - Generate VALID, parseable JSON only
        - Escape all special characters: quotes (\"), backslashes (\\), newlines (\n)
        - NO trailing commas in arrays or objects
        - Ensure proper Unicode encoding for non-Latin scripts
        - Validate structure before output

        Required JSON structure:
        {{
            "transcript": "Complete formatted transcript with [MM:SS] Speaker Name: text for each segment, maintaining chronological order and readability",
            "segments": [
                {{
                    "speaker_name": "Actual name from audio or Speaker 1/2 only if no names found",
                    "text": "Exact transcription in original language with proper escaping - VERIFY SPELLING OF ALL PROPER NOUNS",
                    "start_time": "MM:SS format timestamp",
                    "language": "Language code or name (e.g., 'Hindi', 'English', 'Hindi-English mix')"
                }}
            ]
        }}

        QUALITY CHECKLIST BEFORE OUTPUT:
        ✓ All speaker names are real names from audio (or explicitly generic if none found)
        ✓ Timestamps are sequential and logical
        ✓ Language tags are accurate per segment
        ✓ ALL PROPER NOUNS AND BRAND NAMES VERIFIED FOR CORRECT SPELLING
        ✓ Homophones disambiguated using context
        ✓ Technical terms and jargon spelled correctly
        ✓ JSON is valid (no syntax errors, trailing commas, unescaped quotes)
        ✓ Original language and script preserved accurately
        ✓ Full transcript matches segmented version
        ✓ Each segment reviewed for phonetic accuracy
                """
        
        try:
            response = self._generate_content_with_fallback(
                prompt, 
                gemini_file,
                config={"response_mime_type": "application/json"}
            )
            print(safe_json_loads(response.text))
            return safe_json_loads(response.text)
        except Exception as e:
            logger.error(f"Error generating transcript: {str(e)}")
            return {"transcript": "", "segments": []}

    def decide_transcript_validity(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide if transcript contains valid interview or recruitment content
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Analyze this transcript to determine if it contains a meaningful recruitment interview or HR discussion.
        
        Transcript: "{transcript_text}"
        
        Check for:
        1. Interview structure (questions and answers, professional dialogue)
        2. Number of speakers and interaction patterns
        3. Content type (interview, screening call, HR discussion vs noise, music, casual chat)
        4. Presence of job-related discussion
        
        Classify the content as:
        - VALID_INTERVIEW: Contains legitimate recruitment interview content
        - CASUAL_CONVERSATION: Personal/non-professional conversation
        - MUSIC: Primarily music or songs
        - NOISE: Primarily background noise, static, or unclear audio
        - MONOLOGUE: Single person speaking without interaction
        - INSUFFICIENT_CONTENT: Too short or lacks substantial content
        
        Format as JSON:
        {{
            "is_valid_interview": true/false,
            "content_type": "VALID_INTERVIEW/CASUAL_CONVERSATION/MUSIC/NOISE/MONOLOGUE/INSUFFICIENT_CONTENT",
            "validation_details": {{
                "speaker_count_estimate": 2,
                "has_interview_structure": true,
                "contains_job_discussion": true,
                "content_duration_sufficient": true
            }},
            "reason": "Clear interview conversation between interviewer and candidate",
            "recommendation": "PROCEED/SKIP"
        }}
        """
        
        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            validation_result = safe_json_loads(response.text)
            logger.info(f"Transcript validation: {validation_result.get('content_type')} - {'PASS' if validation_result.get('is_valid_interview') else 'FAIL'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating transcript: {str(e)}")
            return {
                "is_valid_interview": False,
                "content_type": "INSUFFICIENT_CONTENT",
                "recommendation": "SKIP",
                "error": str(e)
            }

    async def analyze_interview_quality_from_audio(self, gemini_file: Any, primary_language: str) -> Dict[str, Any]:
        """
        Analyze interview quality and candidate assessment from audio (parallel call 1)
        """
        prompt = f"""
        You are an expert HR recruitment analyst. Analyze this interview recording comprehensively.

        ## HR RECRUITMENT CONTEXT:

        **Interview Purpose**: Assess candidate suitability for a position
        **Key Evaluation Areas**:
        - Candidate communication skills and professionalism
        - Technical/domain knowledge demonstration
        - Cultural fit and soft skills
        - Interview quality and interviewer effectiveness
        - Overall hiring recommendation

        ## ANALYSIS FRAMEWORK:

        ### Sentiment Analysis (Candidate Impression):
        **CRITICAL**: Sentiment reflects overall candidate impression and hiring potential

        - **overall_sentiment**: "positive" (strong hire), "negative" (reject), "neutral" (undecided)
        - **sentiment_score** (0-100): **HIRING CONFIDENCE SCORE**
          * **90-100**: Strong hire - excellent fit, highly recommended
          * **80-89**: Good hire - solid candidate, recommended with confidence
          * **70-79**: Promising candidate - positive indicators, likely to recommend
          * **60-69**: Adequate candidate - meets requirements but has concerns
          * **50-59**: Neutral - mixed signals, requires further evaluation
          * **40-49**: Weak candidate - significant concerns or gaps
          * **30-39**: Poor fit - multiple red flags
          * **20-29**: Not recommended - clearly unsuitable
          * **0-19**: Strong reject - completely unqualified or unprofessional

        - **sentiment_distribution**: % of interview time in positive vs negative vs neutral tone
        - **emotional_tones**: ["confident", "nervous", "professional", "engaging", "evasive", "knowledgeable"]

        ### Quality Assessment Metrics:
        - **audio_quality_score** (0-100): Technical audio clarity
        - **speech_clarity_score** (0-100): Speaker articulation and clarity
        - **transcription_confidence** (0-100): AI confidence in transcript accuracy
        - **analysis_reliability** (0-100): Overall confidence in analysis
        - **noise_level**: "low/medium/high" - background interference
        - **overall_score** (0-100): Composite quality score

        ### Speaker-Wise Confidence Scores:
        - **confidence_scores_by_speaker**: Array of confidence for each speaker
          * **speaker**: Exact name (e.g., "Interviewer", "Candidate", "John Doe")
          * **score** (0-100): Confidence in analyzing this speaker's responses
        
        - **overall_confidence** (0-100): Combined analysis confidence

        **CRITICAL REQUIREMENTS**:
        - Extract EXACT speaker names from transcript
        - DO NOT truncate names
        - Match speaker names exactly across all components
        - Return ONLY valid JSON - no markdown, no extra text

        **JSON OUTPUT FORMAT:**
        {{
            "sentiment_analysis": {{
                "overall_sentiment": "positive/negative/neutral",
                "sentiment_score": 75,
                "sentiment_distribution": {{
                    "positive": 60,
                    "negative": 10,
                    "neutral": 30
                }},
                "emotional_tones": ["confident", "professional", "nervous"]
            }},
            "quality_scores": {{
                "audio_quality_score": 85,
                "speech_clarity_score": 80,
                "transcription_confidence": 85,
                "analysis_reliability": 82,
                "noise_level": "low",
                "overall_score": 83,
                "confidence_scores_by_speaker": [
                    {{
                        "speaker": "Interviewer",
                        "score": 90
                    }},
                    {{
                        "speaker": "Candidate", 
                        "score": 75
                    }}
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
            logger.info("Completed: Audio-based interview quality analysis")
            return result
        except Exception as e:
            logger.error(f"Error in interview quality analysis: {str(e)}")
            return {
                "sentiment_analysis": {"overall_sentiment": "neutral", "sentiment_score": 0},
                "quality_scores": {"overall_score": 0, "confidence_score": 0},
                "error": str(e)
            }

    async def generate_summary_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Generate interview summary from transcript (parallel call 2)
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Provide a comprehensive summary of this recruitment interview in {primary_language}.
        
        Transcript: "{transcript_text}"
        
        Include:
        - Interview overview and position discussed
        - Key discussion topics
        - Important responses from candidate
        - Notable strengths and concerns
        - Interview structure analysis
        - Applied role/position (extract the exact job title/role the candidate is applying for)
        
        Format as JSON:
        {{
            "summary": "detailed interview summary",
            "key_points": ["candidate's background", "technical discussion", "behavioral questions"],
            "main_topics": ["experience review", "technical skills", "cultural fit"],
            "interview_structure": "structured with behavioral and technical questions",
            "position_discussed": "role/position if mentioned",
            "applied_role": "exact job title/role candidate is applying for (e.g., 'Software Engineer', 'Senior Python Developer', 'Data Scientist')"
        }}
        """
        
        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Summary from transcript")
            return result
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {"summary": "", "key_points": [], "main_topics": [], "error": str(e)}

    async def extract_topics_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Extract interview topics from transcript (parallel call 3)
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Extract key topics and themes discussed in this recruitment interview in {primary_language}.
        
        Transcript: "{transcript_text}"
        
        Identify:
        - Main discussion topics (technical skills, experience, soft skills)
        - Recurring themes
        - Skills and competencies discussed
        - Topic importance for hiring decision
        
        Format as JSON:
        {{
            "key_topics": [
                {{
                    "topic": "Python programming experience",
                    "importance_score": 85,
                    "mentions": 5,
                    "context": "discussed projects and frameworks"
                }}
            ],
            "topic_categories": ["technical_skills", "experience", "soft_skills", "cultural_fit"]
        }}
        """
        
        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            result = convert_scores_to_percentage(result)
            logger.info("Completed: Topic extraction from transcript")
            return result
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return {"key_topics": [], "topic_categories": [], "error": str(e)}

    async def analyze_speakers_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Analyze speakers from transcript (parallel call 4)
        """
        transcript_text = transcript_data.get("transcript", "")
        segments = transcript_data.get("segments", [])
        
        prompt = f"""
        Analyze speaker information from this interview transcript in {primary_language}.
        
        Transcript: "{transcript_text}"
        Segments: {json.dumps(segments)}
        
        Identify:
        - Interviewer and candidate identification
        - Number of interviewers if multiple
        - Speaking patterns and communication style
        - Interaction quality
        
        Format as JSON:
        {{
            "speaker_count": 2,
            "speakers": [
                {{
                    "speaker_name": "Interviewer",
                    "speaking_time_percentage": 45,
                    "turn_count": 20,
                    "characteristics": "professional, structured questioning"
                }},
                {{
                    "speaker_name": "Candidate",
                    "speaking_time_percentage": 55,
                    "turn_count": 20,
                    "characteristics": "articulate, detailed responses"
                }}
            ],
            "interaction_pattern": "balanced conversation/interviewer-led/candidate-dominated"
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

    async def extract_action_items_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Extract action items and next steps from transcript (parallel call 5)
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Extract action items, next steps, and key decisions from this interview in {primary_language}.
        
        Transcript: "{transcript_text}"
        
        Identify:
        - Follow-up actions mentioned
        - Next steps in recruitment process
        - Additional information requested
        - Decisions or commitments made
        
        Format as JSON:
        {{
            "action_items": [
                {{
                    "task": "Send portfolio samples",
                    "assignee": "Candidate",
                    "deadline": "by end of week",
                    "priority": "high"
                }}
            ],
            "next_steps": ["Technical round scheduled", "Reference check pending"],
            "follow_ups": ["Candidate to provide references", "HR to send offer details"]
        }}
        """
        
        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Action items extraction from transcript")
            return result
        except Exception as e:
            logger.error(f"Error extracting action items: {str(e)}")
            return {"action_items": [], "next_steps": [], "error": str(e)}

    async def analyze_emotions_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Analyze emotions from transcript (parallel call 6)
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Perform detailed emotion analysis of this interview in {primary_language}.
        
        Transcript: "{transcript_text}"
        
        Analyze:
        - Primary emotions detected
        - Emotional intensity (0-100 scale)
        - Speaker-specific emotions
        - Candidate's emotional state
        
        Format as JSON:
        {{
            "primary_emotions": ["confident", "nervous", "professional"],
            "emotion_intensity": {{
                "confident": 70,
                "nervous": 30,
                "professional": 85
            }},
            "speaker_emotions": {{
                "Interviewer": ["professional", "engaged"],
                "Candidate": ["confident", "slightly_nervous", "enthusiastic"]
            }},
            "candidate_emotional_state": "Overall confident with some nervousness, appropriate for interview context"
        }}
        """
        
        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Emotion analysis from transcript")
            return result
        except Exception as e:
            logger.error(f"Error analyzing emotions: {str(e)}")
            return {"primary_emotions": [], "emotion_intensity": {}, "error": str(e)}

    async def analyze_segment_sentiment_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Analyze segment sentiment from transcript (parallel call 7)
        """
        transcript_text = transcript_data.get("transcript", "")
        segments = transcript_data.get("segments", [])
        
        prompt = f"""
        Analyze sentiment for different segments of this interview in {primary_language}.
        
        Transcript: "{transcript_text}"
        Segments: {json.dumps(segments)}
        
        Provide:
        - Segment-wise sentiment analysis with scores as percentages (0-100)
        - Sentiment flow throughout interview
        
        Format as JSON:
        {{
            "segments": [
                {{
                    "start_time": "0:00",
                    "end_time": "2:30",
                    "sentiment": "positive",
                    "sentiment_score": 80,
                    "content_summary": "Introduction and rapport building - candidate confident"
                }}
            ],
            "sentiment_timeline": "Started strong, dipped during technical questions, recovered well at end"
        }}
        """
        
        try:
            response = self._generate_content_with_fallback(
                prompt,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Segment sentiment analysis from transcript")
            return result
        except Exception as e:
            logger.error(f"Error analyzing segment sentiment: {str(e)}")
            return {"segments": [], "sentiment_timeline": "", "error": str(e)}
    
    async def analyze_candidate_assessment_from_audio(self, gemini_file: Any, primary_language: str) -> Dict[str, Any]:
        """
        Comprehensive candidate assessment from audio (parallel call 8)
        """
        prompt = f"""
        You are an expert HR recruitment analyst. Perform a comprehensive candidate assessment based on this interview recording.

        ## CANDIDATE ASSESSMENT FRAMEWORK:

        ### 1. Communication Skills (0-100):
        **Definition**: Candidate's ability to articulate thoughts clearly and professionally
        **Scoring Criteria**:
        - **90-100**: Exceptional - articulate, clear, concise, professional language
        - **70-80**: Strong - good communication, minor areas for improvement
        - **50-60**: Adequate - understandable but needs improvement
        - **30-40**: Below average - unclear, rambling, or unprofessional
        - **0-20**: Poor - very difficult to understand, highly unprofessional

        **Key Indicators**:
        - Clarity of expression
        - Professional vocabulary
        - Response structure and coherence
        - Active listening demonstrated

        ### 2. Technical Competency (0-100):
        **Definition**: Demonstrated knowledge and expertise in required domain
        **Scoring Criteria**:
        - **90-100**: Expert level - deep knowledge, practical experience evident
        - **70-80**: Strong competency - solid understanding, good examples
        - **50-60**: Adequate - basic knowledge, some gaps
        - **30-40**: Weak - significant knowledge gaps
        - **0-20**: Insufficient - lacks required knowledge

        **Key Indicators**:
        - Accurate technical responses
        - Depth of knowledge
        - Practical experience examples
        - Problem-solving approach

        ### 3. Cultural Fit (0-100):
        **Definition**: Alignment with company values and work culture
        **Scoring Criteria**:
        - **90-100**: Excellent fit - values align perfectly
        - **70-80**: Good fit - generally aligned
        - **50-60**: Moderate fit - some alignment
        - **30-40**: Poor fit - misalignment evident
        - **0-20**: Not a fit - values conflict

        **Key Indicators**:
        - Attitude and demeanor
        - Team collaboration mindset
        - Company research demonstrated
        - Value alignment

        ### 4. Overall Hiring Recommendation (0-100):
        **Definition**: Final recommendation score for hiring decision
        **Scoring Criteria**:
        - **90-100**: Strong hire - highly recommend
        - **70-80**: Hire - recommend with confidence
        - **50-60**: Maybe - requires discussion
        - **30-40**: Likely no - concerns outweigh positives
        - **0-20**: Do not hire - clearly unsuitable

        **Key Indicators**:
        - Overall interview performance
        - Red flags or concerns
        - Strengths vs weaknesses balance
        - Fit for specific role

        Analyze the audio and return detailed assessment with specific evidence.

        Format as JSON:
        {{
            "candidate_assessment": {{
                "communication_skills": {{
                    "score": 85,
                    "evidence": ["Clear articulation", "Professional language", "Good listening"],
                    "strengths": ["Concise responses", "Good examples"],
                    "weaknesses": ["Occasional filler words"],
                    "examples": ["Clearly explained project at 5:30", "Articulate response at 12:45"]
                }},
                "technical_competency": {{
                    "score": 78,
                    "evidence": ["Solid Python knowledge", "Good system design understanding"],
                    "strengths": ["Strong in frameworks", "Good problem-solving"],
                    "weaknesses": ["Limited cloud experience", "Needs growth in architecture"],
                    "skill_areas": {{
                        "programming": 85,
                        "system_design": 70,
                        "tools_and_tech": 75
                    }}
                }},
                "cultural_fit": {{
                    "score": 82,
                    "evidence": ["Team-oriented responses", "Growth mindset evident"],
                    "alignment_indicators": ["Values collaboration", "Open to feedback", "Continuous learner"],
                    "concerns": ["May need support in fast-paced environment"],
                    "company_research": "Demonstrated good understanding of company"
                }},
                "overall_recommendation": {{
                    "score": 80,
                    "recommendation": "HIRE",
                    "confidence_level": "High",
                    "key_strengths": ["Strong technical foundation", "Excellent communication", "Good cultural fit"],
                    "key_concerns": ["Limited experience in X", "Needs development in Y"],
                    "red_flags": [],
                    "next_steps": ["Technical deep-dive round", "Team fit assessment"]
                }}
            }},
            "interviewer_quality": {{
                "interview_structure_score": 85,
                "question_quality": "Good mix of behavioral and technical",
                "engagement_level": 90,
                "areas_for_improvement": ["Could probe deeper on technical details"]
            }},
            "overall_interview_score": 82,
            "summary": "Strong candidate with good technical skills and excellent communication. Recommend proceeding to next round.",
            "hiring_decision_factors": [
                "Strong technical fundamentals",
                "Excellent culture fit",
                "Good growth potential",
                "Minor experience gaps addressable with training"
            ]
        }}
        """
        
        try:
            response = self._generate_content_with_fallback(
                prompt,
                gemini_file,
                config={"response_mime_type": "application/json"}
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Candidate assessment from audio")
            return result
        except Exception as e:
            logger.error(f"Error in candidate assessment: {str(e)}")
            return {
                "candidate_assessment": {
                    "communication_skills": {"score": 0},
                    "technical_competency": {"score": 0},
                    "cultural_fit": {"score": 0},
                    "overall_recommendation": {"score": 0}
                },
                "overall_interview_score": 0,
                "error": str(e)
            }

    async def analyze_audio(self, s3_url: str) -> Dict[str, Any]:
        """
        Main method to perform comprehensive recruitment interview analysis
        """
        start_time = datetime.now()
        file_path = None
        duration_minutes = None  # ← NEW
        
        try:
            # Step 1: Download audio from S3
            logger.info("Step 1: Downloading audio from S3...")
            file_path = await self.download_audio_from_s3(s3_url)

            # Step 1b: Measure audio duration for usage tracking  ← NEW
            logger.info("Step 1b: Measuring audio duration...")
            duration_minutes = get_audio_duration_minutes(file_path)
            logger.info(f"Audio duration: {duration_minutes} min")
            
            # Step 2: Upload to Gemini and get file ID
            logger.info("Step 2: Uploading file to Gemini...")
            gemini_file = self.upload_file_to_gemini(file_path)
            
            # Step 3: Remove audio from temp directory
            logger.info("Step 3: Cleaning up temporary file...")
            self.cleanup_temp_file(file_path)
            file_path = None
            
            # Step 4: Detect language
            logger.info("Step 4: Detecting primary language...")
            primary_language = self.detect_primary_language(gemini_file)
            
            # Step 5: Generate transcript
            logger.info("Step 5: Generating transcript...")
            transcript_data = self.generate_transcript(gemini_file, primary_language)
            
            # Step 6: Decide transcript validity
            logger.info("Step 6: Validating interview content...")
            validation_result = self.decide_transcript_validity(transcript_data)
            
            # Step 7: Conditional analysis based on validation
            results = {"transcript": transcript_data}
            
            if validation_result.get("is_valid_interview", False):
                logger.info("✓ Interview validation passed - proceeding with full analysis...")
                
                # Step 8: Parallel analysis calls
                logger.info("Step 8: Running parallel analysis...")
                
                async def run_analysis():
                    tasks = [
                        self.analyze_interview_quality_from_audio(gemini_file, primary_language),
                        self.analyze_candidate_assessment_from_audio(gemini_file, primary_language),
                        self.generate_summary_from_transcript(transcript_data, primary_language),
                        self.extract_topics_from_transcript(transcript_data, primary_language),
                        self.analyze_speakers_from_transcript(transcript_data, primary_language),
                        self.extract_action_items_from_transcript(transcript_data, primary_language),
                        self.analyze_emotions_from_transcript(transcript_data, primary_language),
                        self.analyze_segment_sentiment_from_transcript(transcript_data, primary_language),
                    ]
                    
                    analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    return {
                        "interview_quality": analysis_results[0] if not isinstance(analysis_results[0], Exception) else {"error": str(analysis_results[0])},
                        "candidate_assessment": analysis_results[1] if not isinstance(analysis_results[1], Exception) else {"error": str(analysis_results[1])},
                        "summary": analysis_results[2] if not isinstance(analysis_results[2], Exception) else {"error": str(analysis_results[2])},
                        "topics": analysis_results[3] if not isinstance(analysis_results[3], Exception) else {"error": str(analysis_results[3])},
                        "speakers": analysis_results[4] if not isinstance(analysis_results[4], Exception) else {"error": str(analysis_results[4])},
                        "action_items": analysis_results[5] if not isinstance(analysis_results[5], Exception) else {"error": str(analysis_results[5])},
                        "emotions": analysis_results[6] if not isinstance(analysis_results[6], Exception) else {"error": str(analysis_results[6])},
                        "segment_sentiment": analysis_results[7] if not isinstance(analysis_results[7], Exception) else {"error": str(analysis_results[7])}
                    }
                
                parallel_results = await run_analysis()
                
                interview_quality = parallel_results.get("interview_quality", {})
                candidate_assessment = parallel_results.get("candidate_assessment", {})
                
                results.update({
                    "summary": parallel_results.get("summary", {}),
                    "sentiment": interview_quality.get("sentiment_analysis", {}),
                    "topics": parallel_results.get("topics", {}),
                    "speakers": parallel_results.get("speakers", {}),
                    "segment_sentiment": parallel_results.get("segment_sentiment", {}),
                    "quality_scores": interview_quality.get("quality_scores", {}),
                    "action_items": parallel_results.get("action_items", {}),
                    "emotions": parallel_results.get("emotions", {}),
                    "candidate_assessment": candidate_assessment.get("candidate_assessment", {}),
                    "interviewer_quality": candidate_assessment.get("interviewer_quality", {}),
                    "overall_interview_score": candidate_assessment.get("overall_interview_score", 0),
                    "hiring_recommendation": candidate_assessment.get("candidate_assessment", {}).get("overall_recommendation", {})
                })
                
                logger.info("✓ Full parallel analysis completed")
                
            else:
                logger.warning(f"✗ Interview validation failed: {validation_result.get('content_type')} - {validation_result.get('reason')}")
                logger.info("→ Returning transcript only...")
                
                results.update({
                    "summary": {"summary": "Analysis skipped - not a valid interview", "key_points": [], "main_topics": []},
                    "sentiment": {"overall_sentiment": "neutral", "sentiment_score": 0, "note": "Analysis skipped"},
                    "topics": {"key_topics": [], "topic_categories": [], "note": "Analysis skipped"},
                    "speakers": {"speaker_count": 0, "speakers": [], "note": "Analysis skipped"},
                    "segment_sentiment": {"segments": [], "sentiment_timeline": "Analysis skipped"},
                    "quality_scores": {"overall_score": 0, "confidence_score": 0, "note": "Analysis skipped"},
                    "action_items": {"action_items": [], "next_steps": [], "note": "Analysis skipped"},
                    "candidate_assessment": {"overall_recommendation": {"score": 0, "recommendation": "NOT_ANALYZED"}, "note": "Analysis skipped"},
                    "overall_interview_score": 0
                })
            
            # Final response structure
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            final_results = {
                "validation": validation_result,
                "metadata": {
                    "processing_time_seconds": processing_time,
                    "timestamp": start_time.isoformat(),
                    "primary_language": primary_language,
                    "file_url": s3_url,
                    "duration_minutes": duration_minutes,   # ← NEW
                    "status": "completed_full_analysis" if validation_result.get("is_valid_interview", False) else "completed_transcript_only",
                    "analysis_version": "2.0_HR_RECRUITMENT",
                    "analysis_type": "recruitment_interview",
                    "model_used": self.current_model
                },
                "language_info": {
                    "primary_language": primary_language,
                    "multilingual_content": "detected" if primary_language in ["mixed", "multilingual"] else "single_language",
                    "supported_languages": list(self.language_codes.keys())
                },
                "analysis_results": results
            }
            
            logger.info(f"✓ Interview analysis completed in {processing_time:.2f} seconds using {self.current_model}")
            return final_results
            
        except Exception as e:
            logger.error(f"✗ Error in interview analysis: {str(e)}")
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
        Analyze audio directly from file/form data without S3 download
        """
        start_time = datetime.now()
        file_path = None
        
        try:
            logger.info("Step 1: Saving uploaded audio file...")
            
            if not filename:
                filename = f"audio_{int(time.time())}.mp3"
            
            file_path = os.path.join(self.temp_dir, filename)
            
            if hasattr(audio_file, 'read'):
                with open(file_path, 'wb') as f:
                    f.write(audio_file.read())
            elif isinstance(audio_file, bytes):
                with open(file_path, 'wb') as f:
                    f.write(audio_file)
            else:
                raise ValueError("audio_file must be a file-like object or bytes")
            
            logger.info(f"Audio saved successfully: {file_path}")
            
            logger.info("Step 2: Uploading file to Gemini...")
            gemini_file = self.upload_file_to_gemini(file_path)
            
            logger.info("Step 3: Cleaning up temporary file...")
            self.cleanup_temp_file(file_path)
            file_path = None
            
            logger.info("Step 4: Detecting primary language...")
            primary_language = self.detect_primary_language(gemini_file)
            
            logger.info("Step 5: Generating transcript...")
            transcript_data = self.generate_transcript(gemini_file, primary_language)
            
            logger.info("Step 6: Validating interview content...")
            validation_result = self.decide_transcript_validity(transcript_data)
            
            results = {"transcript": transcript_data}
            
            if validation_result.get("is_valid_interview", False):
                logger.info("✓ Interview validation passed - proceeding with full analysis...")
                logger.info("Step 8: Running parallel analysis...")
                
                async def run_analysis():
                    tasks = [
                        self.analyze_interview_quality_from_audio(gemini_file, primary_language),
                        self.analyze_candidate_assessment_from_audio(gemini_file, primary_language),
                        self.generate_summary_from_transcript(transcript_data, primary_language),
                        self.extract_topics_from_transcript(transcript_data, primary_language),
                        self.analyze_speakers_from_transcript(transcript_data, primary_language),
                        self.extract_action_items_from_transcript(transcript_data, primary_language),
                        self.analyze_emotions_from_transcript(transcript_data, primary_language),
                        self.analyze_segment_sentiment_from_transcript(transcript_data, primary_language),
                    ]
                    
                    analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    return {
                        "interview_quality": analysis_results[0] if not isinstance(analysis_results[0], Exception) else {"error": str(analysis_results[0])},
                        "candidate_assessment": analysis_results[1] if not isinstance(analysis_results[1], Exception) else {"error": str(analysis_results[1])},
                        "summary": analysis_results[2] if not isinstance(analysis_results[2], Exception) else {"error": str(analysis_results[2])},
                        "topics": analysis_results[3] if not isinstance(analysis_results[3], Exception) else {"error": str(analysis_results[3])},
                        "speakers": analysis_results[4] if not isinstance(analysis_results[4], Exception) else {"error": str(analysis_results[4])},
                        "action_items": analysis_results[5] if not isinstance(analysis_results[5], Exception) else {"error": str(analysis_results[5])},
                        "emotions": analysis_results[6] if not isinstance(analysis_results[6], Exception) else {"error": str(analysis_results[6])},
                        "segment_sentiment": analysis_results[7] if not isinstance(analysis_results[7], Exception) else {"error": str(analysis_results[7])}
                    }
                
                parallel_results = await run_analysis()
                
                interview_quality = parallel_results.get("interview_quality", {})
                candidate_assessment = parallel_results.get("candidate_assessment", {})
                
                results.update({
                    "summary": parallel_results.get("summary", {}),
                    "sentiment": interview_quality.get("sentiment_analysis", {}),
                    "topics": parallel_results.get("topics", {}),
                    "speakers": parallel_results.get("speakers", {}),
                    "segment_sentiment": parallel_results.get("segment_sentiment", {}),
                    "quality_scores": interview_quality.get("quality_scores", {}),
                    "action_items": parallel_results.get("action_items", {}),
                    "emotions": parallel_results.get("emotions", {}),
                    "candidate_assessment": candidate_assessment.get("candidate_assessment", {}),
                    "interviewer_quality": candidate_assessment.get("interviewer_quality", {}),
                    "overall_interview_score": candidate_assessment.get("overall_interview_score", 0),
                    "hiring_recommendation": candidate_assessment.get("candidate_assessment", {}).get("overall_recommendation", {})
                })
                
                logger.info("✓ Full parallel analysis completed")
                
            else:
                logger.warning(f"✗ Interview validation failed: {validation_result.get('content_type')} - {validation_result.get('reason')}")
                logger.info("→ Returning transcript only...")
                
                results.update({
                    "summary": {"summary": "Analysis skipped - not a valid interview", "key_points": [], "main_topics": []},
                    "sentiment": {"overall_sentiment": "neutral", "sentiment_score": 0, "note": "Analysis skipped"},
                    "topics": {"key_topics": [], "topic_categories": [], "note": "Analysis skipped"},
                    "speakers": {"speaker_count": 0, "speakers": [], "note": "Analysis skipped"},
                    "segment_sentiment": {"segments": [], "sentiment_timeline": "Analysis skipped"},
                    "quality_scores": {"overall_score": 0, "confidence_score": 0, "note": "Analysis skipped"},
                    "action_items": {"action_items": [], "next_steps": [], "note": "Analysis skipped"},
                    "candidate_assessment": {"overall_recommendation": {"score": 0, "recommendation": "NOT_ANALYZED"}, "note": "Analysis skipped"},
                    "overall_interview_score": 0
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
                    "status": "completed_full_analysis" if validation_result.get("is_valid_interview", False) else "completed_transcript_only",
                    "analysis_version": "2.0_HR_RECRUITMENT",
                    "analysis_type": "recruitment_interview",
                    "model_used": self.current_model
                },
                "language_info": {
                    "primary_language": primary_language,
                    "multilingual_content": "detected" if primary_language in ["mixed", "multilingual"] else "single_language",
                    "supported_languages": list(self.language_codes.keys())
                },
                "analysis_results": results
            }
            
            logger.info(f"✓ Interview analysis completed in {processing_time:.2f} seconds using {self.current_model}")
            return final_results
            
        except Exception as e:
            logger.error(f"✗ Error in interview analysis: {str(e)}")
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
    """
    Filters the input JSON and keeps only:
      - metadata
      - analysis_results.transcript
    """
    filtered = {}

    if "metadata" in data:
        filtered["metadata"] = data["metadata"]

    if "analysis_results" in data and "transcript" in data["analysis_results"]:
        filtered["analysis_results"] = {
            "transcript": data["analysis_results"]["transcript"]
        }

    return filtered

async def demo_flow_from_file(audio_file, filename: str = None):
    """
    Demo flow for analyzing audio from uploaded file (form data)
    """
    try:
        analyzer = HRRecruitmentAudioAnalyzer()
        result = await analyzer.analyze_audio_from_file(audio_file, filename)
        
        analysis_status = result.get("metadata", {}).get("status")
        
        if analysis_status == "completed_full_analysis":
            print("Step 1: Full interview analysis completed - generating comprehensive report...")
            report_response = generate_and_upload_report(result)
            print("Step 2: Comprehensive interview report generation completed")
        elif analysis_status == "completed_transcript_only":
            print("Step 1: Transcript-only analysis completed - generating transcript report...")
            refactored_result = filter_json(result)
            report_response = generate_and_upload_report(refactored_result)
            print("Step 2: Transcript report generation completed")
        else:
            print("Step 1: Analysis failed - no report generated")
            report_response = None
        
        if report_response:
            result["report_url"] = report_response
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except Exception as e:
        print("Error during analysis:", e)
        return {"error": str(e)}
    
async def hr_flow(url, client: str = "unknown"):   # ← NEW: client param
    try:
        analyzer = HRRecruitmentAudioAnalyzer()
        result = await analyzer.analyze_audio(url)

        # ── Usage tracking ──────────────────────────────
        duration_minutes = result.get("metadata", {}).get("duration_minutes")
        if client and client != "unknown" and duration_minutes:
            update_usage(client, duration_minutes)
        # ────────────────────────────────────────────────

        analysis_status = result.get("metadata", {}).get("status")
        
        if analysis_status == "completed_full_analysis":
            print("Step 1: Full interview analysis completed - generating comprehensive report...")
            report_response = generate_and_upload_report(result)
            print("Step 2: Comprehensive interview report generation completed")
        elif analysis_status == "completed_transcript_only":
            print("Step 1: Transcript-only analysis completed - generating transcript report...")
            refactored_result = filter_json(result)
            print("Refactored Result for Transcript Report:", json.dumps(refactored_result, indent=2, ensure_ascii=False))
            report_response = generate_and_upload_report(refactored_result)
            print("Step 2: Transcript report generation completed")
            print(f"Content type: {result.get('validation', {}).get('content_type', 'Unknown')}")
        else:
            print("Step 1: Analysis failed - no report generated")
            report_response = None
        
        if report_response:
            result["report_url"] = report_response
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except Exception as e:
        print("Error during analysis:", e)
        return {"error": str(e)}

async def main():
    """
    Example usage of the HR Recruitment Audio Analyzer
    """
    test_urls = [
        "https://example-bucket.s3.amazonaws.com/interviews/interview_001.mp3",
    ]
    
    for s3_audio_url in test_urls:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Analyzing Interview: {s3_audio_url}")
            logger.info(f"{'='*50}")
            
            results = await demo_flow(s3_audio_url)
                
        except Exception as e:
            logger.error(f"Analysis failed for {s3_audio_url}: {str(e)}")

if __name__ == "__main__":
    s3_audio_url = "https://drive.google.com/file/d/1kaSBb35MIWvaqPHvbd6-vcnP4DE42KB_/view"
    asyncio.run(hr_flow(s3_audio_url))