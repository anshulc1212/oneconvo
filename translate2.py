

import os
import tempfile
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from report1 import generate_and_upload_report
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import time

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
            elif isinstance(value, float) and key.endswith('_score') or key in ['score', 'sentiment_score', 'overall_confidence']:
                converted[key] = convert_to_percentage(value)
            else:
                converted[key] = value
        return converted
    elif isinstance(data, list):
        return [convert_scores_to_percentage(item) for item in data]
    else:
        return data

class MultilingualAudioAnalyzer:
    def __init__(self, api_key: str = None):
        """
        Initialize the audio analyzer with Google Gemini API
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass it directly.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.temp_dir = tempfile.mkdtemp()
        
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
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, gemini_file],
            )
            primary_language = response.text.strip().lower()
            logger.info(f"Primary language detected: {primary_language}")
            return primary_language
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return 'english'  # Default fallback

    def generate_transcript(self, gemini_file: Any, primary_language: str) -> Dict[str, Any]:
        """
        Generate detailed transcript with timestamps and speaker identification
        """
        prompt = f"""
        Generate a detailed transcript of this audio in same language as spoken. 
        Include:
        - Complete word-by-word transcription
        - Speaker identification by name
        - Approximate timestamps where possible
        - Handle multilingual content by noting language switches
        CRITICAL: You MUST generate valid JSON. Ensure all quotes are properly escaped and no trailing commas exist.

        SPEAKER IDENTIFICATION RULES:
        - Extract actual names mentioned in audio (like "Ashok Kumar", "अशोक कुमार")
        - Use real names as speaker_name, NOT "Speaker 1" or "Speaker 2"
        - If you hear "अशोक कुमार जी से बात हो रही है" → use "Ashok Kumar" as speaker_name
        - Listen for: introductions, name mentions, addressing by name
        - Only use generic IDs if absolutely no names mentioned

        TRANSCRIPTION REQUIREMENTS:
        - Word-by-word accuracy in original language
        - Proper timestamps (MM:SS format)
        - Language detection per segment
        - Consistent speaker naming throughout
        
        Format as JSON with structure:
        {{
            "transcript": "full transcript text",
            "segments": [
                {{
                    "speaker_name": "name of speaker ",
                    "text": "segment text",
                    "start_time": "approximate time",
                    "language": "detected language for this segment"
                }}
            ]
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, gemini_file],
                config={
                    "response_mime_type": "application/json"
                }
            )
            return safe_json_loads(response.text)
        except Exception as e:
            logger.error(f"Error generating transcript: {str(e)}")
            return {"transcript": "", "segments": []}

    def decide_transcript_validity(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide if transcript contains valid discussion or meeting content
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Analyze this transcript to determine if it contains a meaningful discussion or meeting content.
        
        Transcript: "{transcript_text}"
        
        Check for:
        1. Conversational content (is there actual discussion/dialogue?)
        2. Number of speakers and interactions
        3. Content type (meeting, interview, casual conversation, vs music, noise, monologue, etc.)
        4. Meaningful discussion content
        
        Classify the content as:
        - VALID_DISCUSSION: Contains meaningful conversation/discussion worth analyzing
        - MUSIC: Primarily music or songs
        - NOISE: Primarily background noise, static, or unclear audio
        - MONOLOGUE: Single person speaking without interaction (lectures, presentations)
        - INSUFFICIENT_CONTENT: Too short or lacks substantial content
        
        Format as JSON:
        {{
            "is_valid_discussion": true/false,
            "content_type": "VALID_DISCUSSION/MUSIC/NOISE/MONOLOGUE/INSUFFICIENT_CONTENT",
            "validation_details": {{
                "speaker_count_estimate": 2,
                "has_conversation": true,
                "content_duration_sufficient": true
            }},
            "reason": "Clear conversation between 2 speakers",
            "recommendation": "PROCEED/SKIP"
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt],
                config={
                    "response_mime_type": "application/json"
                }
            )
            validation_result = safe_json_loads(response.text)
            logger.info(f"Transcript validation: {validation_result.get('content_type')} - {'PASS' if validation_result.get('is_valid_discussion') else 'FAIL'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating transcript: {str(e)}")
            return {
                "is_valid_discussion": False,
                "content_type": "INSUFFICIENT_CONTENT",
                "recommendation": "SKIP",
                "error": str(e)
            }
    
        
        




    # Audio-based analysis (using gemini_file)
    async def analyze_scores_and_sentiment_from_audio(self, gemini_file: Any, primary_language: str) -> Dict[str, Any]:
        """
        Analyze scores and sentiment from audio file (parallel call 1)
        """
        prompt = f"""
                    You are an expert call center quality assurance analyst specializing in **Kiko Live's lead conversion process**. Your task is to provide a comprehensive, data-driven analysis of customer service call recordings focused on converting local retailers (kiranas, pharmacies, stationery shops, etc.) to Kiko Live's digital platform.

                    ## BUSINESS CONTEXT - KIKO LIVE SPECIFICS:

                    **Kiko Live Platform**: A digitization solution for local retailers enabling them to:
                    - Create online stores and sell digitally
                    - Manage inventory and orders online
                    - Accept digital payments
                    - Reach wider customer base beyond physical location
                    - Access business analytics and insights
                    - Get marketing and promotional tools

                    **TARGET AUDIENCE**: Local retailers including:
                    - Kirana store owners (grocery/general stores)
                    - Pharmacy owners
                    - Stationery shop owners
                    - Small retail business owners
                    - Traditional brick-and-mortar merchants

                    **CONVERSION INDICATORS TO LISTEN FOR**:
                    - Interest in going digital/online
                    - Pain points about limited customer reach
                    - Concerns about competition from online platforms
                    - Questions about setup process, costs, or technical requirements
                    - Willingness to try technology solutions
                    - References to wanting more customers or sales
                    - Discussions about current business challenges
                    
                Your analysis must focus on two key business objectives: **customer satisfaction** and **sales/deal closing potential**. Your report will be used to train new agents and improve call quality.

                ## SCORE DEFINITIONS & REFERENCES:

                ### Sentiment Analysis Metrics (BUSINESS OUTCOME FOCUSED):
                **CRITICAL**: Sentiment scores should ONLY reflect deal conversion potential, NOT general emotions
                
                - **overall_sentiment**: Business outcome assessment - "positive" (deal converted/high conversion probability), "negative" (deal rejected/no interest), "neutral" (undecided/needs follow-up)
                - **sentiment_score** (0-100): **DEAL CONVERSION PERCENTAGE** - This is your PRIMARY business metric
                * **90-100**: Customer CONFIRMED purchase, signed contract, gave payment details, said "YES" definitively
                * **80-89**: Customer AGREED in principle, showed strong buying intent, asked about next steps
                * **70-79**: Customer very interested, asked detailed questions, requested proposal/quote
                * **60-69**: Customer showed interest but has concerns, needs to think about it, asked for time
                * **50-59**: Customer neutral, listening but not committed, asking basic questions
                * **40-49**: Customer hesitant, raised objections, showed price sensitivity
                * **30-39**: Customer showed little interest, gave negative feedback about product/service
                * **20-29**: Customer clearly not interested, said "no" but remained polite
                * **0-19**: Customer rejected offer completely, hung up, or was hostile
                
                - **sentiment_distribution**: Percentage of call time spent in deal-positive vs deal-negative vs neutral discussion
                - **emotional_tones**: Focus on deal-relevant emotions like ["interested", "convinced", "hesitant", "rejected", "considering", "agreed"]
           
                ### Quality Assessment Metrics:
                - **audio_quality_score** (0-100): Technical audio clarity - 0 = poor/inaudible/distorted, 50 = acceptable with some issues, 100 = crystal clear audio
                - **speech_clarity_score** (0-100): Participant articulation - 0 = mumbled/unclear speech, 50 = moderately clear, 100 = perfectly articulated
                - **transcription_confidence** (0-100): AI confidence in transcript accuracy - 0 = low confidence/many uncertainties, 50 = moderate confidence, 100 = high confidence in transcription
                - **analysis_reliability** (0-100): Overall confidence in the analysis quality - 0 = unreliable due to poor audio/clarity, 50 = moderately reliable, 100 = highly reliable analysis
                - **noise_level**: Categorical background noise assessment - "low" (minimal interference), "medium" (some interference), "high" (significant interference affecting clarity)
                - **overall_score** (0-100): Composite quality metric combining audio, clarity, and reliability factors
                
                ### Speaker-Wise Confidence Scores:
                **CONFIDENCE DEFINITION**: Confidence measures how reliable and accurate the AI analysis is for each speaker, based on audio quality, speech clarity, and definitiveness of responses. Higher confidence = more trustworthy analysis results.

                **confidence_scores_by_speaker**: Array of individual confidence assessments for each participant
                  - **speaker**: Exact full name detected from transcript (e.g., "Ashish", "Imran", "Customer") - NO TRUNCATION
                  - **score** (0-100): System confidence in analyzing this speaker's speech and responses
                
                **CONFIDENCE SCORING CRITERIA:**
                * **Excellent Confidence (90-100)**: 
                  - Crystal clear audio with no background noise
                  - Speaker articulates clearly with no mumbling
                  - Responses are definitive and unambiguous (clear yes/no answers)
                  - Easy to determine speaker's intent and emotions
                  - Transcription accuracy is very high
                
                * **High Confidence (80-89)**:
                  - Good audio quality with minimal interference
                  - Generally clear speech with occasional unclear words
                  - Most responses are clear, some minor ambiguity
                  - Can reliably assess speaker's sentiment and intent
                  - Transcription mostly accurate
                
                * **Medium Confidence (50-79)**:
                  - Acceptable audio with some background noise or distortion
                  - Speech sometimes unclear or muffled
                  - Some ambiguous responses that are hard to interpret
                  - Moderate certainty about speaker's true intentions
                  - Transcription has some uncertainties
                
                * **Low Confidence (30-49)**:
                  - Poor audio quality with significant interference
                  - Frequent unclear speech or heavy accents
                  - Many ambiguous or incomplete responses
                  - Difficult to accurately assess speaker's sentiment
                  - Transcription reliability questionable
                
                * **Very Low Confidence (0-29)**:
                  - Very poor audio (distorted, breaking up, too quiet)
                  - Speech largely unintelligible or heavily garbled
                  - Cannot determine speaker's responses or intent
                  - Analysis results are unreliable for this speaker
                  - Transcription highly inaccurate or incomplete
                
                **WHY CONFIDENCE MATTERS:**
                - **Business Impact**: Low confidence scores indicate unreliable deal conversion predictions
                - **Training Use**: High confidence calls are better for agent training materials  
                - **Quality Control**: Low confidence alerts need human review before making business decisions
                - **Resource Allocation**: Focus improvement efforts on speakers with consistently low confidence
                
                - **overall_confidence** (0-100): Combined system confidence across all speakers and analysis components
                  - Weighted average of all speaker confidences plus overall call quality factors
                  - Indicates reliability of the entire call analysis for business decision-making

                ### Speaker Analysis Details:
                For each identified speaker, assess:
                - **speech_quality**: Individual audio clarity for that speaker
                - **transcription_accuracy**: Confidence in transcript accuracy for that speaker's words
                - **sentiment_clarity**: How clearly that speaker's emotions/intentions come through
                - **response_definitiveness**: How clear and unambiguous their responses are (especially important for customers)

                ### Business Outcome Scoring Guidelines:
                **SENTIMENT SCORE = DEAL CONVERSION PROBABILITY PERCENTAGE**

                **CONFIRMED DEAL (90-100):**
                - Customer said "Yes, I'll take it" or "Let's proceed" or "Where do I sign?"
                - Provided payment information or signed contract
                - Scheduled implementation/delivery
                - Confirmed purchase decision

                **HIGH CONVERSION PROBABILITY (80-89):**
                - Customer agreed in principle: "This sounds good" "I'm interested"
                - Asked about next steps, contracts, or implementation
                - Showed strong buying signals but needs minor details

                **MEDIUM-HIGH INTEREST (70-79):**
                - Customer asked detailed product questions
                - Requested proposals, quotes, or more information
                - Showed genuine interest but needs more convincing

                **MODERATE INTEREST (60-69):**
                - Customer interested but has concerns or objections
                - Needs to discuss with others or think about it
                - Asking for time to decide

                **LOW INTEREST (30-59):**
                - Customer listening but not engaged
                - Raised significant price or feature objections
                - Showed skepticism about product/service

                **NO DEAL POTENTIAL (0-29):**
                - Customer clearly said "No" or "Not interested"
                - Rejected offer completely
                - Ended call early or was hostile

                ### Universal Percentage Scale:
                - **0-30**: Poor/Critical - Requires immediate attention
                - **40-60**: Below Average - Needs improvement
                - **70-80**: Good - Meets standards
                - **90-100**: Excellent - Exceeds expectations

                **CRITICAL SPEAKER NAME HANDLING REQUIREMENTS:**
                - Extract EXACT FULL NAMES from transcript segments (e.g., "Ashish", "Imran", "Customer")
                - DO NOT truncate names (avoid "ASHI" instead of "Ashish")
                - USE CONSISTENT naming across ALL analysis components
                - If transcript shows "Customer" - use "Customer" in confidence scores
                - If transcript shows "Unknown speaker" - use "Unknown speaker" in confidence scores
                - MATCH speaker names exactly between transcript and confidence analysis

                **IMPORTANT**: 
                - IGNORE general politeness, friendliness, or customer service satisfaction
                - FOCUS ONLY on deal conversion indicators and buying signals
                - Sentiment score should predict: "Will this customer buy?" (0% = No, 100% = Yes)
                - Ensure speaker names in confidence_scores_by_speaker EXACTLY match transcript speaker names
                - **CONFIDENCE = ANALYSIS RELIABILITY PERCENTAGE**: How much you can trust the AI's assessment of each speaker
                - Low confidence speakers need human review; high confidence speakers provide reliable business insights
                - Return ONLY valid JSON. No explanations, no markdown formatting, no additional text.

                **JSON OUTPUT FORMAT:**
                {{
                    "sentiment_analysis": {{
                        "overall_sentiment": "positive/negative/neutral",
                        "sentiment_score": 65,
                        "sentiment_distribution": {{
                            "positive": 40,
                            "negative": 10,
                            "neutral": 50
                        }},
                        "emotional_tones": ["interested", "considering", "hesitant"]
                    }},
                    "quality_scores": {{
                        "audio_quality_score": 80,
                        "speech_clarity_score": 85,
                        "transcription_confidence": 85,
                        "analysis_reliability": 80,
                        "noise_level": "medium",
                        "overall_score": 83,
                        "confidence_scores_by_speaker": [
                            {{
                                "speaker": "Ashish",
                                "score": 85
                            }},
                            {{
                                "speaker": "Customer", 
                                "score": 75
                            }},
                            {{
                                "speaker": "Imran",
                                "score": 80
                            }}
                        ],
                        "overall_confidence": 80
                    }}
                }}
                    
                """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, gemini_file],
                config={
                    "response_mime_type": "application/json"
                }
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Audio-based scores and sentiment analysis")
            return result
        except Exception as e:
            logger.error(f"Error in audio scores and sentiment analysis: {str(e)}")
            return {
                "sentiment_analysis": {"overall_sentiment": "neutral", "sentiment_score": 0},
                "quality_scores": {"overall_score": 0, "confidence_score": 0},
                "error": str(e)
            }

    # Transcript-based analysis methods
    async def generate_summary_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Generate summary from transcript (parallel call 2)
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Provide a comprehensive summary of this transcript content in {primary_language}.
        
        Transcript: "{transcript_text}"
        
        Include:
        - Main topics discussed
        - Key points and highlights
        - Important decisions or conclusions
        - Structure analysis
        
        Format as JSON:
        {{
            "summary": "detailed summary text",
            "key_points": ["point 1", "point 2", ...],
            "main_topics": ["topic 1", "topic 2", ...],
            "duration_analysis": "brief analysis of content structure"
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config={
                    "response_mime_type": "application/json"
                }
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Summary from transcript")
            return result
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {"summary": "", "key_points": [], "main_topics": [], "error": str(e)}
        

    async def extract_topics_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Extract topics from transcript (parallel call 3)
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Extract key topics and themes from this transcript in {primary_language}.
        
        Transcript: "{transcript_text}"
        
        Identify:
        - Main discussion topics
        - Recurring themes
        - Important subjects mentioned
        - Topic importance ranking
        
        Format as JSON:
        {{
            "key_topics": [
                {{
                    "topic": "topic name",
                    "importance_score": 85,
                    "mentions": 5,
                    "context": "brief context"
                }}
            ],
            "topic_categories": ["business", "technical", "personal"]
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config={
                    "response_mime_type": "application/json"
                }
            )
            result = safe_json_loads(response.text)
            # Convert importance scores to percentages
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
        Analyze speaker information from this transcript in {primary_language}.
        
        Transcript: "{transcript_text}"
        Segments: {json.dumps(segments)}
        
        Identify:
        - Speaker names from transcript analysis
        - Number of unique speakers
        - Speaking patterns and characteristics
        - Interaction patterns
        
        Format as JSON:
        {{
            "speaker_count": 2,
            "speakers": [
                {{
                    "speaker_name": "name of speaker",
                    "speaking_time_percentage": 60,
                    "turn_count": 15,
                    "characteristics": "tone, pace, etc."
                }}
            ],
            "interaction_pattern": "alternating/monologue/discussion"
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config={
                    "response_mime_type": "application/json"
                }
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Speaker analysis from transcript")
            return result
        except Exception as e:
            logger.error(f"Error analyzing speakers: {str(e)}")
            return {"speaker_count": 0, "speakers": [], "error": str(e)}

    async def extract_action_items_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Extract action items from transcript (parallel call 5)
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Extract action items, tasks, and decisions from this transcript in {primary_language}.
        
        Transcript: "{transcript_text}"
        
        Identify:
        - Specific action items mentioned
        - Assigned responsibilities
        - Deadlines or timelines
        - Decision points
        
        Format as JSON:
        {{
            "action_items": [
                {{
                    "task": "description of task",
                    "assignee": "person responsible",
                    "deadline": "timeline if mentioned",
                    "priority": "high/medium/low"
                }}
            ],
            "decisions_made": ["decision 1", "decision 2"],
            "follow_ups": ["follow up 1", "follow up 2"]
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config={
                    "response_mime_type": "application/json"
                }
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Action items extraction from transcript")
            return result
        except Exception as e:
            logger.error(f"Error extracting action items: {str(e)}")
            return {"action_items": [], "decisions_made": [], "error": str(e)}

    async def analyze_emotions_from_transcript(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Analyze emotions from transcript (parallel call 6)
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Perform detailed emotion analysis of this transcript in {primary_language}.
        
        Transcript: "{transcript_text}"
        
        Analyze:
        - Primary emotions detected
        - Emotional intensity (0-100 scale)
        - Speaker-specific emotions
        
        Format as JSON:
        {{
            "primary_emotions": ["happy", "confident", "excited"],
            "emotion_intensity": {{
                "happy": 80,
                "confident": 90,
                "excited": 60
            }},
            "speaker_emotions": {{
                "Speaker 1": ["confident", "professional"],
                "Speaker 2": ["engaged", "curious"]
            }}
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config={
                    "response_mime_type": "application/json"
                }
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
        Analyze sentiment for different segments of this transcript in {primary_language}.
        
        Transcript: "{transcript_text}"
        Segments: {json.dumps(segments)}
        
        Provide:
        - Segment-wise sentiment analysis with scores as percentages (0-100)
        - Sentiment changes over time
        
        Format as JSON:
        {{
            "segments": [
                {{
                    "start_time": "0:00",
                    "end_time": "1:30",
                    "sentiment": "positive",
                    "sentiment_score": 80,
                    "content_summary": "brief summary of segment"
                }}
            ],
            "sentiment_timeline": "description of sentiment changes"
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config={
                    "response_mime_type": "application/json"
                }
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Segment sentiment analysis from transcript")
            return result
        except Exception as e:
            logger.error(f"Error analyzing segment sentiment: {str(e)}")
            return {"segments": [], "sentiment_timeline": "", "error": str(e)}
    
    async def analyze_sales_performance_from_audio(self, gemini_file: Any, primary_language: str) -> Dict[str, Any]:
        """
        Analyze sales performance metrics from audio (parallel call 8)
        """
        prompt = f"""
        You are an expert sales performance analyst. Analyze this audio recording of a sales call and evaluate the following key performance indicators:

        ## SALES PERFORMANCE ANALYSIS FRAMEWORK:

        ### 1. Lead Engagement (0-100):
        **Definition**: How engaged and interested is the potential customer/lead?
        **Scoring Criteria**:
        - **90-100**: Highly engaged - asking detailed questions, showing strong interest, actively participating
        - **70-80**: Good engagement - asking questions, showing interest, responsive
        - **50-60**: Moderate engagement - listening but limited interaction, some questions
        - **30-40**: Low engagement - minimal responses, distracted, short answers
        - **0-20**: No engagement - unresponsive, clearly disinterested, attempting to end call

        **Key Indicators**:
        - Question frequency and depth
        - Response quality and length
        - Tone of voice and enthusiasm
        - Time spent discussing product/service details

        ### 2. Intent Signals (0-100):
        **Definition**: Strength of buying intent signals from the lead
        **Scoring Criteria**:
        - **90-100**: Strong intent - asking about pricing, contracts, implementation, next steps
        - **70-80**: Good intent - expressing interest, asking specific product questions
        - **50-60**: Moderate intent - general interest but no specific commitment signals
        - **30-40**: Weak intent - listening but not showing clear purchase interest
        - **0-20**: No intent - explicitly declining, not interested, objecting

        **Key Indicators**:
        - Pricing discussions
        - Timeline inquiries
        - Implementation questions
        - Comparison with alternatives
        - Decision-maker involvement

        ### 3. Rep Effectiveness (0-100):
        **Definition**: How effectively is the sales representative performing?
        **Scoring Criteria**:
        - **90-100**: Excellent - clear communication, handling objections well, building rapport
        - **70-80**: Good - effective presentation, adequate objection handling
        - **50-60**: Average - basic presentation, some missed opportunities
        - **30-40**: Below average - unclear communication, poor objection handling
        - **0-20**: Poor - confusing presentation, unable to handle objections

        **Key Indicators**:
        - Communication clarity
        - Product knowledge demonstration
        - Objection handling skills
        - Rapport building
        - Call structure and flow

        ### 4. Outcome Signals (0-100):
        **Definition**: Likelihood of positive sales outcome based on call conclusion
        **Scoring Criteria**:
        - **90-100**: Deal closed or very likely - commitment made, next steps agreed
        - **70-80**: Likely positive outcome - strong interest, follow-up scheduled
        - **50-60**: Uncertain outcome - mixed signals, requires follow-up
        - **30-40**: Unlikely positive outcome - significant objections, low interest
        - **0-20**: No deal - clear rejection, no follow-up interest

        **Key Indicators**:
        - Call conclusion tone
        - Follow-up commitments
        - Decision timelines discussed
        - Final customer sentiment

        Analyze the audio and return detailed scoring with specific evidence from the call.

        Format as JSON:
        {{
            "sales_performance": {{
                "lead_engagement": {{
                    "score": 85,
                    "evidence": ["Customer asked detailed questions about features", "Engaged in 15-minute detailed discussion"],
                    "key_moments": ["Question about implementation at 3:45", "Pricing inquiry at 8:20"],
                    "improvement_areas": ["Could ask more qualifying questions"]
                }},
                "intent_signals": {{
                    "score": 72,
                    "evidence": ["Asked about pricing", "Inquired about contract terms"],
                    "buying_signals": ["When can we start?", "What are the payment terms?"],
                    "objections_raised": ["Price seems high", "Need to check budget"],
                    "decision_timeline": "2-3 weeks mentioned"
                }},
                "rep_effectiveness": {{
                    "score": 88,
                    "evidence": ["Clear product explanation", "Handled price objection well"],
                    "strengths": ["Good rapport building", "Clear communication", "Product knowledge"],
                    "weaknesses": ["Could have probed budget earlier", "Missed upsell opportunity"],
                    "objection_handling_quality": 80
                }},
                "outcome_signals": {{
                    "score": 75,
                    "evidence": ["Positive call ending", "Follow-up scheduled"],
                    "next_steps_agreed": ["Send proposal by Friday", "Follow-up call next week"],
                    "probability_assessment": "Medium-high likelihood of closing",
                    "timeline_to_decision": "2-3 weeks"
                }}
            }},
            "overall_sales_score": 80,
            "call_quality_rating": "Good - promising lead with good engagement",
            "recommendations": [
                "Follow up with detailed proposal",
                "Address pricing concerns in next call",
                "Involve decision maker in next meeting"
            ]
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, gemini_file],
                config={
                    "response_mime_type": "application/json"
                }
            )
            result = safe_json_loads(response.text)
            logger.info("Completed: Sales performance analysis from audio")
            return result
        except Exception as e:
            logger.error(f"Error in sales performance analysis: {str(e)}")
            return {
                "sales_performance": {
                    "lead_engagement": {"score": 0},
                    "intent_signals": {"score": 0},
                    "rep_effectiveness": {"score": 0},
                    "outcome_signals": {"score": 0}
                },
                "overall_sales_score": 0,
                "error": str(e)
            }

    async def analyze_audio(self, s3_url: str) -> Dict[str, Any]:
        """
        Main method to perform comprehensive audio analysis with new flow
        """
        start_time = datetime.now()
        file_path = None
        
        try:
            # Step 1: Download audio from S3
            logger.info("Step 1: Downloading audio from S3...")
            file_path = await self.download_audio_from_s3(s3_url)
            
            # Step 2: Upload to Gemini and get file ID
            logger.info("Step 2: Uploading file to Gemini...")
            gemini_file = self.upload_file_to_gemini(file_path)
            
            # Step 3: Remove audio from temp directory
            logger.info("Step 3: Cleaning up temporary file...")
            self.cleanup_temp_file(file_path)
            file_path = None  # Mark as cleaned
            
            # Step 4: Detect language
            logger.info("Step 4: Detecting primary language...")
            primary_language = self.detect_primary_language(gemini_file)
            
            # Step 5: Generate transcript
            logger.info("Step 5: Generating transcript...")
            transcript_data = self.generate_transcript(gemini_file, primary_language)
            
            # Step 6: Decide transcript validity
            logger.info("Step 6: Validating transcript content...")
            validation_result = self.decide_transcript_validity(transcript_data)
            
            # Step 7: Conditional analysis based on validation
            results = {"transcript": transcript_data}
            
            if validation_result.get("is_valid_discussion", False):
                logger.info("✓ Transcript validation passed - proceeding with full analysis...")
                
                # Step 8: Parallel analysis calls
                logger.info("Step 8: Running parallel analysis...")
                
                async def run_analysis():
                    # Create tasks for parallel execution
                    tasks = [
                        # Audio-based analysis (1 call)
                        self.analyze_scores_and_sentiment_from_audio(gemini_file, primary_language),
                       

                        # Transcript-based analysis (6 calls)
                        self.generate_summary_from_transcript(transcript_data, primary_language),
                        self.extract_topics_from_transcript(transcript_data, primary_language),
                        self.analyze_speakers_from_transcript(transcript_data, primary_language),
                        self.extract_action_items_from_transcript(transcript_data, primary_language),
                        self.analyze_emotions_from_transcript(transcript_data, primary_language),
                        self.analyze_segment_sentiment_from_transcript(transcript_data, primary_language),
                        self.analyze_sales_performance_from_audio(gemini_file, primary_language),
                    ]
                    
                    # Execute all tasks in parallel
                    analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    return {
                        "scores_sentiment": analysis_results[0] if not isinstance(analysis_results[0], Exception) else {"error": str(analysis_results[0])},
                        "summary": analysis_results[1] if not isinstance(analysis_results[1], Exception) else {"error": str(analysis_results[1])},
                        "topics": analysis_results[2] if not isinstance(analysis_results[2], Exception) else {"error": str(analysis_results[2])},
                        "speakers": analysis_results[3] if not isinstance(analysis_results[3], Exception) else {"error": str(analysis_results[3])},
                        "action_items": analysis_results[4] if not isinstance(analysis_results[4], Exception) else {"error": str(analysis_results[4])},
                        "emotions": analysis_results[5] if not isinstance(analysis_results[5], Exception) else {"error": str(analysis_results[5])},
                        "segment_sentiment": analysis_results[6] if not isinstance(analysis_results[6], Exception) else {"error": str(analysis_results[6])},
                        "sales_performance": analysis_results[7] if not isinstance(analysis_results[7], Exception) else {"error": str(analysis_results[7])}
                    }
                
                parallel_results = await run_analysis()
                
                # Restructure results to match original format
                scores_sentiment = parallel_results.get("scores_sentiment", {})
                results.update({
                    "summary": parallel_results.get("summary", {}),
                    "sentiment": scores_sentiment.get("sentiment_analysis", {}),
                    "topics": parallel_results.get("topics", {}),
                    "speakers": parallel_results.get("speakers", {}),
                    "segment_sentiment": parallel_results.get("segment_sentiment", {}),
                    "scores": scores_sentiment.get("quality_scores", {}),
                    "action_items": parallel_results.get("action_items", {}),
                    "emotions": parallel_results.get("emotions", {}),
                    "sales_performance": parallel_results.get("sales_performance", {})
                })
                
                logger.info("✓ Full parallel analysis completed")
                
            else:
                logger.warning(f"✗ Transcript validation failed: {validation_result.get('content_type')} - {validation_result.get('reason')}")
                logger.info("→ Returning transcript only...")
                
                # Return minimal structure for invalid content
                results.update({
                    "summary": {"summary": "Analysis skipped - invalid content type", "key_points": [], "main_topics": []},
                    "sentiment": {"overall_sentiment": "neutral", "sentiment_score": 0, "note": "Analysis skipped"},
                    "topics": {"key_topics": [], "topic_categories": [], "note": "Analysis skipped"},
                    "speakers": {"speaker_count": 0, "speakers": [], "note": "Analysis skipped"},
                    "segment_sentiment": {"segments": [], "sentiment_timeline": "Analysis skipped"},
                    "scores": {"overall_score": 0, "confidence_score": 0, "note": "Analysis skipped"},
                    "action_items": {"action_items": [], "decisions_made": [], "note": "Analysis skipped"},
                    "sales_performance": {"overall_sales_score": 0, "note": "Analysis skipped"},
                    
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
                    "status": "completed_full_analysis" if validation_result.get("is_valid_discussion", False) else "completed_transcript_only",
                    "analysis_version": "2.0"
                },
                "language_info": {
                    "primary_language": primary_language,
                    "multilingual_content": "detected" if primary_language in ["mixed", "multilingual"] else "single_language",
                    "supported_languages": list(self.language_codes.keys())
                },
                "analysis_results": results
            }
            
            logger.info(f"✓ Audio analysis completed in {processing_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            logger.error(f"✗ Error in audio analysis: {str(e)}")
            return {
                "error": str(e),
                "metadata": {
                    "processing_time_seconds": 0,
                    "timestamp": start_time.isoformat(),
                    "status": "failed"
                }
            }
        
        finally:
            # Clean up if file still exists
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

async def demo_flow(url):
    try:
        analyzer = MultilingualAudioAnalyzer()
        result = await analyzer.analyze_audio(url)
        
        analysis_status = result.get("metadata", {}).get("status")
        
        if analysis_status == "completed_full_analysis":
            print("Step 1: Full analysis completed - generating comprehensive report...")
            report_response = generate_and_upload_report(result)
            print("Step 2: Comprehensive report generation completed")
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
            result["url"] = report_response
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except Exception as e:
        print("Error during analysis:", e)
        return {"error": str(e)}

async def main():
    """
    Example usage of the restructured MultilingualAudioAnalyzer
    """
    test_urls = [
        "https://one-convo.s3.us-west-1.amazonaws.com/uploads/hh2WnHk43.mp3",
    ]
    
    for s3_audio_url in test_urls:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Analyzing: {s3_audio_url}")
            logger.info(f"{'='*50}")
            
            # Perform analysis with new flow
            results = await demo_flow(s3_audio_url)
                
        except Exception as e:
            logger.error(f"Analysis failed for {s3_audio_url}: {str(e)}")

# if __name__ == "__main__":
#     s3_audio_url = "https://one-convo.s3.us-west-1.amazonaws.com/uploads/recordings/1764222832444_1758274616204_1758194984553_record-1758194287221.wav"
#     asyncio.run(demo_flow(s3_audio_url))