import os
import tempfile
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed


import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import time

from exitreport import generate_and_upload_exit_report

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

class ExitInterviewAnalyzer:
    def __init__(self, api_key: str = None):
        """
        Initialize the Exit Interview Analyzer with Google Gemini API
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
                model="gemini-2.5-flash",
                contents=[prompt, gemini_file],
            )
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
        Generate a detailed, accurate transcript of this EXIT INTERVIEW audio in the exact language(s) spoken.

        SPEAKER IDENTIFICATION (CRITICAL):
        - PRIMARY: Extract actual names mentioned in the audio (e.g., "John Smith", "राज कुमार", "Dr. Patel")
        - Listen for:
        * Direct introductions: "I'm Sarah", "मेरा नाम अनिल है", "This is Mohammad"
        * Third-party mentions: "We're speaking with Lisa", "अशोक जी से बात हो रही है"
        * Name references: "राज जी", "As John mentioned", "Thank you, Maria"
        - Use FIRST identified name consistently as speaker_name
        - ONLY use generic identifiers (HR Representative, Employee, Speaker 1) if no names found
        - Typical speakers in exit interviews: HR Representative, Exiting Employee, Manager

        TRANSCRIPTION ACCURACY (CRITICAL):
        - LISTEN CAREFULLY: Exact pronunciation and phonetics
        - Word-by-word precision in original language and script
        - PROPER NOUNS: Company names, department names, person names, locations
        - Context-aware spelling for homophones and similar-sounding words
        - Preserve emotional markers: [sighs], [pauses], [laughs], [emotional]
        - Capture overlapping speech: [speaking simultaneously]

        TIMESTAMP PRECISION:
        - Format: MM:SS (e.g., "00:00", "02:45", "15:30")
        - Mark at natural breaks (speaker changes, topic shifts, pauses >2 seconds)

        MULTILINGUAL HANDLING:
        - Label language for EACH segment
        - Note code-switching if languages mix within segments
        - Preserve original script (Devanagari, Latin, Arabic, etc.)

        JSON OUTPUT (CRITICAL - MUST BE VALID):
        {{
            "transcript": "Complete formatted transcript with [MM:SS] Speaker: text...",
            "segments": [
                {{
                    "speaker_name": "HR Representative",
                    "text": "Exact transcription with proper escaping",
                    "start_time": "00:00",
                    "language": "English"
                }}
            ]
        }}

        Ensure valid JSON: escape quotes, no trailing commas, proper Unicode encoding.
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

    def validate_exit_interview(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate if transcript contains valid exit interview content
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Analyze this transcript to determine if it contains a genuine EXIT INTERVIEW conversation.
        
        Transcript: "{transcript_text}"
        
        Exit interviews typically discuss:
        - Reasons for leaving the organization
        - Work experience and satisfaction
        - Management and team relationships
        - Company culture and environment
        - Feedback and suggestions for improvement
        - Future plans
        
        Classify as:
        - VALID_EXIT_INTERVIEW: Contains legitimate exit interview discussion
        - RECRUITMENT_INTERVIEW: Appears to be hiring interview, not exit
        - CASUAL_CONVERSATION: Non-professional conversation
        - PERFORMANCE_REVIEW: Performance discussion, not exit
        - MUSIC: Primarily music or songs
        - NOISE: Primarily background noise or unclear audio
        - INSUFFICIENT_CONTENT: Too short or lacks substance
        
        Return JSON:
        {{
            "is_valid_exit_interview": true/false,
            "content_type": "VALID_EXIT_INTERVIEW/RECRUITMENT_INTERVIEW/CASUAL_CONVERSATION/etc.",
            "validation_details": {{
                "discusses_departure": true,
                "mentions_reasons_for_leaving": true,
                "contains_feedback": true,
                "has_hr_structure": true,
                "speaker_count_estimate": 2
            }},
            "reason": "Clear exit interview discussion about departure and experience",
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
            logger.info(f"Exit interview validation: {validation_result.get('content_type')} - {'PASS' if validation_result.get('is_valid_exit_interview') else 'FAIL'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating exit interview: {str(e)}")
            return {
                "is_valid_exit_interview": False,
                "content_type": "INSUFFICIENT_CONTENT",
                "recommendation": "SKIP",
                "error": str(e)
            }

    async def analyze_exit_interview_quality(self, gemini_file: Any, primary_language: str) -> Dict[str, Any]:
        """
        Analyze exit interview quality and overall sentiment
        """
        prompt = f"""
        Analyze this EXIT INTERVIEW recording for quality metrics and overall sentiment.

        ## EXIT INTERVIEW CONTEXT:
        This is a conversation where an employee is leaving the organization and providing feedback.

        ## ANALYSIS FRAMEWORK:

        ### Overall Sentiment (Employee Experience):
        **CRITICAL**: Sentiment reflects the employee's overall experience and satisfaction

        - **overall_sentiment**: "positive" (good experience), "negative" (poor experience), "neutral" (mixed)
        - **sentiment_score** (0-100): **EMPLOYEE SATISFACTION SCORE**
          * **90-100**: Excellent experience - highly satisfied, leaving for positive reasons
          * **80-89**: Good experience - generally satisfied, some concerns
          * **70-79**: Satisfactory - adequate experience with room for improvement
          * **60-69**: Mixed experience - both positive and negative aspects
          * **50-59**: Neutral - neither satisfied nor dissatisfied
          * **40-49**: Poor experience - significant dissatisfaction
          * **30-39**: Very poor - major issues and problems
          * **20-29**: Terrible experience - multiple serious problems
          * **0-19**: Extremely negative - toxic environment or severe issues

        - **sentiment_distribution**: % of positive vs negative vs neutral tone during interview
        - **emotional_tones**: ["frustrated", "relieved", "grateful", "disappointed", "angry", "professional"]

        ### Quality Scores:
        - **audio_quality_score** (0-100): Technical audio clarity
        - **speech_clarity_score** (0-100): Speaker articulation
        - **transcription_confidence** (0-100): Confidence in transcript accuracy
        - **analysis_reliability** (0-100): Overall analysis confidence
        - **noise_level**: "low/medium/high"
        - **overall_score** (0-100): Composite quality score

        ### Speaker Confidence:
        - **confidence_scores_by_speaker**: Array with confidence for each speaker
          * **speaker**: Exact name (e.g., "HR Representative", "Exiting Employee")
          * **score** (0-100): Confidence in analyzing this speaker
        - **overall_confidence** (0-100): Combined confidence

        **CRITICAL**: Extract EXACT speaker names, no truncation, match across all components.

        Return ONLY valid JSON:
        {{
            "sentiment_analysis": {{
                "overall_sentiment": "positive/negative/neutral",
                "sentiment_score": 65,
                "sentiment_distribution": {{
                    "positive": 40,
                    "negative": 35,
                    "neutral": 25
                }},
                "emotional_tones": ["professional", "disappointed", "relieved"]
            }},
            "quality_scores": {{
                "audio_quality_score": 85,
                "speech_clarity_score": 80,
                "transcription_confidence": 85,
                "analysis_reliability": 82,
                "noise_level": "low",
                "overall_score": 83,
                "confidence_scores_by_speaker": [
                    {{"speaker": "HR Representative", "score": 90}},
                    {{"speaker": "Exiting Employee", "score": 75}}
                ],
                "overall_confidence": 82
            }}
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
            result = safe_json_loads(response.text)
            logger.info("Completed: Exit interview quality analysis")
            return result
        except Exception as e:
            logger.error(f"Error in quality analysis: {str(e)}")
            return {
                "sentiment_analysis": {"overall_sentiment": "neutral", "sentiment_score": 0},
                "quality_scores": {"overall_score": 0, "overall_confidence": 0},
                "error": str(e)
            }

    async def analyze_exit_reasons(self, gemini_file: Any, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Analyze reasons for leaving with structured hierarchy
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Analyze this EXIT INTERVIEW to extract and categorize reasons for leaving.

        Transcript: "{transcript_text}"

        ## STRUCTURED REASON HIERARCHY:

        ### PRIMARY CATEGORIES:
        1. **Career Growth** - advancement, learning, stagnation
        2. **Compensation** - salary, benefits, equity, bonuses
        3. **Work-Life Balance** - hours, flexibility, burnout, stress
        4. **Management** - leadership, support, communication, conflict
        5. **Company Culture** - values, environment, diversity, inclusion
        6. **Job Role** - responsibilities, alignment, fulfillment
        7. **Team Dynamics** - colleagues, collaboration, conflict
        8. **External Factors** - relocation, family, health, education
        9. **Better Opportunity** - competitive offer, career change
        10. **Company Issues** - financials, direction, stability, changes

        ### ANALYSIS REQUIREMENTS:
        - Identify PRIMARY reason (most impactful)
        - List CONTRIBUTING reasons (secondary factors)
        - Rate impact/importance (0-100) for each reason
        - Extract verbatim quotes as evidence
        - Determine if voluntary or involuntary departure
        - Assess preventability (could company have retained?)

        Return JSON:
        {{
            "departure_type": "voluntary/involuntary/mutual",
            "primary_reason": {{
                "category": "Career Growth",
                "subcategory": "Lack of advancement opportunities",
                "impact_score": 90,
                "description": "Employee felt stagnant with no clear path to promotion",
                "employee_quotes": ["I've been in the same role for 3 years", "No growth opportunities"]
            }},
            "contributing_reasons": [
                {{
                    "category": "Compensation",
                    "subcategory": "Below market salary",
                    "impact_score": 70,
                    "description": "Salary not competitive with market rates",
                    "employee_quotes": ["Received better offer elsewhere"]
                }}
            ],
            "reason_summary": "Employee leaving primarily due to lack of career advancement, compounded by compensation concerns",
            "preventability_assessment": {{
                "could_be_prevented": true,
                "prevention_score": 75,
                "required_actions": ["Clear promotion path", "Compensation review", "Development plan"]
            }},
            "structured_reason_list": [
                "Primary: Career Growth - Lack of advancement (Impact: 90/100)",
                "Contributing: Compensation - Below market (Impact: 70/100)"
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
            result = safe_json_loads(response.text)
            result = convert_scores_to_percentage(result)
            logger.info("Completed: Exit reasons analysis")
            return result
        except Exception as e:
            logger.error(f"Error analyzing exit reasons: {str(e)}")
            return {
                "primary_reason": {},
                "contributing_reasons": [],
                "error": str(e)
            }

    async def identify_key_strengths(self, gemini_file: Any, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Identify key strengths mentioned by exiting employee
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Extract KEY STRENGTHS the employee appreciated about the organization from this exit interview.

        Transcript: "{transcript_text}"

        ## STRENGTH CATEGORIES:
        - **People & Team**: Great colleagues, supportive team, talented people
        - **Management**: Good leadership, supportive managers, clear communication
        - **Culture**: Positive environment, values alignment, inclusivity
        - **Work**: Interesting projects, meaningful work, autonomy
        - **Growth**: Learning opportunities, skill development, mentorship
        - **Benefits**: Good compensation, benefits, perks, flexibility
        - **Company**: Mission, products, market position, reputation

        For each strength:
        - Assign category
        - Rate importance/emphasis (0-100)
        - Extract supporting quotes
        - Note specific examples mentioned

        Return JSON:
        {{
            "key_strengths": [
                {{
                    "category": "People & Team",
                    "strength": "Talented and supportive colleagues",
                    "importance_score": 85,
                    "description": "Employee highly valued team collaboration and colleague relationships",
                    "supporting_quotes": ["Amazing team to work with", "Will miss the colleagues the most"],
                    "specific_examples": ["Weekly team collaboration sessions", "Mentorship from senior developers"]
                }}
            ],
            "strengths_summary": "Employee appreciated talented team, good learning opportunities, and positive culture",
            "top_3_strengths": [
                "Talented and supportive team",
                "Strong learning culture",
                "Meaningful work"
            ],
            "overall_positive_sentiment": 70
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
            result = safe_json_loads(response.text)
            result = convert_scores_to_percentage(result)
            logger.info("Completed: Key strengths identification")
            return result
        except Exception as e:
            logger.error(f"Error identifying strengths: {str(e)}")
            return {"key_strengths": [], "error": str(e)}

    async def identify_concerns(self, gemini_file: Any, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Identify concerns and areas for improvement
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Extract CONCERNS and areas for improvement mentioned in this exit interview.

        Transcript: "{transcript_text}"

        ## CONCERN CATEGORIES:
        - **Management Issues**: Poor leadership, lack of support, micromanagement
        - **Communication Problems**: Unclear expectations, lack of feedback, silos
        - **Process Issues**: Inefficient workflows, bureaucracy, unclear processes
        - **Team Problems**: Conflict, lack of collaboration, toxic individuals
        - **Career Issues**: No growth path, limited learning, skill stagnation
        - **Workload Issues**: Overwork, unrealistic expectations, burnout
        - **Culture Issues**: Lack of diversity, unclear values, poor environment
        - **Resource Issues**: Inadequate tools, budget constraints, staffing

        For each concern:
        - Severity level (low/medium/high/critical)
        - Impact score (0-100)
        - Actionability (can company address?)
        - Supporting quotes
        - Suggested improvements if mentioned

        Return JSON:
        {{
            "concerns": [
                {{
                    "category": "Management Issues",
                    "concern": "Lack of regular feedback and communication",
                    "severity": "high",
                    "impact_score": 80,
                    "actionable": true,
                    "description": "Employee felt disconnected from manager with infrequent check-ins",
                    "supporting_quotes": ["Never knew where I stood", "Feedback only during annual review"],
                    "suggested_improvement": "Implement regular 1-on-1 meetings and continuous feedback",
                    "affects_retention": true
                }}
            ],
            "concerns_summary": "Primary concerns around management communication and career development opportunities",
            "high_priority_concerns": ["Lack of regular feedback", "No clear growth path"],
            "systemic_vs_individual": "systemic",
            "overall_concern_level": 75
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
            result = safe_json_loads(response.text)
            result = convert_scores_to_percentage(result)
            logger.info("Completed: Concerns identification")
            return result
        except Exception as e:
            logger.error(f"Error identifying concerns: {str(e)}")
            return {"concerns": [], "error": str(e)}

    async def identify_red_flags(self, gemini_file: Any, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Identify critical red flags that require immediate attention
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Identify CRITICAL RED FLAGS in this exit interview that indicate serious organizational issues.

        Transcript: "{transcript_text}"

        ## RED FLAG CATEGORIES:

        ### CRITICAL (Immediate Action Required):
        - Harassment or discrimination
        - Hostile work environment
        - Unethical behavior
        - Legal/compliance violations
        - Safety concerns

        ### SERIOUS (Urgent Attention Needed):
        - Toxic management/leadership
        - Systematic unfair treatment
        - Retaliation or intimidation
        - Severe burnout or mental health impact
        - Mass exodus concerns

        ### SIGNIFICANT (Important to Address):
        - High-level dissatisfaction
        - Multiple systemic issues
        - Culture deterioration
        - Loss of key talent indicators
        - Competitive disadvantage

        For each red flag:
        - Severity: critical/serious/significant
        - Urgency: immediate/urgent/important
        - Risk level (0-100)
        - Potential legal/compliance implications
        - Recommended immediate actions
        - Affects other employees?

        Return JSON:
        {{
            "red_flags": [
                {{
                    "category": "Toxic Management",
                    "flag": "Manager created hostile work environment",
                    "severity": "serious",
                    "urgency": "urgent",
                    "risk_score": 90,
                    "description": "Employee reported consistent hostile behavior from direct manager",
                    "supporting_evidence": ["Manager yelled in meetings", "Excluded from important decisions", "Public criticism"],
                    "potential_implications": [
                        "Legal risk - hostile workplace claims",
                        "Team morale impact",
                        "Additional turnover risk"
                    ],
                    "affects_others": true,
                    "estimated_affected_employees": "team of 8",
                    "immediate_actions_required": [
                        "Investigate manager behavior",
                        "Interview remaining team members",
                        "Consider management training or action"
                    ]
                }}
            ],
            "critical_red_flags_count": 1,
            "requires_immediate_action": true,
            "legal_risk_present": true,
            "retention_risk_for_others": "high",
            "executive_notification_required": true,
            "red_flags_summary": "Critical management issue with potential legal implications and team impact",
            "overall_risk_score": 85
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
            result = safe_json_loads(response.text)
            result = convert_scores_to_percentage(result)
            logger.info("Completed: Red flags identification")
            return result
        except Exception as e:
            logger.error(f"Error identifying red flags: {str(e)}")
            return {"red_flags": [], "critical_red_flags_count": 0, "error": str(e)}

    async def generate_exit_assessment(self, gemini_file: Any, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Generate comprehensive exit interview assessment
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Provide a comprehensive ASSESSMENT of this exit interview similar to a call evaluation.

        Transcript: "{transcript_text}"

        ## ASSESSMENT FRAMEWORK:

        ### Interview Quality Assessment:
        - **HR Professionalism** (0-100): How well did HR conduct the interview?
        - **Question Quality** (0-100): Were questions effective and comprehensive?
        - **Employee Openness** (0-100): How candid was the employee?
        - **Depth of Discussion** (0-100): Were topics explored thoroughly?
        - **Action Item Clarity** (0-100): Are next steps clear?

        ### Employee Profile:
        - Role and tenure
        - Performance level
        - Value to organization
        - Rehire eligibility
        - Knowledge transfer status

        ### Interview Completeness:
        - All standard topics covered?
        - Missing critical areas?
        - Follow-up needed?

        ### Insights Quality:
        - Actionable feedback provided?
        - Patterns identified?
        - Organizational learnings?

        Return JSON:
        {{
            "interview_quality_assessment": {{
                "hr_professionalism_score": 85,
                "question_quality_score": 80,
                "employee_openness_score": 75,
                "depth_of_discussion_score": 70,
                "action_item_clarity_score": 80,
                "overall_interview_quality": 78,
                "strengths": ["Good rapport building", "Open-ended questions", "Active listening"],
                "weaknesses": ["Could probe deeper on management issues", "Missed career development discussion"],
                "interviewer_notes": "HR conducted professional interview, employee moderately open"
            }},
            "employee_profile": {{
                "role": "Senior Software Engineer",
                "tenure_years": 3.5,
                "department": "Engineering",
                "performance_level": "high_performer",
                "value_to_organization": 85,
                "rehire_eligible": true,
                "reason_for_eligibility": "Left on good terms, performance excellent",
                "knowledge_transfer_status": "in_progress",
                "replacement_difficulty": "high"
            }},
            "interview_completeness": {{
                "standard_topics_covered": [
                    "Reason for leaving",
                    "Work experience",
                    "Management feedback",
                    "Company culture",
                    "Suggestions for improvement"
                ],
                "missing_topics": ["Career development discussions", "Compensation satisfaction"],
                "follow_up_needed": true,
                "follow_up_areas": ["Technical knowledge transfer", "Project handoff details"]
            }},
            "insights_quality": {{
                "actionable_feedback_provided": true,
                "actionability_score": 75,
                "patterns_identified": ["Management communication gap", "Career path unclear"],
                "organizational_learnings": [
                    "Need better career development framework",
                    "Manager training on regular feedback",
                    "Competitive compensation review needed"
                ],
                "applies_to_broader_team": true,
                "strategic_implications": "May indicate broader retention risk in engineering"
            }},
            "overall_assessment_score": 76,
            "assessment_summary": "Valuable exit interview with actionable insights on management and career development. Employee was moderately open. Key findings suggest systemic issues requiring attention."
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
            result = safe_json_loads(response.text)
            result = convert_scores_to_percentage(result)
            logger.info("Completed: Exit assessment")
            return result
        except Exception as e:
            logger.error(f"Error generating assessment: {str(e)}")
            return {
                "interview_quality_assessment": {"overall_interview_quality": 0},
                "overall_assessment_score": 0,
                "error": str(e)
            }

    async def extract_qa_pairs(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Extract question and answer pairs from transcript
        """
        transcript_text = transcript_data.get("transcript", "")
        segments = transcript_data.get("segments", [])
        
        prompt = f"""
        Extract QUESTION and ANSWER pairs from this exit interview transcript.

        Transcript: "{transcript_text}"
        Segments: {json.dumps(segments)}

        Identify:
        - Questions asked by HR/interviewer
        - Corresponding answers from employee
        - Link questions to answer segments
        - Categorize by topic

        Return JSON:
        {{
            "qa_pairs": [
                {{
                    "question_number": 1,
                    "timestamp": "00:30",
                    "asker": "HR Representative",
                    "question": "What was your primary reason for deciding to leave?",
                    "category": "Reason for Leaving",
                    "answer_timestamp": "00:35",
                    "responder": "Exiting Employee",
                    "answer": "I felt like I wasn't growing in my role anymore...",
                    "answer_sentiment": "negative",
                    "key_points": ["Lack of growth", "Career stagnation"]
                }}
            ],
            "total_questions": 15,
            "questions_by_category": {{
                "Reason for Leaving": 3,
                "Work Experience": 4,
                "Management": 3,
                "Culture": 2,
                "Suggestions": 3
            }},
            "employee_response_quality": "detailed and candid"
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
            result = safe_json_loads(response.text)
            logger.info("Completed: Q&A extraction")
            return result
        except Exception as e:
            logger.error(f"Error extracting Q&A: {str(e)}")
            return {"qa_pairs": [], "total_questions": 0, "error": str(e)}

    async def generate_summary(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Generate executive summary of exit interview
        """
        transcript_text = transcript_data.get("transcript", "")
        
        prompt = f"""
        Generate a concise executive summary of this exit interview.

        Transcript: "{transcript_text}"

        Include:
        - Employee background (role, tenure)
        - Key reason for departure
        - Main points discussed
        - Critical insights
        - Recommended actions

        Return JSON:
        {{
            "executive_summary": "3-5 sentence high-level summary of the interview",
            "key_takeaways": [
                "Primary reason: Career growth limitations",
                "Valued: Team and culture",
                "Concerned: Management communication",
                "Red flag: Manager behavior needs investigation"
            ],
            "recommended_immediate_actions": [
                "Investigate management concerns",
                "Review career development process",
                "Conduct team pulse check"
            ],
            "strategic_implications": "May indicate broader retention risk in department",
            "overall_outcome": "Regrettable departure - high performer leaving due to addressable issues"
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
            result = safe_json_loads(response.text)
            logger.info("Completed: Summary generation")
            return result
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {"executive_summary": "", "key_takeaways": [], "error": str(e)}

    async def analyze_speakers(self, transcript_data: Dict[str, Any], primary_language: str) -> Dict[str, Any]:
        """
        Analyze speaker information
        """
        segments = transcript_data.get("segments", [])
        
        prompt = f"""
        Analyze speakers in this exit interview.

        Segments: {json.dumps(segments)}

        Return JSON:
        {{
            "speaker_count": 2,
            "speakers": [
                {{
                    "speaker_name": "HR Representative",
                    "role": "interviewer",
                    "speaking_time_percentage": 40,
                    "turn_count": 15,
                    "characteristics": "professional, empathetic, structured"
                }},
                {{
                    "speaker_name": "Exiting Employee",
                    "role": "departing_employee",
                    "speaking_time_percentage": 60,
                    "turn_count": 15,
                    "characteristics": "candid, thoughtful, professional"
                }}
            ],
            "interaction_quality": "good rapport, open dialogue"
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
            result = safe_json_loads(response.text)
            logger.info("Completed: Speaker analysis")
            return result
        except Exception as e:
            logger.error(f"Error analyzing speakers: {str(e)}")
            return {"speaker_count": 0, "speakers": [], "error": str(e)}

    async def analyze_audio(self, s3_url: str) -> Dict[str, Any]:
        """
        Main method to perform comprehensive exit interview analysis
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
            
            # Step 6: Validate exit interview
            logger.info("Step 6: Validating exit interview content...")
            validation_result = self.validate_exit_interview(transcript_data)
            
            # Step 7: Conditional analysis
            results = {"transcript": transcript_data}
            
            if validation_result.get("is_valid_exit_interview", False):
                logger.info("✓ Exit interview validation passed - proceeding with full analysis...")
                
                # Step 8: Parallel analysis
                logger.info("Step 8: Running parallel exit interview analysis...")
                
                async def run_exit_analysis():
                    tasks = [
                        # Audio-based (2 calls)
                        self.analyze_exit_interview_quality(gemini_file, primary_language),
                        self.analyze_exit_reasons(gemini_file, transcript_data, primary_language),
                        
                        # Transcript-based (7 calls)
                        self.identify_key_strengths(gemini_file, transcript_data, primary_language),
                        self.identify_concerns(gemini_file, transcript_data, primary_language),
                        self.identify_red_flags(gemini_file, transcript_data, primary_language),
                        self.generate_exit_assessment(gemini_file, transcript_data, primary_language),
                        self.extract_qa_pairs(transcript_data, primary_language),
                        self.generate_summary(transcript_data, primary_language),
                        self.analyze_speakers(transcript_data, primary_language),
                    ]
                    
                    analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    return {
                        "quality": analysis_results[0] if not isinstance(analysis_results[0], Exception) else {"error": str(analysis_results[0])},
                        "exit_reasons": analysis_results[1] if not isinstance(analysis_results[1], Exception) else {"error": str(analysis_results[1])},
                        "strengths": analysis_results[2] if not isinstance(analysis_results[2], Exception) else {"error": str(analysis_results[2])},
                        "concerns": analysis_results[3] if not isinstance(analysis_results[3], Exception) else {"error": str(analysis_results[3])},
                        "red_flags": analysis_results[4] if not isinstance(analysis_results[4], Exception) else {"error": str(analysis_results[4])},
                        "assessment": analysis_results[5] if not isinstance(analysis_results[5], Exception) else {"error": str(analysis_results[5])},
                        "qa_pairs": analysis_results[6] if not isinstance(analysis_results[6], Exception) else {"error": str(analysis_results[6])},
                        "summary": analysis_results[7] if not isinstance(analysis_results[7], Exception) else {"error": str(analysis_results[7])},
                        "speakers": analysis_results[8] if not isinstance(analysis_results[8], Exception) else {"error": str(analysis_results[8])}
                    }
                
                parallel_results = await run_exit_analysis()
                
                quality = parallel_results.get("quality", {})
                
                results.update({
                    "summary": parallel_results.get("summary", {}),
                    "sentiment": quality.get("sentiment_analysis", {}),
                    "quality_scores": quality.get("quality_scores", {}),
                    "exit_reasons": parallel_results.get("exit_reasons", {}),
                    "key_strengths": parallel_results.get("strengths", {}),
                    "concerns": parallel_results.get("concerns", {}),
                    "red_flags": parallel_results.get("red_flags", {}),
                    "assessment": parallel_results.get("assessment", {}),
                    "qa_pairs": parallel_results.get("qa_pairs", {}),
                    "speakers": parallel_results.get("speakers", {})
                })
                print("results", results)
                logger.info("✓ Full exit interview analysis completed")
                
            else:
                logger.warning(f"✗ Validation failed: {validation_result.get('content_type')} - {validation_result.get('reason')}")
                logger.info("→ Returning transcript only...")
                
                results.update({
                    "summary": {"executive_summary": "Analysis skipped - not a valid exit interview", "key_takeaways": []},
                    "sentiment": {"overall_sentiment": "neutral", "sentiment_score": 0, "note": "Analysis skipped"},
                    "quality_scores": {"overall_score": 0, "overall_confidence": 0, "note": "Analysis skipped"},
                    "exit_reasons": {"primary_reason": {}, "contributing_reasons": [], "note": "Analysis skipped"},
                    "key_strengths": {"key_strengths": [], "note": "Analysis skipped"},
                    "concerns": {"concerns": [], "note": "Analysis skipped"},
                    "red_flags": {"red_flags": [], "critical_red_flags_count": 0, "note": "Analysis skipped"},
                    "assessment": {"overall_assessment_score": 0, "note": "Analysis skipped"},
                    "qa_pairs": {"qa_pairs": [], "total_questions": 0, "note": "Analysis skipped"},
                    "speakers": {"speaker_count": 0, "speakers": [], "note": "Analysis skipped"}
                })
            
            # Final response
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            final_results = {
                "validation": validation_result,
                "metadata": {
                    "processing_time_seconds": processing_time,
                    "timestamp": start_time.isoformat(),
                    "primary_language": primary_language,
                    "file_url": s3_url,
                    "status": "completed_full_analysis" if validation_result.get("is_valid_exit_interview", False) else "completed_transcript_only",
                    "analysis_version": "1.0_EXIT_INTERVIEW",
                    "analysis_type": "exit_interview"
                },
                "language_info": {
                    "primary_language": primary_language,
                    "multilingual_content": "detected" if primary_language in ["mixed", "multilingual"] else "single_language"
                },
                "analysis_results": results
            }
            
            logger.info(f"✓ Exit interview analysis completed in {processing_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            logger.error(f"✗ Error in exit interview analysis: {str(e)}")
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
        Analyze audio directly from file/form data
        """
        start_time = datetime.now()
        file_path = None
        
        try:
            logger.info("Step 1: Saving uploaded audio file...")
            
            if not filename:
                filename = f"exit_audio_{int(time.time())}.mp3"
            
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
            
            # Upload to Gemini
            logger.info("Step 2: Uploading to Gemini...")
            gemini_file = self.upload_file_to_gemini(file_path)
            
            # Cleanup
            logger.info("Step 3: Cleaning up...")
            self.cleanup_temp_file(file_path)
            file_path = None
            
            # Detect language
            logger.info("Step 4: Detecting language...")
            primary_language = self.detect_primary_language(gemini_file)
            
            # Generate transcript
            logger.info("Step 5: Generating transcript...")
            transcript_data = self.generate_transcript(gemini_file, primary_language)
            
            # Validate
            logger.info("Step 6: Validating exit interview...")
            validation_result = self.validate_exit_interview(transcript_data)
            
            results = {"transcript": transcript_data}
            
            if validation_result.get("is_valid_exit_interview", False):
                logger.info("✓ Validation passed - full analysis...")
                
                async def run_exit_analysis():
                    tasks = [
                        self.analyze_exit_interview_quality(gemini_file, primary_language),
                        self.analyze_exit_reasons(gemini_file, transcript_data, primary_language),
                        self.identify_key_strengths(gemini_file, transcript_data, primary_language),
                        self.identify_concerns(gemini_file, transcript_data, primary_language),
                        self.identify_red_flags(gemini_file, transcript_data, primary_language),
                        self.generate_exit_assessment(gemini_file, transcript_data, primary_language),
                        self.extract_qa_pairs(transcript_data, primary_language),
                        self.generate_summary(transcript_data, primary_language),
                        self.analyze_speakers(transcript_data, primary_language),
                    ]
                    
                    analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    return {
                        "quality": analysis_results[0] if not isinstance(analysis_results[0], Exception) else {"error": str(analysis_results[0])},
                        "exit_reasons": analysis_results[1] if not isinstance(analysis_results[1], Exception) else {"error": str(analysis_results[1])},
                        "strengths": analysis_results[2] if not isinstance(analysis_results[2], Exception) else {"error": str(analysis_results[2])},
                        "concerns": analysis_results[3] if not isinstance(analysis_results[3], Exception) else {"error": str(analysis_results[3])},
                        "red_flags": analysis_results[4] if not isinstance(analysis_results[4], Exception) else {"error": str(analysis_results[4])},
                        "assessment": analysis_results[5] if not isinstance(analysis_results[5], Exception) else {"error": str(analysis_results[5])},
                        "qa_pairs": analysis_results[6] if not isinstance(analysis_results[6], Exception) else {"error": str(analysis_results[6])},
                        "summary": analysis_results[7] if not isinstance(analysis_results[7], Exception) else {"error": str(analysis_results[7])},
                        "speakers": analysis_results[8] if not isinstance(analysis_results[8], Exception) else {"error": str(analysis_results[8])}
                    }
                
                parallel_results = await run_exit_analysis()
                quality = parallel_results.get("quality", {})
                
                results.update({
                    "summary": parallel_results.get("summary", {}),
                    "sentiment": quality.get("sentiment_analysis", {}),
                    "quality_scores": quality.get("quality_scores", {}),
                    "exit_reasons": parallel_results.get("exit_reasons", {}),
                    "key_strengths": parallel_results.get("strengths", {}),
                    "concerns": parallel_results.get("concerns", {}),
                    "red_flags": parallel_results.get("red_flags", {}),
                    "assessment": parallel_results.get("assessment", {}),
                    "qa_pairs": parallel_results.get("qa_pairs", {}),
                    "speakers": parallel_results.get("speakers", {})
                })
                
                logger.info("✓ Full analysis completed")
                
            else:
                logger.warning(f"✗ Validation failed: {validation_result.get('content_type')}")
                results.update({
                    "summary": {"executive_summary": "Analysis skipped", "key_takeaways": []},
                    "sentiment": {"overall_sentiment": "neutral", "sentiment_score": 0},
                    "quality_scores": {"overall_score": 0},
                    "exit_reasons": {"primary_reason": {}, "contributing_reasons": []},
                    "key_strengths": {"key_strengths": []},
                    "concerns": {"concerns": []},
                    "red_flags": {"red_flags": [], "critical_red_flags_count": 0},
                    "assessment": {"overall_assessment_score": 0},
                    "qa_pairs": {"qa_pairs": [], "total_questions": 0},
                    "speakers": {"speaker_count": 0, "speakers": []}
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
                    "status": "completed_full_analysis" if validation_result.get("is_valid_exit_interview", False) else "completed_transcript_only",
                    "analysis_version": "1.0_EXIT_INTERVIEW",
                    "analysis_type": "exit_interview"
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


async def exit_flow(url):
    """Main exit interview analysis flow"""
    try:
        analyzer = ExitInterviewAnalyzer()
        result = await analyzer.analyze_audio(url)
        
        analysis_status = result.get("metadata", {}).get("status")
        
        if analysis_status == "completed_full_analysis":
            print("✓ Full analysis completed - generating comprehensive exit report...")
            report_response = generate_and_upload_exit_report(result)
            print("✓ Exit report generated")
        elif analysis_status == "completed_transcript_only":
            print("✓ Transcript-only - generating transcript report...")
            refactored_result = filter_json(result)
            report_response = generate_and_upload_exit_report(refactored_result)
            print("✓ Transcript report generated")
        else:
            print("✗ Analysis failed")
            report_response = None
        
        if report_response:
            result["report_url"] = report_response
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


async def exit_flow_from_file(audio_file, filename: str = None):
    """Exit interview analysis from uploaded file"""
    try:
        analyzer = ExitInterviewAnalyzer()
        result = await analyzer.analyze_audio_from_file(audio_file, filename)
        
        analysis_status = result.get("metadata", {}).get("status")
        
        if analysis_status == "completed_full_analysis":
            print("✓ Full analysis - generating exit report...")
            report_response = generate_and_upload_exit_report(result)
            print("✓ Report generated")
        elif analysis_status == "completed_transcript_only":
            print("✓ Transcript-only - generating report...")
            refactored_result = filter_json(result)
            report_response = generate_and_upload_exit_report(refactored_result)
            print("✓ Transcript report generated")
        else:
            print("✗ Analysis failed")
            report_response = None
        
        if report_response:
            result["report_url"] = report_response
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}


async def main():
    """Example usage"""
    test_urls = [
        "https://example-bucket.s3.amazonaws.com/exit-interviews/exit_001.mp3",
    ]
    
    for url in test_urls:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Analyzing Exit Interview: {url}")
            logger.info(f"{'='*50}")
            
            results = await exit_flow(url)
                
        except Exception as e:
            logger.error(f"Failed: {str(e)}")


if __name__ == "__main__":
    # Replace with actual exit interview URL
    s3_audio_url = "https://drive.google.com/file/d/1knSDnkcna7IHzNdrmrrYVF2GQZ9AhRaI/view?usp=sharing"
    asyncio.run(exit_flow(s3_audio_url))