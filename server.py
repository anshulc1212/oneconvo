from flask import Flask, request, jsonify
import asyncio
from flask_cors import CORS
import os
import file_upload
import translate_test
from translate2 import demo_flow
from recruitment import hr_flow,demo_flow_from_file
import threading
from helper import process_and_notify
from patient import patient_flow

from snapfind import SnapfindInterviewAnalyzer

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    """API endpoint to handle file uploads."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400      
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        file_id = file_upload.upload_file_to_genai(file)      
        return jsonify({
            'message': 'File uploaded successfully',
            'file_id': file_id,
            'filename': file.filename
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

    
@app.route("/one_convo", methods=["POST"])
def one_convo():
    """
    Combined endpoint: 
    1. Uploads a file to GenAI
    2. Runs convlyze analysis using the uploaded file ID
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        file_id = file_upload.upload_file_to_genai(file)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(translate_test.demo_flow2(file_id))
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# @app.route("/hr_audio_url", methods=["POST"])
# def analyze_audio_url():
#     """
#     HR Recruitment Interview Analysis from S3/URL
#     Expects JSON: {"url": "https://..."}
#     """
#     data = request.json
#     audio_url = data.get("url")
    
#     if not audio_url:
#         return jsonify({"error": "Missing 'url' in request"}), 400

#     try:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         result = loop.run_until_complete(hr_flow(audio_url))
#         return jsonify(result), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
@app.route("/hr_audio_file", methods=["POST"])
def analyze_audio_file():
    """
    HR Recruitment Interview Analysis from uploaded audio file
    Expects multipart/form-data with 'file' field
    
    Usage:
    curl -X POST http://localhost:3000/analyze_audio_file \
         -F "file=@interview.mp3"
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension (optional but recommended)
        allowed_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Unsupported file format. Allowed formats: {", ".join(allowed_extensions)}'
            }), 400
        
        # Run analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(demo_flow_from_file(file, file.filename))
        
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route("/snapfind_interview", methods=["POST"])
def snapfind_interview():
    """
    Snapfind Interview Analysis from audio URL.

    Expects JSON body:
    {
        "url": "https://s3.amazonaws.com/interview.mp3",
        "questions": [
            { "question_id": 1, "question_text": "How many recruitment processes..." },
            { "question_id": 2, "question_text": "Describe your experience leading..." }
        ],
        "client": "onetab"
    }

    Returns JSON with transcript, question_answers, and communication_skills.

    Usage:
    curl -X POST http://localhost:3000/snapfind_interview \
         -H "Content-Type: application/json" \
         -d '{"url": "https://...", "questions": [...], "client": "onetab"}'
    """
    try:
        payload = request.get_json()

        # ── Validate payload ──────────────────────────────
        if not payload:
            return jsonify({"error": "Request body must be JSON"}), 400

        url = payload.get("url", "").strip()
        if not url:
            return jsonify({"error": "Missing required field: url"}), 400

        questions = payload.get("questions")
        if not questions or not isinstance(questions, list) or len(questions) == 0:
            return jsonify({"error": "Missing or empty required field: questions"}), 400

        # Validate each question has question_id and question_text
        for i, q in enumerate(questions):
            if "question_id" not in q or "question_text" not in q:
                return jsonify({
                    "error": f"Question at index {i} must have 'question_id' and 'question_text'"
                }), 400

        client = payload.get("client", "").strip()
        if not client:
            return jsonify({"error": "Missing required field: client"}), 400

        # ── Run analysis ──────────────────────────────────
        analyzer = SnapfindInterviewAnalyzer()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                analyzer.analyze(url=url, questions=questions, client=client)
            )
        finally:
            loop.close()

        # If analyzer returned an error (e.g. audio too short), return 400
        if "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# @app.route("/snapfind_interview", methods=["POST"])
# def analyze_interview():
#     """
#     Interview Audio Analysis from uploaded audio file.
#     Expects multipart/form-data with 'file' field.

#     Returns JSON with 'transcript' and 'question_answers' keys.

#     Usage:
#     curl -X POST http://localhost:3000/analyze_interview \
#          -F "file=@interview.mp3"
#     """
#     try:
#         # Validate file presence
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part in the request'}), 400

#         file = request.files['file']

#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400

#         # Validate file extension
#         allowed_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac'}
#         file_ext = os.path.splitext(file.filename)[1].lower()

#         if file_ext not in allowed_extensions:
#             return jsonify({
#                 'error': f'Unsupported file format. Allowed formats: {", ".join(allowed_extensions)}'
#             }), 400

#         # Run async analysis in a new event loop
#         analyzer = InterviewAnalyzer()
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         try:
#             result = loop.run_until_complete(
#                 analyzer.analyze(file, filename=file.filename)
#             )
#         finally:
#             loop.close()

#         return jsonify(result), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    
@app.route("/try_gemini", methods=["POST"])
def try_gemini():
    data = request.json
    audio_url = data.get("url")
    if not audio_url:
        return jsonify({"error": "Missing 'url' in request"}), 400

    try:
       
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(demo_flow(audio_url))
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

 
    
@app.route("/convlyze", methods=["POST"])
def convlyze():
    data = request.json
    file_id = data.get("fileId")
    if not file_id:
        return jsonify({"error": "Missing 'fileId' in request"}), 400

    try:
       
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(translate_test.demo_flow2(file_id))
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/hr_audio_url_async", methods=["POST"])
def analyze_audio_url_async():
    """
    Async HR Recruitment Interview Analysis from URL.
    Expects JSON: {"url": "https://...", "candidate_id": "...", "webhookUrl": "https://...", "client": "..."}
    Returns 202 immediately. Result POSTed to webhookUrl on completion.
    """
    data = request.json

    audio_url    = data.get("url")
    candidate_id = data.get("candidate_id")
    webhook_url  = data.get("webhookUrl")
    client       = data.get("client", "").strip()   # ← NEW

    if not audio_url:
        return jsonify({"error": "Missing 'url' in request"}), 400
    if not candidate_id:
        return jsonify({"error": "Missing 'candidate_id' in request"}), 400
    if not webhook_url:
        return jsonify({"error": "Missing 'webhookUrl' in request"}), 400
    if not client:                                   # ← NEW
        return jsonify({"error": "Missing 'client' in request"}), 400

    thread = threading.Thread(
        target=process_and_notify,
        args=(candidate_id, audio_url, webhook_url, client),   # ← pass client
        daemon=True
    )
    thread.start()

    return jsonify({
        "status": "accepted",
        "candidate_id": candidate_id,
        "message": "Processing started. Results will be POSTed to webhookUrl."
    }), 202


@app.route("/patient_audio_url", methods=["POST"])
def analyze_patient_audio():
    """
    Patient Clinical Audio Analysis from S3/URL.
    Transcribes and extracts structured clinical notes from:
    - Doctor dictation (monologue)
    - Doctor-patient consultation (two-person)
 
    Expects JSON: {"url": "https://..."}
 
    Returns structured clinical notes JSON with keys:
    patient_name, chief_complaint, history_of_present_illness,
    examination_findings, vital_signs, provisional_diagnosis,
    investigations_advised, laboratory_reports_reviewed,
    final_diagnosis, management_plan, follow_up_advice,
    summary, transcript
 
    Usage:
    curl -X POST http://localhost:3000/patient_audio_url \\
         -H "Content-Type: application/json" \\
         -d '{"url": "https://s3.amazonaws.com/patient/consultation.mp3"}'
    """
    data = request.json
 
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
 
    audio_url = data.get("url", "").strip()
    if not audio_url:
        return jsonify({"error": "Missing 'url' in request"}), 400
 
    patient_name = data.get("patient_name", "not mentioned").strip()
 
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(patient_flow(audio_url, patient_name))
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/client_usage", methods=["GET"])
def get_client_usage():
    """
    Get usage stats for a specific client or all clients.

    Query param (optional): ?client=onetab
    If no client passed, returns usage for ALL clients.

    Examples:
    curl http://localhost:3000/client_usage?client=onetab
    curl http://localhost:3000/client_usage
    """
    from snapfind import load_usage

    client = request.args.get("client", "").strip()
    usage = load_usage()

    if client:
        if client not in usage:
            return jsonify({"error": f"Client '{client}' not found"}), 404
        return jsonify({
            "client": client,
            "total_minutes_processed": usage[client]["total_minutes_processed"],
            "last_used": usage[client]["last_used"]
        }), 200
    else:
        # Return all clients
        return jsonify(usage), 200

@app.route("/hr_audio_url", methods=["POST"])
def analyze_audio_url():
    """
    HR Recruitment Interview Analysis from S3/URL
    Expects JSON: {"url": "https://...", "client": "..."}
    """
    data = request.json
    audio_url = data.get("url")
    client = data.get("client", "").strip()

    if not audio_url:
        return jsonify({"error": "Missing 'url' in request"}), 400
    if not client:
        return jsonify({"error": "Missing 'client' in request"}), 400

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(hr_flow(audio_url, client))
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, port=3000, host="0.0.0.0")