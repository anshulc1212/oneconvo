from google import genai
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

api_key = os.getenv("GENAI_API_KEY")
client = genai.Client(api_key=api_key)



def upload_file_to_genai(file):
    """Uploads a file to Google GenAI and returns the file ID.""" 
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        file.save(temp_file.name)
        temp_file_path = temp_file.name
    
    try:
        myfile = client.files.upload(file=temp_file_path)
        return myfile.name
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

