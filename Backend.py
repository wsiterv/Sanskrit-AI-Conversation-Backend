from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from uuid import uuid4
from gtts import gTTS
import whisper
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
model = whisper.load_model("small")  

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/api/whisper/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio = request.files['audio']
    filename = f"{uuid4()}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio.save(filepath)
    
    try:
        result = model.transcribe(filepath, language='Sanskrit', task='translate')
        print("Whisper result:", result)
        english_text = result['text'].strip()
        print("English transcription:", english_text)

        improve_prompt = f"""
Fix and improve this SANSKRIT sentence to make it grammatically correct and clear:
"{english_text}"

Respond with only the corrected Sanskrit sentence, nothing else.
"""
        
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        improve_response = gemini_model.generate_content(improve_prompt)
        corrected_english = improve_response.text.strip()
        print("Corrected English:", corrected_english)

        sanskrit_prompt = f"""
A person said in Sanskrit: "{corrected_english}"

Respond appropriately in Sanskrit using Devanagari script. Give only the Sanskrit response, nothing else. No explanations, no English text, just the Sanskrit response.
Tip:If someone asks your name type मिथुन .
"""
        
        sanskrit_response = gemini_model.generate_content(sanskrit_prompt)
        sanskrit_text = sanskrit_response.text.strip()
        print("Sanskrit response:", sanskrit_text)

        return jsonify({
            'transcribed': corrected_english,
            'response': sanskrit_text
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/tts/generate', methods=['POST'])
def tts():
    data = request.json
    text = data.get('text', '')
    filename = f"{uuid4()}.mp3"
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    try:
        tts = gTTS(text=text, lang='hi', slow=False)  
        tts.save(filepath)
        return jsonify({'audio_url': f'/outputs/{filename}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/outputs/<path:filename>')
def serve_audio(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)