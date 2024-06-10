from dotenv import load_dotenv
from deepgram import Deepgram
import os
load_dotenv()
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')

def transcribe_audio(filename):
    dg_client = Deepgram(DEEPGRAM_API_KEY)
    with open(filename, 'rb') as audio:
        source = {'buffer': audio, 'mimetype': 'audio/mp3'}
        response = dg_client.transcription.sync_prerecorded(source,model='nova-2-ea',smart_format=True)
        return response['results']['channels'][0]['alternatives'][0]['transcript']