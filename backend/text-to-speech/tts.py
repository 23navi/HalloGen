import io
import os
import sys
from typing import Optional
import uuid

import modal
from chatterbox.tts import ChatterboxTTS
from pydantic import BaseModel
import torch
import torchaudio

app = modal.App("chatterbox-tts-generator")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .pip_install_from_requirements("text-to-speech/requirements.txt")
    .apt_install("ffmpeg")
)

volume = modal.Volume.from_name("hf-cache-chatterbox", create_if_missing=True)

s3_secret = modal.Secret.from_name("hallo-gen-backend-secret")


class TextToSpeechRequest(BaseModel):
    text: str
    voice_S3_key: Optional[str] = None


class TextToSpeechResponse(BaseModel):
    s3_key: str


@app.cls(
    image=image,
    gpu="L40S",
    volumes={
        "/root/.cache/huggingface": volume,
        "/s3-mount": modal.CloudBucketMount("hallo-gen-private-bucket", secret=s3_secret)
    },
    scaledown_window=120,
    secrets=[s3_secret]
)
class TextToSpeechServer:
    @modal.enter()
    def load_model(self):
        print("Loading chatterbox model...")
        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        print("Model loaded successfully")

    @modal.fastapi_endpoint(method="POST")
    def generate_speech(self, request: TextToSpeechRequest) -> TextToSpeechResponse:
        print(f"Received request to generate speech for: {request.text}")

        with torch.no_grad():
            if request.voice_S3_key:
                print("Using voice cloning with S3 speech prompt...")
                audio_prompt_path = f"/s3-mount/{request.voice_S3_key}"
                if not os.path.exists(audio_prompt_path):
                    raise FileNotFoundError(
                        f"Prompt audio not found at {audio_prompt_path}")
                wav = self.model.generate(
                    request.text, audio_prompt_path=audio_prompt_path)

            else:
                print("Using basic text-to-speech without voice cloning...")
                wav = self.model.generate(request.text)

            wav_cpu = wav.cpu()

        buffer = io.BytesIO()
        torchaudio.save(buffer, wav_cpu, self.model.sr, format="wav")
        buffer.seek(0)
        audio_bytes = buffer.read()

        audio_uuid = str(uuid.uuid4())
        s3_key = f"tts/{audio_uuid}.wav"
        s3_path = f"/s3-mount/{s3_key}"
        os.makedirs(os.path.dirname(s3_path), exist_ok=True)
        with open(s3_path, "wb") as f:
            f.write(audio_bytes)
        print(f"Saved audio to S3: {s3_key}")

        return TextToSpeechResponse(s3_key=s3_key)


@app.local_entrypoint()
def main():
    import requests

    server = TextToSpeechServer()
    endpoint_url = server.generate_speech.get_web_url()

    request = TextToSpeechRequest(
        text="Hello this is Navi from HalloGen. This is a test of the voice cloning capabilities of the Chatterbox text to speech model.",
        voice_S3_key="samples/voices/1.wav"
    )

    payload = request.model_dump()

    # headers = {
    #     "Modal-Key": "wHU5",
    #     "Modal-Secret": "wLikGya"
    # }

    # response = requests.post(endpoint_url, json=payload, headers=headers)
    response = requests.post(endpoint_url, json=payload)
    response.raise_for_status()

    result = TextToSpeechResponse(**response.json())

    print(result.s3_key)
