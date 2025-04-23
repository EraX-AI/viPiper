import os
import json
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import onnxruntime
import asyncio
import io
import wave

# Import necessary functions from your Piper_Infer
from piper_phonemize import phonemize_codepoints, phonemize_espeak, tashkeel_run
from piper_train.vits.utils import audio_float_to_int16
from underthesea import text_normalize
from vinorm import TTSnorm
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants from your existing code
PAD = "_"
BOS = "^"
EOS = "$"

# Define speaker IDs
SPEAKER_IDS = {
    "vi_male": 1,
    "vi_female": 0
}

class TTSRequest(BaseModel):
    text: str
    speaker: str = "vi_male"
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8

app = FastAPI(title="Piper TTS Streaming API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    length_function=len,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    keep_separator=False
)

# Global variables for model and config
model = None
config = None

class PhonemeType:
    ESPEAK = "espeak"
    TEXT = "text"

def load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)

def load_onnx(model_path, config_path, use_gpu=False):
    sess_options = onnxruntime.SessionOptions()
    providers = [
        "CPUExecutionProvider"
        if not use_gpu
        else ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})
    ]
    config = load_config(config_path)
    model = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers
    )
    return model, config

def phonemize(config, text: str) -> List[List[str]]:
    """Text to phonemes grouped by sentence."""
    if config["phoneme_type"] == PhonemeType.ESPEAK:
        if config["espeak"]["voice"] == "ar":
            # Arabic diacritization
            text = tashkeel_run(text)
        return phonemize_espeak(text, config["espeak"]["voice"])
    if config["phoneme_type"] == PhonemeType.TEXT:
        return phonemize_codepoints(text)
    raise ValueError(f'Unexpected phoneme type: {config["phoneme_type"]}')

def phonemes_to_ids(config, phonemes: List[str]) -> List[int]:
    """Phonemes to ids."""
    id_map = config["phoneme_id_map"]
    ids: List[int] = list(id_map[BOS])
    for phoneme in phonemes:
        if phoneme not in id_map:
            print(f"Missing phoneme from id map: {phoneme}")
            continue
        ids.extend(id_map[phoneme])
        ids.extend(id_map[PAD])
    ids.extend(id_map[EOS])
    return ids

def generate_audio_chunk(chunk_content, model, config, sid, length_scale, noise_scale, noise_scale_w):
    """Generate audio for a single text chunk"""
    chunk_content = chunk_content.strip()
    audios = []
    
    if config["phoneme_type"] == "PhonemeType.ESPEAK":
        config["phoneme_type"] = "espeak"
        
    phonem_content = phonemize(config, chunk_content)
    for phonemes in phonem_content:
        phoneme_ids = phonemes_to_ids(config, phonemes)
        num_speakers = config["num_speakers"]
        
        if num_speakers == 1:
            speaker_id = None
        else:
            speaker_id = sid
            
        text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        text_lengths = np.array([text.shape[1]], dtype=np.int64)
        scales = np.array(
            [noise_scale, length_scale, noise_scale_w],
            dtype=np.float32,
        )
        
        if speaker_id is not None:
            speaker_id = np.array([speaker_id], dtype=np.int64)
            
        audio = model.run(
            None,
            {
                "input": text,
                "input_lengths": text_lengths,
                "scales": scales,
                "sid": speaker_id,
            },
        )[0].squeeze((0, 1))
        
        audio = audio_float_to_int16(audio.squeeze())
        audios.append(audio)
    
    return np.concatenate(audios) if audios else np.array([], dtype=np.int16)

def create_wave_header(sample_rate, num_channels=1, bits_per_sample=16):
    """Create a wave header for streaming"""
    header = wave.Wave_write(io.BytesIO())
    header.setnchannels(num_channels)
    header.setsampwidth(bits_per_sample // 8)
    header.setframerate(sample_rate)
    header.close()
    
    # Get the header bytes
    header_bytes = header.fp.getvalue()
    return header_bytes

async def stream_audio_generator(request: TTSRequest):
    """Generate audio chunks and stream them"""
    global model, config
    
    if model is None or config is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Get speaker ID
    speaker_id = SPEAKER_IDS.get(request.speaker)
    if speaker_id is None:
        raise HTTPException(status_code=400, detail=f"Invalid speaker: {request.speaker}")
    
    # Normalize and split text
    content = request.text
    splitted_content = text_splitter.split_text(content)
    splitted_content = [TTSnorm(text_normalize(i), punc=False).strip() for i in splitted_content]
    
    sample_rate = config["audio"]["sample_rate"]
    
    # Yield wave header first
    yield create_wave_header(sample_rate)
    
    # Process each chunk and yield audio data
    for chunk in splitted_content:
        if not chunk.strip():
            continue
            
        # Generate audio for this chunk
        audio_chunk = generate_audio_chunk(
            chunk, model, config, speaker_id,
            request.length_scale, request.noise_scale, request.noise_scale_w
        )
        
        if len(audio_chunk) > 0:
            # Convert numpy array to bytes
            audio_bytes = audio_chunk.tobytes()
            yield audio_bytes
        
        # Small delay to allow client processing
        await asyncio.sleep(0.01)

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model, config
    
    # Update these paths to match your environment
    model_path = "./steve_combined_multi_extra_char/to_train/outputs/weights/checkpoints/model_last.onnx"
    config_path = "./steve_combined_multi_extra_char/to_train/config.json"
    
    try:
        model, config = load_onnx(model_path, config_path, use_gpu=False)
        print(f"Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    """Stream TTS audio for given text"""
    return StreamingResponse(
        stream_audio_generator(request),
        media_type="audio/wav"
    )

@app.get("/speakers")
async def get_speakers():
    """Get available speakers"""
    return SPEAKER_IDS

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or config is None:
        return {"status": "error", "message": "Model not initialized"}
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
