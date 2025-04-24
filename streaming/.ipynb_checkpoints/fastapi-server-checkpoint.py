import os
import json
import numpy as np
from typing import List, Optional, Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse # Added HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import onnxruntime
import asyncio
import io
import wave
import traceback # For logging

# --- Piper Specific Imports ---
# Assuming these are in the environment or PYTHONPATH
try:
    from piper_phonemize import phonemize_codepoints, phonemize_espeak, tashkeel_run
    from piper_train.vits.utils import audio_float_to_int16
except ImportError:
    print("Warning: Failed to import Piper helper functions. Ensure piper-phonemize and piper-train are installed or accessible.")
    # Define dummy functions if needed for basic server structure testing
    def phonemize_espeak(text, voice): return [[text]]
    def phonemize_codepoints(text): return [[text]]
    def audio_float_to_int16(audio): return np.array(audio * 32767, dtype=np.int16)
    def tashkeel_run(text): return text # Placeholder

# --- Text Processing Imports ---
try:
    from underthesea import text_normalize # Vietnamese specific?
except ImportError:
    print("Warning: 'underthesea' not found. Using basic normalization.")
    def text_normalize(text): return text # Placeholder
try:
    from vinorm import TTSnorm # Vietnamese specific?
except ImportError:
    print("Warning: 'vinorm' not found. Using basic normalization.")
    def TTSnorm(text, **kwargs): return text # Placeholder

from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
# !! IMPORTANT: Update these paths !!
MODEL_PATH = "./steve_combined_multi_extra_char/to_train/outputs/weights/checkpoints/model_last.onnx"
CONFIG_PATH = "./steve_combined_multi_extra_char/to_train/config.json"
USE_GPU = False # Set to True if ONNXRuntime-GPU is installed and CUDA is available

# Speaker mapping (Ensure IDs match your model's config.json `speaker_id_map`)
# Example: Get from config['speaker_id_map'] or define manually if known
# SPEAKER_IDS = {"vi_female": 0, "vi_male": 1} # Example, load dynamically below

# Text Splitting
TEXT_SPLITTER_CHUNK_SIZE = 120 # Experiment with this
TEXT_SPLITTER_CHUNK_OVERLAP = 10 # Experiment with this

# Phonemization constants
PAD = "_"
BOS = "^"
EOS = "$"

# --- Pydantic Models ---
class TTSRequest(BaseModel):
    text: str
    speaker: str # Speaker name (key from SPEAKER_IDS)
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8

# --- FastAPI App Setup ---
app = FastAPI(title="Piper TTS Streaming API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables ---
onnx_model: Optional[onnxruntime.InferenceSession] = None
model_config: Optional[Dict[str, Any]] = None
speaker_id_map: Optional[Dict[str, int]] = None # Populated at startup

# --- Helper Functions ---

class PhonemeType:
    ESPEAK = "espeak"
    TEXT = "text"

def load_piper_model(model_path_str: str, config_path_str: str, use_gpu: bool):
    """Loads the Piper ONNX model and its configuration."""
    model_path = Path(model_path_str)
    config_path = Path(config_path_str)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading model from: {model_path}")
    print(f"Loading config from: {config_path}")

    sess_options = onnxruntime.SessionOptions()
    providers = []
    if use_gpu:
        providers.append(("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}))
    providers.append("CPUExecutionProvider") # Fallback or default

    print(f"Using ONNX Runtime providers: {providers}")
    try:
        model = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        print("Ensure ONNX Runtime is installed (onnxruntime or onnxruntime-gpu).")
        raise

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Extract speaker map if it exists
    s_id_map = config.get("speaker_id_map")
    if not s_id_map:
         # Handle single-speaker models
         if config.get("num_speakers", 1) == 1:
             # Assign a default name or use None? Let's use a default.
             s_id_map = {"default": 0}
             print("Model is single-speaker. Using 'default' speaker name.")
         else:
             print("Warning: Multi-speaker model detected, but 'speaker_id_map' missing in config.json!")
             # Attempt to create a dummy map based on num_speakers if needed, but this is risky.
             s_id_map = {f"speaker_{i}": i for i in range(config.get("num_speakers", 0))}


    print("Model and config loaded successfully.")
    return model, config, s_id_map

def normalize_and_split_text(text: str) -> List[str]:
    """Applies normalization and splitting to the input text."""
    print(f"Original text (first 100): {text[:100]}...")
    # 1. Apply general Vietnamese normalization (if applicable)
    try:
        # Assuming TTSnorm handles basic punctuation and normalization
        # Adjust punc=False if you want punctuation kept for splitting guides
        normalized = TTSnorm(text, punc=True).strip()
    except Exception as e:
        print(f"Warning: TTSnorm failed ({e}). Skipping.")
        normalized = text.strip()

    # 2. Apply specific text normalization (if applicable, e.g., underthesea)
    try:
        normalized = text_normalize(normalized)
    except Exception as e:
         print(f"Warning: underthesea.text_normalize failed ({e}). Skipping.")
         # Keep the result from TTSnorm

    print(f"Normalized text (first 100): {normalized[:100]}...")

    # 3. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP,
        length_function=len,
        # Keep common sentence-ending punctuation to potentially guide splitting
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        keep_separator=True # Keep separators to avoid merging words across chunks
    )
    chunks = text_splitter.split_text(normalized)
    # Clean up empty chunks or just whitespace chunks resulting from split
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk and not chunk.isspace()]
    print(f"Split into {len(cleaned_chunks)} non-empty chunks.")
    return cleaned_chunks

def phonemize_chunk(text_chunk: str, config: Dict[str, Any]) -> List[List[str]]:
    """Converts a text chunk to phonemes using the method specified in config."""
    phoneme_type = config.get("phoneme_type", PhonemeType.ESPEAK) # Default to espeak if missing

    try:
        if phoneme_type == PhonemeType.ESPEAK:
            espeak_voice = config.get("espeak", {}).get("voice")
            if not espeak_voice:
                raise ValueError("Missing 'espeak.voice' in model config for phonemization")
            if espeak_voice == "ar":
                # Optional: Arabic diacritization - requires piper-phonemize[ar]
                 try:
                    text_chunk = tashkeel_run(text_chunk)
                 except NameError:
                     print("Warning: tashkeel_run not available (optional Arabic dependency missing?)")
                 except Exception as e:
                     print(f"Warning: tashkeel_run failed: {e}")
            return phonemize_espeak(text_chunk, espeak_voice)

        elif phoneme_type == PhonemeType.TEXT:
            return phonemize_codepoints(text_chunk)
        else:
            raise ValueError(f"Unsupported phoneme type in config: {phoneme_type}")
    except Exception as e:
         print(f"Error during phonemization for chunk '{text_chunk[:50]}...': {e}")
         traceback.print_exc()
         return [] # Return empty list on error

def phonemes_to_ids(phonemes: List[str], config: Dict[str, Any]) -> List[int]:
    """Converts a list of phonemes to a list of IDs."""
    if "phoneme_id_map" not in config:
        raise ValueError("Missing 'phoneme_id_map' in model config.")
    id_map = config["phoneme_id_map"]
    ids: List[int] = []

    # Check if BOS/EOS/PAD characters are in the map
    bos_id = id_map.get(BOS)
    eos_id = id_map.get(EOS)
    pad_id = id_map.get(PAD)

    if bos_id is None or eos_id is None or pad_id is None:
         raise ValueError(f"Missing required characters ({BOS}, {EOS}, {PAD}) in phoneme_id_map")

    ids.extend(bos_id)
    for phoneme in phonemes:
        phoneme_ids = id_map.get(phoneme)
        if phoneme_ids is None:
            print(f"Warning: Phoneme '{phoneme}' not found in id map. Skipping.")
            continue
        ids.extend(phoneme_ids)
        ids.extend(pad_id) # Pad between phonemes

    ids.extend(eos_id)
    return ids

def synthesize_audio_chunk(
    text_ids: List[int],
    speaker_id: Optional[int],
    model: onnxruntime.InferenceSession,
    config: Dict[str, Any],
    length_scale: float,
    noise_scale: float,
    noise_scale_w: float
) -> np.ndarray:
    """Runs ONNX inference for a single chunk of phoneme IDs."""
    text_np = np.expand_dims(np.array(text_ids, dtype=np.int64), 0)
    text_lengths_np = np.array([text_np.shape[1]], dtype=np.int64)
    scales_np = np.array(
        [noise_scale, length_scale, noise_scale_w],
        dtype=np.float32,
    )
    speaker_id_np = None
    if speaker_id is not None:
        # Ensure speaker ID is valid if model expects it
        if config.get("num_speakers", 1) > 1:
            speaker_id_np = np.array([speaker_id], dtype=np.int64)
        else:
             # Ignore speaker_id for single-speaker models
             pass # speaker_id_np remains None

    # Prepare inputs for ONNX model, ensuring keys match model's expected input names
    # Common names are 'input', 'input_lengths', 'scales', 'sid'
    # Verify these against your specific model using Netron or model inspection
    onnx_input = {
        "input": text_np,
        "input_lengths": text_lengths_np,
        "scales": scales_np,
    }
    if speaker_id_np is not None:
         onnx_input["sid"] = speaker_id_np
    elif "sid" in [inp.name for inp in model.get_inputs()]:
         # Handle case where model has 'sid' input but it's single speaker or ID wasn't provided
         # This might require setting a default ID (e.g., 0) if the model strictly requires it
         # print("Warning: Model expects 'sid' input, but none provided/needed. Check model structure.")
         # onnx_input["sid"] = np.array([0], dtype=np.int64) # Example: provide default if required
         pass # Usually optional inputs don't need to be provided if None

    try:
        # Run inference
        audio_output = model.run(None, onnx_input)[0]
        # Output shape might vary, common is (1, 1, num_samples) or (1, num_samples)
        # Squeeze to get 1D array
        audio_float = np.squeeze(audio_output)
        # Convert to int16
        audio_int16 = audio_float_to_int16(audio_float)
        return audio_int16
    except Exception as e:
        print(f"Error during ONNX inference: {e}")
        traceback.print_exc()
        # Return empty array on failure
        return np.array([], dtype=np.int16)


def create_wave_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Creates a WAV header for streaming (unknown file size)."""
    # Using wave module to ensure correctness
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(bits_per_sample // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(b'') # Write 0 frames to finalize header for unknown size
    header_bytes = buffer.getvalue()
    return header_bytes

# --- Streaming Generator ---
async def stream_audio_generator(request: TTSRequest) -> AsyncGenerator[bytes, None]:
    """
    Generates and streams audio chunks for the given request.
    """
    global onnx_model, model_config, speaker_id_map
    start_time = time.time()
    print(f"[Generator] Request received: speaker='{request.speaker}', text='{request.text[:50]}...'")

    # 1. Validate Model and Config
    if onnx_model is None or model_config is None or speaker_id_map is None:
        print("Error: Model or config not loaded.")
        raise HTTPException(status_code=503, detail="TTS Service not ready.")

    # 2. Validate Speaker
    speaker_id = speaker_id_map.get(request.speaker)
    # Handle single-speaker model case where 'default' might be the key
    if speaker_id is None and "default" in speaker_id_map and config.get("num_speakers", 1) == 1:
         speaker_id = speaker_id_map["default"]
         print(f"Info: Using default speaker ID {speaker_id} for single-speaker model.")
    elif speaker_id is None:
        valid_speakers = list(speaker_id_map.keys())
        print(f"Error: Invalid speaker '{request.speaker}'. Valid options: {valid_speakers}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid speaker specified. Available speakers: {', '.join(valid_speakers)}"
        )
    print(f"[Generator] Using speaker: '{request.speaker}' (ID: {speaker_id})")

    # 3. Normalize and Split Text
    try:
        text_chunks = normalize_and_split_text(request.text)
        if not text_chunks:
            print("Warning: Text normalization/splitting resulted in no processable chunks.")
            # Yield only header for empty result?
            yield create_wave_header(model_config["audio"]["sample_rate"])
            return
    except Exception as e:
        print(f"Error during text processing: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error processing input text.")

    # 4. Yield WAV Header
    try:
        sample_rate = model_config["audio"]["sample_rate"]
        yield create_wave_header(sample_rate)
        print(f"[Generator] Yielded WAV header (Sample Rate: {sample_rate})")
    except Exception as e:
        print(f"Error creating WAV header: {e}")
        raise HTTPException(status_code=500, detail="Internal error creating audio header.")

    # 5. Process Chunks and Stream Audio
    total_bytes_yielded = 0
    num_chunks = len(text_chunks)
    for i, chunk in enumerate(text_chunks):
        chunk_num = i + 1
        print(f"[Generator] Processing chunk {chunk_num}/{num_chunks}: '{chunk[:50]}...'")
        try:
            # Phonemize the chunk
            phoneme_sentences = phonemize_chunk(chunk, model_config)
            if not phoneme_sentences:
                 print(f"  [Chunk {chunk_num}] Skipped - Phonemization returned empty.")
                 continue # Skip to next chunk if phonemization fails/is empty

            audio_data_list = []
            for phonemes in phoneme_sentences:
                 if not phonemes: continue # Skip empty phoneme lists within the chunk

                 # Convert phonemes to IDs
                 phoneme_ids = phonemes_to_ids(phonemes, model_config)

                 # Synthesize audio for this phoneme sequence
                 audio_segment = synthesize_audio_chunk(
                    phoneme_ids,
                    speaker_id,
                    onnx_model,
                    model_config,
                    request.length_scale,
                    request.noise_scale,
                    request.noise_scale_w
                 )
                 if audio_segment.size > 0:
                     audio_data_list.append(audio_segment)

            # Concatenate audio segments if multiple sentences were in the chunk
            if audio_data_list:
                final_audio_chunk = np.concatenate(audio_data_list)
                audio_bytes = final_audio_chunk.tobytes()
                yield audio_bytes
                bytes_yielded = len(audio_bytes)
                total_bytes_yielded += bytes_yielded
                print(f"  [Chunk {chunk_num}] Yielded {bytes_yielded} audio bytes.")
            else:
                 print(f"  [Chunk {chunk_num}] Skipped - Synthesis resulted in empty audio.")

            # Small delay - potentially helps client processing, tune or remove if needed
            await asyncio.sleep(0.01)

        except Exception as e:
             # Log error for the specific chunk but try to continue with others
             print(f"Error processing chunk {chunk_num}: {e}")
             traceback.print_exc()
             # Optionally yield silence or just skip? Skipping seems safer.

    # 6. Finalization
    final_delay = 0.05 # Add small delay for potential buffer flushing
    print(f"[Generator] Finished processing all chunks. Adding final delay: {final_delay}s")
    await asyncio.sleep(final_delay)
    end_time = time.time()
    print(f"[Generator] Stream generation complete. Total bytes yielded: {total_bytes_yielded}. Time: {end_time - start_time:.2f}s")


# --- FastAPI Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Load the ONNX model and config when the server starts."""
    global onnx_model, model_config, speaker_id_map
    print("Server startup: Loading Piper model...")
    try:
        onnx_model, model_config, speaker_id_map = load_piper_model(MODEL_PATH, CONFIG_PATH, USE_GPU)
        # Validate essential config keys
        if "audio" not in model_config or "sample_rate" not in model_config["audio"]:
             raise ValueError("Config missing required 'audio.sample_rate'")
        if "phoneme_type" not in model_config:
             raise ValueError("Config missing required 'phoneme_type'")
        print("Startup complete. Model is ready.")
    except FileNotFoundError as e:
        print(f"FATAL STARTUP ERROR: {e}")
        print("Please ensure MODEL_PATH and CONFIG_PATH are correct.")
        # Server will likely fail to handle requests, but FastAPI might still start.
        onnx_model = None
        model_config = None
        speaker_id_map = None
    except Exception as e:
        print(f"FATAL STARTUP ERROR during model load: {e}")
        traceback.print_exc()
        onnx_model = None
        model_config = None
        speaker_id_map = None

@app.post("/tts/stream")
async def tts_stream_endpoint(request: TTSRequest):
    """Endpoint to stream TTS audio for the given text and parameters."""
    print(f"Received POST /tts/stream request")
    try:
        return StreamingResponse(
            stream_audio_generator(request),
            media_type="audio/wav"
        )
    except HTTPException as he:
        # Re-raise known HTTP exceptions
        raise he
    except Exception as e:
        print(f"Error during stream processing: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.get("/speakers", response_model=Dict[str, int])
async def get_speakers_endpoint():
    """Returns the available speaker names and their corresponding IDs."""
    if speaker_id_map is None:
         raise HTTPException(status_code=503, detail="Speakers not available (Model not loaded).")
    # Return the map directly { "speaker_name": id }
    return speaker_id_map

@app.get("/health")
async def health_check_endpoint():
    """Provides a health check status for the service."""
    if onnx_model and model_config and speaker_id_map:
        return {"status": "ok", "message": "Piper TTS model loaded and ready."}
    else:
        return {"status": "error", "message": "Piper TTS model not loaded."}

@app.get("/", response_class=HTMLResponse)
async def serve_html_client():
    """Serves the js-client.html file."""
    html_file_path = Path("./js-client.html") # Assume HTML is in the same directory
    if not html_file_path.is_file():
        raise HTTPException(status_code=404, detail="HTML client file not found.")
    with open(html_file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Piper TTS FastAPI server with Uvicorn...")
    # Use reload=True only for development
    uvicorn.run(app, host="0.0.0.0", port=8000) # Default Piper port