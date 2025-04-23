## Simple, work out of the box Piper Streaming Server w/ HTML/Jacascript or NodeJS client

### Server Component (FastAPI)

The FastAPI server handles:

1. **Text Processing**: Normalizes and chunks the input text
2. **Phoneme Conversion**: Converts text to phonemes using your existing pipeline
3. **Audio Generation**: Processes each chunk and generates audio
4. **Streaming Response**: Returns audio chunks as they're generated

**Key Features:**
- Efficient chunking of long text
- Real-time streaming of audio as it's generated
- Support for multiple speakers
- Configurable parameters (length scale, noise scale)
- Clean API endpoints for TTS, health check, and speaker information

### Web Client (HTML/JavaScript)

The web client provides:

1. **User Interface**: Clean, responsive design with controls for all parameters
2. **Audio Streaming**: Processes audio chunks as they arrive
3. **Visualization**: Real-time audio waveform visualization
4. **Playback Controls**: Start/stop functionality for audio playback

**Key Features:**
- Real-time audio visualization
- Controls for all TTS parameters
- Audio playback via Web Audio API
- Clean error handling and status updates

### Node.js Client

For server-side applications, the Node.js client offers:

1. **Streaming API Integration**: Handles API requests and streaming responses
2. **Audio Playback**: Optional direct playback using the Speaker library
3. **File Saving**: Option to save streamed audio to a file

## How to Use

### Server Setup

1. Install requirements:
```bash
pip install fastapi uvicorn piper-tts langchain underthesea vinorm
```

2. Run the FastAPI server:
```bash
python piper_tts_server.py
```

3. The server will start on `http://localhost:8000` and load your model.

### Client Usage

**Web Client:**
1. Open the HTML file in a browser
2. Enter text and adjust parameters
3. Click "Speak" to begin streaming

**Node.js Client:**
1. Install dependencies:
```bash
npm install node-fetch speaker
```

2. Use in your code:
```javascript
const { streamTTS } = require('./node-client.js');

// Stream and play audio
streamTTS('Your text here');

// Stream to file
streamTTS('Your text here', { output: 'output.wav' });
```

## Technical Details

### How the Streaming Works

1. **Server-side:**
   - Text is broken into small, processable chunks
   - Each chunk is converted to phonemes, then to audio
   - Audio data is streamed back as it's generated using FastAPI's `StreamingResponse`

2. **Client-side:**
   - Browser: Uses `response.body.getReader()` to read chunks as they arrive
   - Node.js: Pipes response through streams to handle audio data

### Performance Considerations

- **Memory Efficiency**: Processes and streams one chunk at a time
- **Latency Control**: Small chunks allow for faster initial playback
- **Error Handling**: Both server and client handle errors gracefully

## Customization Options

You can customize:

1. **Chunk Size**: Adjust `chunk_size` in the server for different latency/quality tradeoffs
2. **Speaker Voices**: Add more to the `SPEAKER_IDS` dictionary
3. **Audio Parameters**: Adjust range values in the client UI
4. **CORS Settings**: Configure allowed origins for production use

Enjoy.