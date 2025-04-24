const fs = require('fs');
const path = require('path');
// npm install node-fetch@2
const fetch = require('node-fetch');
// npm install speaker
const Speaker = require('speaker');
// npm install wav
const wav = require('wav');

/**
 * Client for the Piper TTS Streaming API
 */
class PiperTTSClient {
  /**
   * Create a new Piper TTS client
   * @param {Object} options - Client options
   * @param {string} [options.apiUrl='http://localhost:8000'] - URL of the Piper TTS API server
   * @param {string} [options.defaultSpeaker='vi_male'] - Default speaker name
   */
  constructor(options = {}) {
    this.apiUrl = options.apiUrl || 'http://localhost:8000';
    this.defaultSpeaker = options.defaultSpeaker || 'vi_male'; // Default Piper speaker
    this.defaultConfig = {
      length_scale: 1.0,
      noise_scale: 0.667,
      noise_scale_w: 0.8,
    };
    this.activeSpeaker = null;
    this.activeWavReader = null;
    this.activeResponseStream = null;
  }

  /** Helper to handle fetch errors */
  async _handleFetchError(response) {
    // (Same as in previous NodeJS client example)
    const status = response.status;
    let errorText = `API responded with status ${status}`;
    let errorDetail = '';
    try {
      const errorJson = await response.json();
      errorDetail = errorJson.detail || JSON.stringify(errorJson);
    } catch (e) {
      try { errorDetail = await response.text(); } catch (e2) { errorDetail = "(Could not read error response body)";}
    }
    return new Error(`${errorText}: ${errorDetail}`);
  }

  /**
   * Get available speakers from the server
   * @returns {Promise<Object>} - Available speakers { speaker_name: id }
   */
  async getSpeakers() {
    console.log(`Fetching speakers from ${this.apiUrl}/speakers`);
    try {
      const response = await fetch(`${this.apiUrl}/speakers`);
      if (!response.ok) {
        throw await this._handleFetchError(response);
      }
      return await response.json();
    } catch (error) {
      console.error(`Error fetching speakers: ${error.message}`);
      throw error;
    }
  }

  /** Check API health */
  async checkHealth() {
     // (Same as in previous NodeJS client example)
    console.log(`Checking health at ${this.apiUrl}/health`);
    try {
      const response = await fetch(`${this.apiUrl}/health`);
      const data = await response.json();
      if (!response.ok) {
          console.warn(`API health check responded with status ${response.status}`);
      }
      return data;
    } catch (error) {
      console.error(`Error checking API health: ${error.message}`);
       return { status: 'error', message: `Connection failed: ${error.message}` };
    }
  }

  /**
   * Stream TTS audio from Piper API and optionally play or save it.
   * @param {string} text - Text to synthesize
   * @param {Object} [options={}] - TTS options override
   * @param {string} [options.speaker] - Speaker name (overrides defaultSpeaker)
   * @param {number} [options.length_scale] - Length scale
   * @param {number} [options.noise_scale] - Noise scale
   * @param {number} [options.noise_scale_w] - Noise scale W
   * @param {string|null} [options.output=null] - Path to save output audio. If null, plays through speakers.
   * @returns {Promise<string|void>} - Output path if saved, void if played successfully.
   */
  async streamTTS(text, options = {}) {
    // Merge options correctly
    const config = {
      speaker: options.speaker || this.defaultSpeaker,
      length_scale: options.length_scale !== undefined ? options.length_scale : this.defaultConfig.length_scale,
      noise_scale: options.noise_scale !== undefined ? options.noise_scale : this.defaultConfig.noise_scale,
      noise_scale_w: options.noise_scale_w !== undefined ? options.noise_scale_w : this.defaultConfig.noise_scale_w,
      output: options.output !== undefined ? options.output : null // Explicitly handle output path
    };

    const requestData = {
      text: text,
      speaker: config.speaker,
      length_scale: config.length_scale,
      noise_scale: config.noise_scale,
      noise_scale_w: config.noise_scale_w
    };

    if (!requestData.text || !requestData.text.trim()) {
         throw new Error("Text cannot be empty.");
    }
    if (!requestData.speaker) {
         throw new Error("Speaker name must be specified.");
    }

    console.log(`Synthesizing with speaker '${requestData.speaker}' (Params: L=${requestData.length_scale}, N=${requestData.noise_scale}, Nw=${requestData.noise_scale_w})`);

    try {
      // Make streaming request
      const response = await fetch(`${this.apiUrl}/tts/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'audio/wav' },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw await this._handleFetchError(response);
      }
      if (!response.body) {
        throw new Error("API response did not contain a readable stream body.");
      }

      this.activeResponseStream = response.body;

      // --- Process the response stream ---
      if (config.output) {
        // --- Save to file ---
        // (Identical saving logic as previous NodeJS client)
        const outputPath = path.resolve(config.output);
        const outputDir = path.dirname(outputPath);
        try { if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true }); }
        catch(dirError) { throw new Error(`Failed to create output directory '${outputDir}': ${dirError.message}`); }

        console.log(`Saving audio stream to: ${outputPath}`);
        const fileStream = fs.createWriteStream(outputPath);
        this.activeResponseStream.pipe(fileStream);

        return new Promise((resolve, reject) => {
          fileStream.on('finish', () => { console.log(`Audio saved to ${outputPath}`); this.activeResponseStream = null; resolve(outputPath); });
          fileStream.on('error', (err) => { console.error(`Error writing file: ${err.message}`); this.activeResponseStream = null; reject(err); });
          this.activeResponseStream.on('error', (err) => {
              console.error(`Error reading response stream: ${err.message}`);
              fileStream.close(); try { if(fs.existsSync(outputPath)) fs.unlinkSync(outputPath); } catch (e) {} this.activeResponseStream = null; reject(err);
          });
        });

      } else {
        // --- Play through speakers ---
        console.log("Streaming audio to speakers...");
        this.stopPlayback(); // Stop previous playback

        return new Promise((resolve, reject) => {
            // Use wav.Reader for robust header parsing
            const wavReader = new wav.Reader();
            this.activeWavReader = wavReader;

            wavReader.on('format', (format) => {
                console.log(`Audio Format: ${format.channels}ch, ${format.sampleRate}Hz, ${format.bitDepth}bit, PCM: ${format.audioFormat === 1}`);
                if (format.audioFormat !== 1) {
                     const fmtErr = new Error(`Unsupported WAV format: ${format.audioFormat}`); this.stopPlayback(); return reject(fmtErr);
                }
                 try {
                    this.activeSpeaker = new Speaker(format);
                    this.activeSpeaker.on('error', (err) => { console.error("Speaker error:", err.message); this.activeSpeaker = null; this.activeWavReader = null; reject(err); });
                    this.activeSpeaker.on('close', () => { console.log("Speaker closed."); this.activeSpeaker = null; this.activeWavReader = null; resolve(); });
                    console.log("Piping parsed WAV to speaker...");
                    this.activeWavReader.pipe(this.activeSpeaker);
                 } catch(spkErr) { console.error("Speaker init error:", spkErr.message); this.stopPlayback(); reject(spkErr); }
            });

            wavReader.on('error', (err) => { console.error("WAV parsing error:", err.message); this.stopPlayback(); reject(err); });
            this.activeResponseStream.pipe(wavReader);
            this.activeResponseStream.on('error', (err) => { console.error(`Response stream error: ${err.message}`); this.activeWavReader = null; this.activeSpeaker = null; this.activeResponseStream = null; reject(err); });
            this.activeResponseStream.on('close', () => { console.log("Source stream closed."); });
        });
      }
    } catch (error) {
      console.error(`Error in streamTTS setup: ${error.message}`);
      this.stopPlayback();
      this.activeResponseStream = null;
      throw error;
    }
  }

 /** Stop current playback */
 stopPlayback() {
     // (Identical stopping logic as previous NodeJS client)
     if (this.activeSpeaker) { console.log("Stopping speaker..."); try { this.activeSpeaker.end(); } catch (e) { console.error("Err ending speaker:", e.message); } this.activeSpeaker = null; }
     if (this.activeWavReader) { console.log("Stopping WAV reader..."); try { if (this.activeResponseStream) this.activeResponseStream.unpipe(this.activeWavReader); this.activeWavReader.destroy(); } catch (e) { console.error("Err stopping WAV reader:", e.message); } this.activeWavReader = null; }
     if (this.activeResponseStream) { console.log("Stopping response stream..."); try { this.activeResponseStream.destroy(); } catch(e) { console.error("Err destroying response stream:", e.message); } this.activeResponseStream = null; }
 }

} // End of PiperTTSClient class

// --- Example Usage ---
async function main() {
  const client = new PiperTTSClient({
    apiUrl: 'http://localhost:8000', // Ensure port matches server
    defaultSpeaker: 'vi_male' // Default speaker for examples
  });

  const outputDir = './output_piper'; // Separate output dir
  if (!fs.existsSync(outputDir)) { try { fs.mkdirSync(outputDir); } catch(e) { console.error(`Failed to create ${outputDir}: ${e.message}`); } }

  try {
    // 1. Health Check
    console.log("\n--- 1. Health Check ---");
    const health = await client.checkHealth();
    console.log('Server health:', health);
    if (health.status === 'error') { console.error("Server not healthy! Exiting."); return; }

    // 2. Get Speakers
    console.log("\n--- 2. Get Speakers ---");
    const speakers = await client.getSpeakers();
    const speakerNames = Object.keys(speakers);
    console.log('Available speakers:', speakerNames.join(', ') || 'None found');
    if (speakerNames.length === 0 && health.status !== 'error') {
         console.warn("Model loaded, but no speakers reported by server?");
    }
    // Select a female speaker if available, otherwise use default
    const femaleSpeaker = speakerNames.find(name => name.includes('female')) || client.defaultSpeaker;
    const maleSpeaker = speakerNames.find(name => name.includes('male')) || client.defaultSpeaker;


    // 3. Playback with Male Speaker (default or found)
    console.log(`\n--- 3. Playback (Speaker: ${maleSpeaker}) ---`);
    try {
        await client.streamTTS(
            'Xin chào, đây là bản demo truyền phát máy khách nút js cho Piper TTS.',
            { speaker: maleSpeaker }
        );
        console.log("Playback 3 finished.");
    } catch(e) { console.error(`Playback 3 failed: ${e.message}`); }
    await new Promise(resolve => setTimeout(resolve, 300));


    // 4. Save to file with Female Speaker and different params
    console.log(`\n--- 4. Save to File (Speaker: ${femaleSpeaker}) ---`);
    try {
        const outputPath = path.join(outputDir, 'piper-node-output.wav');
        await client.streamTTS(
            'Câu này sẽ được lưu vào một tập tin với các thông số khác nhau.',
            {
                speaker: femaleSpeaker,
                length_scale: 1.1,
                noise_scale: 0.5,
                noise_scale_w: 0.7,
                output: outputPath
            }
        );
        console.log(`Save 4 finished.`);
    } catch(e) { console.error(`Save 4 failed: ${e.message}`); }


  } catch (error) {
    console.error('\n--- Error in main execution ---');
    console.error(error.message);
    // console.error(error.stack);
    console.error('---------------------------------');
  } finally {
      console.log("\n--- Example finished, stopping playback ---");
      client.stopPlayback();
  }
}

// Run example
if (require.main === module) {
  main();
}

module.exports = { PiperTTSClient }; // Export class