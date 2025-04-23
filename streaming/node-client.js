const fs = require('fs');
const fetch = require('node-fetch');
const { Readable } = require('stream');
const Speaker = require('speaker');

/**
 * Stream TTS audio from Piper API and play it
 * @param {string} text - The text to convert to speech
 * @param {Object} options - TTS options
 */
async function streamTTS(text, options = {}) {
  // API endpoint
  const apiUrl = 'http://localhost:8000/tts/stream';
  
  // Default options
  const defaultOptions = {
    speaker: 'vi_male',
    length_scale: 1.0,
    noise_scale: 0.667,
    noise_scale_w: 0.8,
    output: null // Output file path, null for playback
  };
  
  // Merge options
  const ttsOptions = { ...defaultOptions, ...options };
  
  console.log(`Converting text to speech with ${ttsOptions.speaker} voice...`);
  
  try {
    // Make the API request
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text,
        speaker: ttsOptions.speaker,
        length_scale: ttsOptions.length_scale,
        noise_scale: ttsOptions.noise_scale,
        noise_scale_w: ttsOptions.noise_scale_w
      })
    });
    
    if (!response.ok) {
      throw new Error(`API responded with status ${response.status}`);
    }
    
    // The response body is a ReadableStream
    const reader = response.body;
    
    if (ttsOptions.output) {
      // Save to file
      const fileStream = fs.createWriteStream(ttsOptions.output);
      reader.pipe(fileStream);
      
      return new Promise((resolve, reject) => {
        fileStream.on('finish', () => {
          console.log(`Audio saved to ${ttsOptions.output}`);
          resolve(ttsOptions.output);
        });
        
        fileStream.on('error', (err) => {
          reject(err);
        });
      });
    } else {
      // Play through speakers
      
      // First 44 bytes are the WAV header - we need to parse this to set up speaker correctly
      const headerChunks = [];
      let headerSize = 0;
      let sampleRate = 22050; // Default, will be overridden from WAV header
      
      // Use NodeJS stream to process the incoming data
      const audioStream = new Readable({
        read() {} // No-op implementation required
      });
      
      // Process the incoming data
      reader.on('data', (chunk) => {
        if (headerSize < 44) {
          headerChunks.push(chunk.slice(0, Math.min(44 - headerSize, chunk.length)));
          headerSize += chunk.length;
          
          // If we have the full header, parse it to get sample rate
          if (headerSize >= 44) {
            const headerBuffer = Buffer.concat(headerChunks).slice(0, 44);
            sampleRate = headerBuffer.readUInt32LE(24);
            
            // Initialize speaker with the correct sample rate
            const speaker = new Speaker({
              channels: 1,
              bitDepth: 16,
              sampleRate: sampleRate
            });
            
            // Connect the stream to the speaker
            audioStream.pipe(speaker);
            
            // Push remaining data
            if (chunk.length > 44 - (headerSize - chunk.length)) {
              audioStream.push(chunk.slice(44 - (headerSize - chunk.length)));
            }
          }
        } else {
          // Push the data to the audio stream
          audioStream.push(chunk);
        }
      });
      
      // Handle end of stream
      reader.on('end', () => {
        console.log('Audio playback complete');
        audioStream.push(null); // End the stream
      });
      
      reader.on('error', (err) => {
        console.error('Error during streaming:', err);
        audioStream.emit('error', err);
      });
      
      return new Promise((resolve, reject) => {
        audioStream.on('end', resolve);
        audioStream.on('error', reject);
      });
    }
  } catch (error) {
    console.error('Error streaming TTS:', error);
    throw error;
  }
}

/**
 * Example usage
 */
async function main() {
  const text = 'Trước tình hình tăng giá xăng dầu, Ban Dân Nguyện kiến nghị Uỷ ban thường vụ Quốc Hội đề nghị Chính phủ, Thủ tướng Chính phủ quan tâm chỉ đạo Bộ Công thương, Bộ Tài chính và các Bộ ngành liên quan';
  
  try {
    // Example 1: Stream and play audio
    await streamTTS(text);
    
    // Example 2: Stream to file
    await streamTTS(text, { 
      speaker: 'vi_female',
      output: 'output.wav'
    });
    
    console.log('All examples completed');
  } catch (error) {
    console.error('Error in example:', error);
  }
}

// Run the example if this file is executed directly
if (require.main === module) {
  main();
}

module.exports = { streamTTS };
