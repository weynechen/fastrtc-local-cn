# FastRTC Local Chinese Demo

A real-time voice conversation system built with FastRTC, featuring Chinese speech recognition (STT) and text-to-speech (TTS) capabilities running locally.

## Features

- **Real-time Voice Interaction**: Stream-based audio processing with interrupt capability
- **Chinese Speech Recognition**: Powered by FunASR's SenseVoiceSmall model
- **Chinese Text-to-Speech**: Using Kokoro v1.1 Chinese TTS model
- **Local LLM Integration**: Compatible with Ollama for local language model inference
- **Web Interface**: Beautiful Gradio-based UI for easy interaction
- **GPU Acceleration**: Automatic CUDA detection for enhanced performance

## Prerequisites

- Python 3.12+
- CUDA-compatible GPU (optional, for acceleration)
- [Ollama](https://ollama.com/) running locally on port 11434
- Qwen2.5 model installed in Ollama

## Installation

1. Clone the repository:
```bash
git clone https://github.com/weynechen/fastrtc-local-cn
cd fastrtc-local-cn
```

2. Install dependencies using uv (recommended):
```bash
pip install uv
uv sync
```

3. Set up Ollama and install Qwen2.5:
```bash
# Install Ollama (visit https://ollama.com/ for instructions)
ollama pull qwen2.5:latest
ollama serve
```

## Usage

1. Start the application:
```bash
uv run main.py
```

2. Open your web browser and navigate to the displayed URL (typically `http://localhost:7860`)

3. Click the microphone button to start voice conversation

4. Speak in Chinese - the system will:
   - Convert your speech to text using FunASR
   - Process the text through Qwen2.5 LLM
   - Generate audio response using Kokoro TTS
   - Stream the audio back in real-time

## Architecture

### Components

- **main.py**: Core application with FastRTC stream handling and Gradio UI
- **stt_adapter.py**: Speech-to-text adapter using FunASR SenseVoiceSmall
- **tts_adapter.py**: Text-to-speech adapter using Kokoro v1.1 Chinese model

### Audio Processing Pipeline

1. **Audio Input**: Microphone captures voice at 16kHz
2. **Voice Activity Detection**: FastRTC's built-in VAD detects speech
3. **Speech Recognition**: FunASR converts speech to Chinese text
4. **Language Processing**: Ollama/Qwen2.5 generates intelligent responses
5. **Text-to-Speech**: Kokoro synthesizes natural Chinese speech at 24kHz
6. **Audio Output**: Real-time streaming with interrupt capability

### Key Features

- **Streaming TTS**: Text is processed sentence by sentence for faster response
- **Interrupt Capability**: Users can interrupt AI responses
- **Smart Sentence Segmentation**: Automatic detection of complete sentences
- **GPU Optimization**: Automatic CUDA utilization when available

## Configuration

### STT Configuration
- Model: `iic/SenseVoiceSmall`
- Language: Chinese (zh)
- VAD: 30-second maximum single segment time
- Format: 16-bit PCM, 16kHz

### TTS Configuration
- Model: `hexgrad/Kokoro-82M-v1.1-zh`
- Voice: `zf_001` (female voice)
- Sample Rate: 24kHz
- Dynamic speed adjustment based on text length

### LLM Configuration
- Model: `ollama/qwen2.5:latest`
- API Base: `http://localhost:11434`
- System prompt: Configured for helpful, concise Chinese responses

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure sufficient disk space for model downloads
   - Check internet connection for initial model downloads

2. **CUDA Issues**:
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch CUDA compatibility: `torch.cuda.is_available()`

3. **Ollama Connection**:
   - Verify Ollama is running: `curl http://localhost:11434/api/tags`
   - Ensure Qwen2.5 model is installed: `ollama list`

4. **Audio Issues**:
   - Check microphone permissions
   - Verify audio device compatibility
   - Test with shorter audio clips (>2 seconds required)

### Performance Optimization

- Use GPU acceleration when available
- Adjust `audio_chunk_duration` for latency vs. quality trade-offs
- Tune VAD thresholds for your environment

## Dependencies

Core dependencies include:
- `fastrtc[stt,tts,vad]>=0.0.28`: Real-time communication framework
- `funasr>=1.2.6`: Speech recognition
- `kokoro>=0.9.4`: Text-to-speech synthesis
- `litellm>=1.72.6`: LLM integration
- `torch>=2.7.1`: Deep learning framework
- `gradio`: Web interface

## License

MIT

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review logs for detailed error information
3. Ensure all prerequisites are properly installed
4. Verify model compatibility and versions
