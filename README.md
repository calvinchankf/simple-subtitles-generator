# Real-time Subtitle Translator

A command-line tool that transcribes live audio and translates it to Traditional Chinese in real-time. Based on `simple-meeting-note-taker` with added translation capabilities.

## Demo

![img](/readme_static/demo.png)

## Features

🎤 **Real-time Audio Transcription**
- Uses faster-whisper for high-performance transcription (up to 4x faster)
- Voice Activity Detection (VAD) for smart speech segmentation
- Multi-language support with automatic language detection

🌐 **Instant Translation**
- Translates transcribed speech to Traditional Chinese (繁體中文)
- Uses Google Translate API
- Displays bilingual subtitles in real-time

📝 **Subtitle Display & Export**
- Color-coded bilingual subtitles in terminal
- Shows both original and translated text
- Auto-saves subtitles with timestamps
- Performance metrics (transcription/translation time)

⚡ **Optimized Performance**
- GPU and CPU support
- Configurable precision (int8, float16, float32)
- Voice Activity Detection to reduce unnecessary processing

## Prerequisites

### macOS
```bash
# Install PortAudio for PyAudio
brew install portaudio

# Install FFmpeg for audio processing
brew install ffmpeg
```

### Linux
```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg

# Fedora
sudo dnf install portaudio-devel ffmpeg
```

### Windows
- Download and install FFmpeg from https://ffmpeg.org/download.html
- PyAudio wheels available via pip

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd subtitle-translator
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python subtitle_translator.py
```

The tool will prompt you to configure:
- **Model**: Choose Whisper model (tiny/base/small/medium/large)
- **Device**: CPU or CUDA (GPU)
- **Compute type**: int8 (fastest), float16, or float32
- **VAD mode**: 0-3 (speech detection aggressiveness)

### Quick Start (Recommended Settings)

For best balance of speed and quality:
- Model: `base`
- Device: `cpu` (or `cuda` if you have NVIDIA GPU)
- Compute type: `int8`
- VAD mode: `2`

### How It Works

1. **Speak into your microphone** - The tool listens for speech
2. **Voice Activity Detection** - Detects when you start and stop speaking
3. **Transcription** - Converts speech to text using faster-whisper
4. **Translation** - Translates text to Traditional Chinese
5. **Display** - Shows bilingual subtitles in color-coded format:

```
======================================================================
[14:23:15] Segment #1
----------------------------------------------------------------------
 ORIGINAL (EN)
Hello, how are you today?

 中文翻譯 (TRADITIONAL CHINESE)
你好，你今天好嗎？
----------------------------------------------------------------------
⚡ Transcription: 0.45s | Translation: 0.23s | Confidence: 0.98
======================================================================
```

### Output Files

Subtitles are automatically saved to `subtitles/` directory:
- Filename format: `subtitles_YYYYMMDD_HHMMSS.txt`
- Contains both bilingual subtitles and continuous transcripts
- Includes metadata and performance statistics

## Model Selection Guide

| Model  | VRAM | Speed | Accuracy | Best For |
|--------|------|-------|----------|----------|
| tiny   | ~1GB | Fastest | Good | Quick testing, low-end hardware |
| base   | ~1GB | Fast | Better | **Recommended for most users** |
| small  | ~2GB | Medium | High | Better accuracy needed |
| medium | ~5GB | Slow | Higher | Professional use |
| large  | ~10GB | Slowest | Highest | Maximum accuracy |

## VAD (Voice Activity Detection) Modes

- **0**: Quality mode - least aggressive, may capture more silence
- **1**: Low bitrate mode
- **2**: Aggressive - **recommended for general use**
- **3**: Very aggressive - best for noisy environments

## Translation Languages

Currently configured for Traditional Chinese (`zh-TW`). To translate to other languages, modify the `target_lang` parameter:

Common language codes:
- `zh-TW`: Traditional Chinese (繁體中文)
- `zh-CN`: Simplified Chinese (简体中文)
- `ja`: Japanese (日本語)
- `ko`: Korean (한국어)
- `es`: Spanish
- `fr`: French
- `de`: German

## Troubleshooting

### No audio detected
- Check microphone permissions in system settings
- Verify microphone is working with `python -m pyaudio` test
- Try different VAD modes (0-3)

### Translation errors
- Requires internet connection for Google Translate API
- If you see "Translation error", check your network connection
- Consider implementing retry logic for unstable connections

### PyAudio installation issues (macOS)
```bash
brew install portaudio
pip install pyaudio
```

### Memory errors
- Use smaller model (tiny or base)
- Use int8 compute type
- Close other applications

### Slow transcription
- Switch to smaller model
- Use int8 compute type
- Enable GPU if available (device: cuda)

## Performance Tips

1. **For fastest speed**: Use `tiny` or `base` model with `int8` compute type
2. **For best accuracy**: Use `medium` or `large` model with `float16` (requires GPU)
3. **For balanced performance**: Use `base` model with `int8` on CPU
4. **Reduce latency**: Use more aggressive VAD mode (2 or 3)

## Advanced Configuration

Edit `subtitle_translator.py` to customize:
- Default target language (line 40)
- Audio settings (lines 44-48)
- VAD parameters (lines 51-53)
- Output formatting (method `display_subtitle`)

## Credits

Built on top of:
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast Whisper transcription
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [googletrans](https://github.com/ssut/googletrans) - Translation
- [webrtcvad](https://github.com/wiseman/py-webrtcvad) - Voice Activity Detection

## License

Based on the `simple-meeting-note-taker` project.

## Future Enhancements

- [ ] Support for multiple translation services (DeepL, Azure)
- [ ] SRT/VTT subtitle file export for video editing
- [ ] WebSocket server for OBS subtitle overlay
- [ ] Configurable subtitle display styles
- [ ] Batch translation mode for reduced latency
- [ ] Support for translating between non-English language pairs
