#!/usr/bin/env python3
"""
Real-time Subtitle Translator - Live audio transcription with translation.

Transcribes live audio using faster-whisper and translates to Traditional Chinese
using Google Translate API. Displays bilingual subtitles in real-time.
"""

import pyaudio
import wave
from faster_whisper import WhisperModel
import webrtcvad
import threading
import time
import tempfile
import os
import queue
import numpy as np
from datetime import datetime
import collections
from googletrans import Translator
from colorama import Fore, Back, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

class SubtitleTranslator:
    def __init__(self, model_name="base", device="cpu", compute_type="int8",
                 vad_mode=2, target_lang="zh-TW", save_to_file=True):
        """
        Initialize the subtitle translator.

        Args:
            model_name (str): Whisper model to use (tiny, base, small, medium, large-v1, large-v2, large-v3)
            device (str): Device to use ("cpu", "cuda")
            compute_type (str): Computation type ("int8", "float16", "float32")
            vad_mode (int): VAD aggressiveness (0-3, 3 is most aggressive)
            target_lang (str): Target language for translation (default: zh-TW for Traditional Chinese)
            save_to_file (bool): Whether to save subtitles to file (default: True)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.vad_mode = vad_mode
        self.target_lang = target_lang
        self.save_to_file = save_to_file
        self.model = None
        self.translator = Translator()
        self.is_recording = False

        # Audio settings optimized for WebRTC VAD
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 480  # 30ms at 16kHz
        self.audio_format = pyaudio.paInt16

        # VAD settings
        self.vad = webrtcvad.Vad(vad_mode)
        self.frame_duration_ms = 30
        self.padding_duration_ms = 300
        self.ring_buffer_size = self.padding_duration_ms // self.frame_duration_ms

        # Speech detection
        self.num_padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)
        self.ring_buffer = collections.deque(maxlen=self.ring_buffer_size)
        self.triggered = False
        self.voiced_frames = []

        # Performance tracking
        self.transcription_times = []
        self.translation_times = []
        self.total_segments = 0

        # Subtitle storage
        self.subtitles = []
        self.session_start_time = None

        # Threading
        self.audio_queue = queue.Queue()
        self.transcription_thread = None
        self.recording_thread = None

        # PyAudio instance
        self.p = pyaudio.PyAudio()

    def load_model(self):
        """Load the faster-whisper model."""
        print(f"{Fore.CYAN}Loading faster-whisper model: {self.model_name}")
        print(f"{Fore.CYAN}Device: {self.device} | Compute type: {self.compute_type}")

        try:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=None,
                local_files_only=False
            )
            print(f"{Fore.GREEN}✅ Faster-Whisper model loaded successfully!")

        except Exception as e:
            print(f"{Fore.RED}❌ Error loading model: {e}")
            print(f"{Fore.YELLOW}Falling back to CPU with int8...")
            self.device = "cpu"
            self.compute_type = "int8"
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def list_audio_devices(self):
        """List available audio input devices."""
        print("\nAvailable audio devices:")
        print("-" * 40)
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"Device {i}: {info['name']} (Input channels: {info['maxInputChannels']})")

    def is_speech(self, frame):
        """Check if frame contains speech using WebRTC VAD."""
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception:
            return False

    def save_audio_segment(self, frames):
        """Save audio frames to a temporary WAV file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name

            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.audio_format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            return temp_filename

    def record_audio_vad(self):
        """Record audio using VAD to detect speech segments."""
        try:
            stream = self.p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            print(f"{Fore.GREEN}🎤 Recording started!")
            print(f"{Fore.CYAN}⚡ Real-time subtitle translation active")
            print(f"{Fore.CYAN}📊 VAD Mode: {self.vad_mode} | Device: {self.device}")
            print(f"{Fore.CYAN}🌐 Target language: {self.target_lang} (Traditional Chinese)")
            print(f"{Fore.YELLOW}🎯 Waiting for speech... (speak naturally)")
            print(f"{Fore.YELLOW}Press Ctrl+C to stop\n")

            while self.is_recording:
                frame = stream.read(self.chunk_size, exception_on_overflow=False)
                is_speech = self.is_speech(frame)

                if not self.triggered:
                    self.ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, speech in self.ring_buffer if speech])

                    if num_voiced > 0.9 * self.ring_buffer.maxlen:
                        self.triggered = True
                        print(f"{Fore.MAGENTA}[{datetime.now().strftime('%H:%M:%S')}] 🗣️  Speech detected, recording...")
                        self.voiced_frames.extend([f for f, s in self.ring_buffer])
                        self.ring_buffer.clear()

                else:
                    self.voiced_frames.append(frame)
                    self.ring_buffer.append((frame, is_speech))
                    num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])

                    if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                        self.triggered = False
                        print(f"{Fore.MAGENTA}[{datetime.now().strftime('%H:%M:%S')}] 🔇 Speech ended, processing...")

                        if len(self.voiced_frames) > 10:
                            temp_filename = self.save_audio_segment(self.voiced_frames)
                            self.audio_queue.put(temp_filename)

                        self.voiced_frames = []
                        self.ring_buffer.clear()
                        print(f"{Fore.YELLOW}[{datetime.now().strftime('%H:%M:%S')}] 🎯 Waiting for speech...")

        except Exception as e:
            print(f"{Fore.RED}Error recording audio: {e}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()

            if self.voiced_frames:
                temp_filename = self.save_audio_segment(self.voiced_frames)
                self.audio_queue.put(temp_filename)

    def translate_text(self, text, source_lang='auto'):
        """
        Translate text using Google Translate.

        Args:
            text (str): Text to translate
            source_lang (str): Source language (default: 'auto' for auto-detection)

        Returns:
            tuple: (translated_text, detected_source_lang)
        """
        try:
            translation = self.translator.translate(text, src=source_lang, dest=self.target_lang)
            return translation.text, translation.src
        except Exception as e:
            print(f"{Fore.RED}❌ Translation error: {e}")
            return text, source_lang

    def display_subtitle(self, timestamp, original_text, translated_text, language, confidence,
                        transcription_time, translation_time):
        """Display bilingual subtitles in a formatted way."""
        print("\n" + "=" * 70)
        print(f"{Fore.CYAN}[{timestamp}] Segment #{self.total_segments}")
        print("-" * 70)
        print(f"{Fore.WHITE}{Back.BLUE} ORIGINAL ({language.upper()}) {Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{original_text}")
        print()
        print(f"{Fore.WHITE}{Back.GREEN} 中文翻譯 (TRADITIONAL CHINESE) {Style.RESET_ALL}")
        print(f"{Fore.GREEN}{translated_text}")
        print("-" * 70)
        print(f"{Fore.CYAN}⚡ Transcription: {transcription_time:.2f}s | Translation: {translation_time:.2f}s | Confidence: {confidence:.2f}")
        print("=" * 70)

    def transcribe_worker(self):
        """Worker thread that processes speech segments and translates them."""
        while self.is_recording or not self.audio_queue.empty():
            try:
                audio_file = self.audio_queue.get(timeout=1)

                # Transcribe
                transcription_start = time.time()
                timestamp = datetime.now().strftime("%H:%M:%S")

                segments, info = self.model.transcribe(
                    audio_file,
                    beam_size=1,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    word_timestamps=False
                )

                # Extract text
                transcription_text = ""
                for segment in segments:
                    transcription_text += segment.text

                transcription_time = time.time() - transcription_start
                self.transcription_times.append(transcription_time)

                original_text = transcription_text.strip()
                if original_text:
                    # Translate
                    translation_start = time.time()
                    translated_text, detected_lang = self.translate_text(original_text, source_lang=info.language)
                    translation_time = time.time() - translation_start
                    self.translation_times.append(translation_time)

                    self.total_segments += 1

                    # Display subtitle
                    self.display_subtitle(
                        timestamp=timestamp,
                        original_text=original_text,
                        translated_text=translated_text,
                        language=info.language,
                        confidence=info.language_probability,
                        transcription_time=transcription_time,
                        translation_time=translation_time
                    )

                    # Store subtitle data
                    self.subtitles.append({
                        'timestamp': timestamp,
                        'original': original_text,
                        'translated': translated_text,
                        'language': info.language,
                        'confidence': info.language_probability,
                        'transcription_time': transcription_time,
                        'translation_time': translation_time
                    })
                else:
                    print(f"{Fore.YELLOW}[{timestamp}] ❓ No clear speech detected")

                # Clean up
                try:
                    os.unlink(audio_file)
                except OSError:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"{Fore.RED}❌ Error during processing: {e}")

    def start(self):
        """Start live subtitle translation."""
        if not self.model:
            self.load_model()

        self.session_start_time = datetime.now()
        self.is_recording = True

        # Start threads
        self.recording_thread = threading.Thread(target=self.record_audio_vad)
        self.transcription_thread = threading.Thread(target=self.transcribe_worker)

        self.recording_thread.start()
        self.transcription_thread.start()

        try:
            while self.is_recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop live translation and save subtitles."""
        print(f"\n{Fore.YELLOW}🛑 Stopping subtitle translator...")
        self.is_recording = False

        # Process final speech
        if self.voiced_frames:
            print(f"{Fore.CYAN}🔄 Processing final speech segment...")
            temp_filename = self.save_audio_segment(self.voiced_frames)
            self.audio_queue.put(temp_filename)

        # Wait for threads
        if self.recording_thread:
            self.recording_thread.join()
        if self.transcription_thread:
            self.transcription_thread.join()

        # Show statistics
        self.print_statistics()

        # Save subtitles if enabled
        if self.save_to_file:
            self.save_subtitles()
        else:
            print(f"{Fore.YELLOW}📝 Subtitle saving is disabled. Subtitles not saved to file.")

        print(f"{Fore.GREEN}✅ Subtitle translator stopped.")

    def print_statistics(self):
        """Print performance statistics."""
        if self.transcription_times:
            print(f"\n{Fore.CYAN}📊 Performance Statistics:")
            print(f"{Fore.CYAN}{'='*50}")
            print(f"   Total segments: {self.total_segments}")

            avg_trans = sum(self.transcription_times) / len(self.transcription_times)
            print(f"   Avg transcription time: {avg_trans:.2f}s")

            if self.translation_times:
                avg_transl = sum(self.translation_times) / len(self.translation_times)
                print(f"   Avg translation time: {avg_transl:.2f}s")
                print(f"   Total avg time: {avg_trans + avg_transl:.2f}s per segment")

    def save_subtitles(self, output_dir="subtitles"):
        """Save subtitles to file."""
        if not self.subtitles or not self.session_start_time:
            print(f"{Fore.YELLOW}📝 No subtitles to save.")
            return None

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"{Fore.GREEN}📁 Created directory: {output_dir}/")

        # Create filename
        filename = os.path.join(
            output_dir,
            f"subtitles_{self.session_start_time.strftime('%Y%m%d_%H%M%S')}.txt"
        )

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 70 + "\n")
                f.write("BILINGUAL SUBTITLE TRANSLATION SESSION\n")
                f.write("=" * 70 + "\n")
                f.write(f"Session started: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Target language: {self.target_lang} (Traditional Chinese)\n")
                f.write(f"Total segments: {len(self.subtitles)}\n")
                f.write("\n" + "=" * 70 + "\n\n")

                # Subtitles
                for i, sub in enumerate(self.subtitles, 1):
                    f.write(f"[{sub['timestamp']}] Segment {i}\n")
                    f.write(f"Language: {sub['language']} (confidence: {sub['confidence']:.2f})\n")
                    f.write(f"Original: {sub['original']}\n")
                    f.write(f"中文翻譯: {sub['translated']}\n")
                    f.write("-" * 50 + "\n\n")

                # Continuous transcripts
                f.write("=" * 70 + "\n")
                f.write("CONTINUOUS ORIGINAL TRANSCRIPT\n")
                f.write("=" * 70 + "\n\n")
                original_continuous = " ".join([s['original'] for s in self.subtitles])
                f.write(original_continuous + "\n\n")

                f.write("=" * 70 + "\n")
                f.write("CONTINUOUS CHINESE TRANSLATION\n")
                f.write("=" * 70 + "\n\n")
                translated_continuous = " ".join([s['translated'] for s in self.subtitles])
                f.write(translated_continuous + "\n")

            print(f"{Fore.GREEN}📝 Subtitles saved to: {filename}")
            return filename

        except Exception as e:
            print(f"{Fore.RED}❌ Error saving subtitles: {e}")
            return None

    def cleanup(self):
        """Clean up resources."""
        self.p.terminate()

def main():
    print(f"{Fore.CYAN}{'='*50}")
    print(f"{Fore.CYAN}⚡ Real-time Subtitle Translator")
    print(f"{Fore.CYAN}   Audio → Transcription → Translation")
    print(f"{Fore.CYAN}{'='*50}\n")

    # Configuration
    available_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
    devices = ["cpu", "cuda"]
    compute_types = {
        "cpu": ["int8", "float32"],
        "cuda": ["int8", "float16", "float32"]
    }
    vad_modes = {
        "0": "Quality (least aggressive)",
        "1": "Low bitrate",
        "2": "Aggressive",
        "3": "Very aggressive (best for noisy environments)"
    }

    print(f"Available models: {', '.join(available_models)}")
    model_choice = input("Choose model (default: base): ").strip() or "base"

    if model_choice not in available_models:
        print(f"Invalid model. Using 'base'")
        model_choice = "base"

    print(f"\nAvailable devices: {', '.join(devices)}")
    device_choice = input("Choose device (default: cpu): ").strip() or "cpu"

    if device_choice not in devices:
        device_choice = "cpu"

    print(f"\nCompute types for {device_choice}: {', '.join(compute_types[device_choice])}")
    compute_choice = input(f"Choose compute type (default: int8): ").strip() or "int8"

    if compute_choice not in compute_types[device_choice]:
        compute_choice = "int8"

    print(f"\nVAD modes:")
    for mode, desc in vad_modes.items():
        print(f"  {mode}: {desc}")

    vad_mode = input("Choose VAD mode (default: 2): ").strip()
    try:
        vad_mode = int(vad_mode) if vad_mode else 2
        if vad_mode not in range(4):
            vad_mode = 2
    except ValueError:
        vad_mode = 2

    print(f"\nSave subtitles to file?")
    save_choice = input("Save to file? (y/n, default: y): ").strip().lower()
    save_to_file = save_choice != 'n'

    # Initialize translator
    translator = SubtitleTranslator(
        model_name=model_choice,
        device=device_choice,
        compute_type=compute_choice,
        vad_mode=vad_mode,
        target_lang="zh-TW",  # Traditional Chinese
        save_to_file=save_to_file
    )

    # Show devices
    translator.list_audio_devices()

    print(f"\n{Fore.GREEN}🚀 Starting subtitle translator:")
    print(f"{Fore.CYAN}   Model: {model_choice}")
    print(f"{Fore.CYAN}   Device: {device_choice}")
    print(f"{Fore.CYAN}   Compute: {compute_choice}")
    print(f"{Fore.CYAN}   VAD Mode: {vad_mode} ({vad_modes[str(vad_mode)]})")
    print(f"{Fore.CYAN}   Translation: English → Traditional Chinese (繁體中文)")
    print(f"{Fore.CYAN}   Save to file: {'Yes' if save_to_file else 'No'}")

    try:
        translator.start()
    except KeyboardInterrupt:
        pass
    finally:
        translator.cleanup()

if __name__ == "__main__":
    main()
