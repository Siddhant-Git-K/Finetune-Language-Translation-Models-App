
import os
import sys
import json
import threading
import pyaudio
from vosk import Model, KaldiRecognizer
import torch
from transformers import MarianMTModel, MarianTokenizer
import subprocess
import signal
import time
import tempfile
from gtts import gTTS
import pygame

# ============= CONFIGURATION =============
MODEL_PATH = "model"
TRANSLATOR_PATH = "my_hi_translator"
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096
MIN_CONFIDENCE = 0.7
MIN_WORD_LENGTH = 2

running = True

def signal_handler(sig, frame):
    global running
    print("\n[SYSTEM] Shutdown signal received...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============= AUDIO ROUTING =============
def setup_bluetooth_audio():
    
    try:
        print("[Audio] Checking for Bluetooth audio devices...")
        
        result = subprocess.run(['pactl', 'list', 'short', 'sinks'],
                              capture_output=True, text=True)
        
        bluetooth_sink = None
        for line in result.stdout.split('\n'):
            if 'bluez_sink' in line.lower():
                bluetooth_sink = line.split()[1]
                print(f"[Audio] Found Bluetooth sink: {bluetooth_sink}")
                break
        
        if bluetooth_sink:
            subprocess.run(['pactl', 'set-default-sink', bluetooth_sink])
            print("[Audio]  Bluetooth audio output configured")
        else:
            print("[Audio] No Bluetooth audio found, using default")
        
        result = subprocess.run(['pactl', 'list', 'short', 'sources'],
                              capture_output=True, text=True)
        
        bluetooth_source = None
        for line in result.stdout.split('\n'):
            if 'bluez_source' in line.lower() or 'input' in line.lower():
                parts = line.split()
                if len(parts) > 1:
                    bluetooth_source = parts[1]
                    print(f"[Audio] Found audio source: {bluetooth_source}")
                    break
        
        if bluetooth_source:
            subprocess.run(['pactl', 'set-default-source', bluetooth_source])
            print("[Audio]  Audio input configured")
        
    except Exception as e:
        print(f"[Audio] Warning: {e}")

# ============= TEXT-TO-SPEECH (HIGH QUALITY) =============
# Initialize pygame mixer
pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=512)

def speak_text_gtts(text):
    """Speak Hindi text using Google TTS (natural human voice)"""
    if not text or not text.strip():
        return
    
    try:
        print(f"[Speaking] {text}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
        
        # Generate Hindi speech with Google TTS
        tts = gTTS(text=text, lang='hi', slow=False)
        tts.save(temp_file)
        
        # Play using pygame
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Cleanup
        pygame.mixer.music.unload()
        time.sleep(0.1)
        
        try:
            os.remove(temp_file)
        except:
            pass
            
    except Exception as e:
        print(f"[TTS Error] Google TTS failed: {e}")
        print("[TTS] Falling back to espeak...")
        speak_text_espeak(text)

def speak_text_espeak(text):
    """Fallback: Improved espeak (works offline)"""
    if not text or not text.strip():
        return
    
    try:
        subprocess.run(
            ['espeak', '-v', 'hi+f3', '-s', '140', '-p', '60', '-a', '180', '-g', '10', text],
            check=True,
            stderr=subprocess.DEVNULL
        )
    except:
        pass

def speak_text(text):
    """Main TTS function - uses best available method"""
    speak_text_gtts(text)

# ============= TRANSLATOR =============
class Translator:
    def __init__(self, model_path=TRANSLATOR_PATH):
        print("[Translator] Loading model...")
        
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found at: {model_path}")
            sys.exit(1)
        
        self.device = "cpu"
        print(f"[Translator] Using device: {self.device}")
        
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = MarianMTModel.from_pretrained(model_path, local_files_only=True).to(self.device)
            self.model.eval()
            print("[Translator] Model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            sys.exit(1)
    
    def translate(self, text, max_length=128, num_beams=4):
        if not text or not text.strip():
            return ""
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                                  truncation=True, max_length=max_length)
            
            with torch.no_grad():
                translated = self.model.generate(**inputs, max_length=max_length, num_beams=num_beams)
            
            return self.tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            print(f"[Translation Error] {e}")
            return ""

# ============= SPEECH RECOGNITION =============
def init_audio():
    print("[Audio] Initializing microphone...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Vosk model not found")
        sys.exit(1)
    
    try:
        vosk_model = Model(MODEL_PATH)
        print("[Audio] Speech recognition model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load Vosk model: {e}")
        sys.exit(1)
    
    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    recognizer.SetWords(True)
    
    p = pyaudio.PyAudio()
    
    print("[Audio] Available input devices:")
    input_device_index = None
    bluetooth_device_index = None
    
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:
            print(f"  [{i}] {dev_info['name']}")
            
            if 'bluez' in dev_info['name'].lower() or 'bluetooth' in dev_info['name'].lower():
                bluetooth_device_index = i
            elif input_device_index is None:
                input_device_index = i
    
    if bluetooth_device_index is not None:
        input_device_index = bluetooth_device_index
        print(f"[Audio] ✓ Using Bluetooth microphone")
    
    if input_device_index is None:
        print("[ERROR] No microphone found!")
        p.terminate()
        sys.exit(1)
    
    selected_device = p.get_device_info_by_index(input_device_index)
    print(f"[Audio] Selected: {selected_device['name']}")
    
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                       input=True, input_device_index=input_device_index,
                       frames_per_buffer=8192)
    except Exception as e:
        print(f"[ERROR] Failed to open audio stream: {e}")
        p.terminate()
        sys.exit(1)
    
    return p, stream, recognizer

# ============= MAIN PROGRAM =============
def main():
    global running
    
    print("\n" + "="*70)
    print(" "*15 + "ENGLISH → HINDI SPEECH TRANSLATOR")
    print(" "*20 + "(NATURAL VOICE QUALITY)")
    print("="*70)
    
    setup_bluetooth_audio()
    time.sleep(2)
    
    translator = Translator()
    p, stream, recognizer = init_audio()
    
    print("\n[System Status]")
    print("  ✓ Translator model loaded")
    print("  ✓ Speech recognition ready")
    print("  ✓ Natural Hindi voice (Google TTS)")
    print("  ✓ Bluetooth audio configured")
    print("  ✓ Confidence filtering enabled")
    print("\n[Instructions]")
    print("  • Speak clearly in English")
    print("  • Natural Hindi voice will play")
    print("\n[Commands]")
    print("  • Say 'exit', 'stop', or 'quit' to shutdown")
    print("="*70 + "\n")
    
    stream.start_stream()
    
    try:
        while running:
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get('text', '').strip()
                    
                    if 'result' in result:
                        words = result['result']
                        confident_words = [
                            w['word'] for w in words 
                            if w.get('conf', 0) >= MIN_CONFIDENCE and len(w['word']) >= MIN_WORD_LENGTH
                        ]
                        
                        filtered_text = ' '.join(confident_words).strip()
                        
                        if filtered_text:
                            text = filtered_text
                            print(f"\n{'─'*70}")
                            print(f"[YOU SAID] {text}")
                            
                            if text.lower() in ['exit', 'stop', 'quit', 'bye', 'shutdown']:
                                print("\n[SYSTEM] Shutting down...")
                                running = False
                                break
                            
                            try:
                                hindi_text = translator.translate(text)
                                
                                if hindi_text:
                                    print(f"[HINDI]    {hindi_text}")
                                    
                                    tts_thread = threading.Thread(target=speak_text, args=(hindi_text,))
                                    tts_thread.daemon = True
                                    tts_thread.start()
                                    
                            except Exception as e:
                                print(f"[ERROR] Translation failed: {e}")
                    
                    elif text and len(text) >= MIN_WORD_LENGTH:
                        print(f"\n{'─'*70}")
                        print(f"[YOU SAID] {text}")
                        
                        if text.lower() in ['exit', 'stop', 'quit']:
                            running = False
                            break
                        
                        try:
                            hindi_text = translator.translate(text)
                            if hindi_text:
                                print(f"[HINDI]    {hindi_text}")
                                threading.Thread(target=speak_text, args=(hindi_text,), daemon=True).start()
                        except Exception as e:
                            print(f"[ERROR] {e}")
            
            except IOError as e:
                if e.errno == -9981:
                    continue
                else:
                    raise
    
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted")
    
    finally:
        print("\n[SYSTEM] Cleaning up...")
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        p.terminate()
        pygame.mixer.quit()
        print("[SYSTEM] Shutdown complete")

if __name__ == "__main__":
    main()