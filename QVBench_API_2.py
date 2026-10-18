import os
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
import jiwer
import soundfile as sf
from kokoro_onnx import Kokoro
import ollama
import site

for p in site.getsitepackages() + [site.getusersitepackages()]:
    if os.path.exists(p):
        for root, dirs, _ in os.walk(p):
            if "bin" in dirs:
                bin_path = os.path.join(root, "bin")
                try:
                    os.add_dll_directory(bin_path)
                except Exception:
                    pass
                os.environ["PATH"] = bin_path + ";" + os.environ["PATH"]

import requests

QVV_API_URL = "http://129.152.0.229:8000/evaluate" 
QVV_API_KEY = "qvbench-evalut-key-yZSovE1A72h9"     

def compute_qvv_score(reference: str, hypothesis: str) -> float:
    """Chiama il server esterno su Oracle per calcolare il QVV score."""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": QVV_API_KEY, 
    }
    payload = {
        "reference": reference,
        "hypothesis": hypothesis
    }
    try:
        response = requests.post(QVV_API_URL, json=payload, headers=headers, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        
        # Recupera il valore numerico (gestisce se restituisce float diretto o dizionario es. {"score": 85.5})
        if isinstance(data, dict):
            return float(data.get("score", data.get("qvv_score", data.get("result", 0.0))))
        return float(data)
    except Exception as e:
        print(f"[Errore API QVV]: {e}")
        return 0.0

MODEL_SIZE = "small"
LANG_WHISPER = "en"
LANG_KOKORO = "en-us"
SPEED = "1.8"
PARALLEL_LINES = 1

KOKORO_MODEL = "kokoro-v1.0.onnx"
KOKORO_VOICE = "af_heart" 
KOKORO_VOICES_BIN = "voices-v1.0.bin"

DEVICE = "cuda"          
COMPUTE_TYPE = "float16"

OLLAMA_MODEL = "gemma4:e2b" 
ENABLE_GEMMA = True

print("Initializing Kokoro v1.0 engine...")
kokoro_engine = Kokoro(KOKORO_MODEL, KOKORO_VOICES_BIN)

def ask_gemma_judge(ref, hyp, score):
    prompt = f"""You are a forensic linguist. Analyze this English speech-to-text transcription error.
Original: "{ref}"
Transcribed: "{hyp}"

Respond in English in a single concise sentence: Does the transcription error fundamentally alter the intended meaning, or would a native speaker easily understand it from context?"""
    
    try:
        client = ollama.Client(timeout=15.0)
        response = client.chat(model=OLLAMA_MODEL.strip(), messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        return f"[Ollama failed to respond: {e}]"

def prepare_audio_file(args):
    idx, text_input, lang_k, speed = args
    raw_audio = f"frase_{idx}_norm.wav"
    fast_audio = f"frase_{idx}_fast.mp3"

    for f in (raw_audio, fast_audio):
        if os.path.exists(f):
            os.remove(f)

    samples, sample_rate = kokoro_engine.create(
        text_input, 
        voice=KOKORO_VOICE, 
        speed=1.0, 
        lang=lang_k
    )
    sf.write(raw_audio, samples, sample_rate)

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", raw_audio,
        "-filter:a", f"atempo={speed}",
        "-loglevel", "quiet",
        fast_audio
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    
    return idx, text_input, fast_audio

def transcribe_single(model, idx, ref_sentence, fast_audio, lang_w):
    print(f"-> Transcribing line {idx}...")
    t0 = time.time()
    
    segments, _ = model.transcribe(fast_audio, beam_size=5, language=lang_w)
    live_hyp = "".join(s.text for s in segments).strip()
    latency = time.time() - t0
    print(f"-> Line {idx} transcribed in {latency:.2f}s!")

    wer_acc = max(0.0, 100.0 - (jiwer.wer(ref_sentence.lower(), live_hyp.lower()) * 100.0))
    cer_acc = max(0.0, 100.0 - (jiwer.cer(ref_sentence.lower(), live_hyp.lower()) * 100.0))
    qvv_score = compute_qvv_score(ref_sentence, live_hyp)

    gemma_comment = ""
    if qvv_score < 100.0 and ENABLE_GEMMA:
        print(f"-> Querying Ollama ({OLLAMA_MODEL}) for semantic evaluation...")
        gemma_comment = ask_gemma_judge(ref_sentence, live_hyp, qvv_score)

    return idx, ref_sentence, live_hyp, latency, wer_acc, cer_acc, qvv_score, gemma_comment

if __name__ == "__main__":
    input_file = "frasi.txt"
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"\n=== DYNQ-VOICE BENCHMARK (Kokoro + Whisper + Gemma Semantic Judge) ===")
    print(f"Test Sentences : {len(sentences)}")
    print(f"Audio Speed    : {SPEED}x")
    print(f"STT Hardware   : {DEVICE.upper()} ({COMPUTE_TYPE})")
    print("=========================================================\n")
    
    print("Synthesizing audio...")
    prep_tasks = [(idx, s, LANG_KOKORO, SPEED) for idx, s in enumerate(sentences, 1)]
    with ThreadPoolExecutor(max_workers=PARALLEL_LINES) as executor:
        prepared_audios = list(executor.map(prepare_audio_file, prep_tasks))
    print("Audio files generated successfully!\n")

    print(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE}...")
    try:
        stt_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"\n[CUDA/CTranslate2 Error]: {e}")
        print("Falling back to CPU...")
        stt_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    print("Whisper model ready! Starting benchmark...\n")

    bench_start = time.time()
    results = []

    for idx, ref, audio in prepared_audios:
        res = transcribe_single(stt_model, idx, ref, audio, LANG_WHISPER)
        results.append(res)
        
        _, ref_sentence, live_hyp, latency, wer_acc, cer_acc, qvv_score, g_note = res
        print(f"\n[Line {idx}] (STT Latency: {latency:.2f}s)")
        print(f"Orig: {ref_sentence}")
        print(f"STT : {live_hyp}")
        print(f"Score -> WER: {wer_acc:.1f}% | CER: {cer_acc:.1f}% | QVV: {qvv_score:.1f}%")
        if g_note:
            print(f">> Gemma Judge: {g_note}")
        print("-" * 50)
        
    total_bench_time = time.time() - bench_start

    wer_list = [r[4] for r in results]
    cer_list = [r[5] for r in results]
    qvv_list = [r[6] for r in results]
    throughput = len(sentences) / total_bench_time if total_bench_time > 0 else 0.0

    print("\n================================================")
    print("FINAL RESULTS - DYNQ-VOICE:")
    print(f"Processed lines      : {len(sentences)}")
    print(f"Total STT time       : {total_bench_time:.2f}s ({throughput:.2f} sentences/sec)")
    print("------------------------------------------------")
    print(f"Average WER (Words)  : {sum(wer_list)/len(wer_list):.1f}%")
    print(f"Average CER (Chars)  : {sum(cer_list)/len(cer_list):.1f}%")
    print(f"Average QVV (Score)  : {sum(qvv_list)/len(qvv_list):.1f}%")
    print("================================================")
