# QVBench
import os
import time
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
import jiwer
import soundfile as sf
from kokoro_onnx import Kokoro
import ollama


try:
    from algorithm_QVV import compute_qvv_score as local_qvv_score
except ImportError:
    local_qvv_score = None

MODEL_SIZE = "small"
LANG_WHISPER = "en"
LANG_KOKORO = "en-us"
SPEED = "1.8"
PARALLEL_LINES = 2

KOKORO_MODEL = "kokoro-v1.0.onnx"
KOKORO_VOICE = "af_heart" 
KOKORO_VOICES_BIN = "voices-v1.0.bin"

DEVICE = "cuda"          
COMPUTE_TYPE = "float16"

QVV_API_ENDPOINT = "http://129.152.0.229:8000/evaluate"
QVV_API_KEY = "qvbench-public-test-veOkVKAg0KQo6aue1P"

OLLAMA_MODEL = "gemma4:e2b" 
ENABLE_GEMMA = True

print("[*] Initializing Kokoro v1.0 neural audio engine...")
kokoro_engine = Kokoro(KOKORO_MODEL, KOKORO_VOICES_BIN)

if ENABLE_GEMMA:
    try:
        print(f"[*] Pre-loading {OLLAMA_MODEL} into memory...")
        client_warmup = ollama.Client(timeout=35.0)
        client_warmup.chat(model=OLLAMA_MODEL.strip(), messages=[{'role': 'user', 'content': 'hi'}])
        print("[*] Semantic Judge ready.")
    except Exception as e:
        print(f"[!] Warning: Ollama warm-up skipped: {e}")


def get_qvv_score(ref: str, hyp: str) -> float:
    """Queries the remote QVV evaluation engine with local fallback."""
    try:
        response = requests.post(
            QVV_API_ENDPOINT,
            headers={"X-API-Key": QVV_API_KEY},
            json={"reference": ref, "hypothesis": hyp},
            timeout=4.0
        )
        if response.status_code == 200:
            return float(response.json().get("qvv_score", 0.0))
    except Exception:
        pass
    
    if local_qvv_score:
        return local_qvv_score(ref, hyp)
    return 0.0

def ask_gemma_judge(ref: str, hyp: str, score: float) -> str:
    """Executes forensic semantic evaluation using Gemma."""
    prompt = f"""You are a forensic computational linguist. Analyze this ASR transcription discrepancy.
Original Reference : "{ref}"
Model Hypothesis   : "{hyp}"

Respond in English in a single concise sentence: Does this discrepancy alter the essential business meaning, or is it fully understood by a native speaker?"""
    
    try:
        client = ollama.Client(timeout=35.0)
        response = client.chat(model=OLLAMA_MODEL.strip(), messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        return f"[Judge Evaluation Offline: {e}]"

def prepare_audio_file(args):
    idx, text_input, lang_k, speed = args
    raw_audio = f"frase_{idx}_norm.wav"
    fast_audio = f"frase_{idx}_fast.mp3"

    for f in (raw_audio, fast_audio):
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

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

def transcribe_worker(args):
    model, idx, ref_sentence, fast_audio, lang_w = args
    t0 = time.time()
    
    segments, _ = model.transcribe(fast_audio, beam_size=5, language=lang_w)
    live_hyp = "".join(s.text for s in segments).strip()
    latency = time.time() - t0

    wer_acc = max(0.0, 100.0 - (jiwer.wer(ref_sentence.lower(), live_hyp.lower()) * 100.0))
    cer_acc = max(0.0, 100.0 - (jiwer.cer(ref_sentence.lower(), live_hyp.lower()) * 100.0))
    
    qvv_score = get_qvv_score(ref_sentence, live_hyp)

    gemma_comment = ""
    if qvv_score < 100.0 and ENABLE_GEMMA:
        gemma_comment = ask_gemma_judge(ref_sentence, live_hyp, qvv_score)

    return idx, ref_sentence, live_hyp, latency, wer_acc, cer_acc, qvv_score, gemma_comment


if __name__ == "__main__":
    input_file = "frasi.txt"
    if not os.path.exists(input_file):
        print(f"[!] Error: Target corpus '{input_file}' not found.")
        exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"\n==================================================================")
    print(f"  DYNQ-VOICE BENCHMARK SUITE (Edge-to-Cloud Arch)")
    print(f"==================================================================")
    print(f"Target Sentences  : {len(sentences)}")
    print(f"Stream Velocity   : {SPEED}x")
    print(f"Hardware Engine   : GPU Accelerated (ROCm / High-Density float16)")
    print(f"Scoring Service   : Hybrid Cloud API + Secure Local Fallback")
    print(f"Semantic Evaluator: Google DeepMind Gemma ({OLLAMA_MODEL})")
    print(f"==================================================================\n")
    
    print("[1/2] Generating acoustic streams via Kokoro v1.0...")
    prep_tasks = [(idx, s, LANG_KOKORO, SPEED) for idx, s in enumerate(sentences, 1)]
    with ThreadPoolExecutor(max_workers=PARALLEL_LINES) as executor:
        prepared_audios = list(executor.map(prepare_audio_file, prep_tasks))
    print("--> Acoustic streams compiled successfully.\n")

    print(f"[2/2] Loading ASR Engine '{MODEL_SIZE}' on hardware accelerator...")
    try:
        stt_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"[!] Fallback to CPU int8 mode: {e}")
        stt_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    bench_start = time.time()
    stt_tasks = [(stt_model, idx, ref, audio, LANG_WHISPER) for idx, ref, audio in prepared_audios]

    with ThreadPoolExecutor(max_workers=PARALLEL_LINES) as executor:
        results = list(executor.map(transcribe_worker, stt_tasks))
        
    total_bench_time = time.time() - bench_start

    wer_list = [r[4] for r in results]
    cer_list = [r[5] for r in results]
    qvv_list = [r[6] for r in results]
    throughput = len(sentences) / total_bench_time if total_bench_time > 0 else 0.0

    for idx, ref_sentence, live_hyp, latency, wer_acc, cer_acc, qvv_score, g_note in results:
        print(f"\n[Stream {idx}] (ASR Latency: {latency:.2f}s)")
        print(f"Ref : {ref_sentence}")
        print(f"Hyp : {live_hyp}")
        print(f"Score -> WER: {wer_acc:.1f}% | CER: {cer_acc:.1f}% | QVV: {qvv_score:.1f}%")
        if g_note:
            print(f">> Gemma Judge: {g_note}")
        print("-" * 65)

    print("\n==================================================================")
    print("DYNQ-VOICE BENCHMARK REPORT")
    print(f"Processed Streams     : {len(sentences)} (Parallel Channels: {PARALLEL_LINES})")
    print(f"Total Execution Time  : {total_bench_time:.2f}s ({throughput:.2f} streams/sec)")
    print("------------------------------------------------------------------")
    print(f"Average WER Accuracy  : {sum(wer_list)/len(wer_list):.1f}%")
    print(f"Average CER Accuracy  : {sum(cer_list)/len(cer_list):.1f}%")
    print(f"Average QVV Accuracy  : {sum(qvv_list)/len(qvv_list):.1f}%")
    print("==================================================================")
