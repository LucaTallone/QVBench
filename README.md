# QVBench
Slide 1: Cover & Project Identity
    • Title: DynQ-Voice (QVBench)
    • Subtitle: High-Throughput Acoustic Pipeline & Semantic Evaluation Suite on AMD ROCm
    • Team: NordOvest AI (Luca Tallone)
    • Key Concept: Solving the Real-World Speech-to-Text Bottleneck: Moving Beyond Word Error Rate (WER) with QVV & Gemma 4

      
Slide 2: The Industrial Problem (Why Traditional Benchmarks Fail)
    • Single-Thread Slowness: Traditional ASR evaluation processes audio files sequentially. Testing enterprise datasets takes days.
    • Lack of Acoustic Stress Testing: Models are tested on slow studio speech, failing in fast-paced real-world telephony (1.8x velocity).
    • The Legacy Metric Flaw:
        ◦ WER (Word Error Rate) is purely syntactic: it treats minor formatting (e.g., numbers "four" vs "4", currency, missing commas) as catastrophic errors.
        ◦ CER (Character Error Rate) is deceptively optimistic: it gives high scores to hallucinated words if enough characters match.

Exl.1

Slide 3: The Architecture & Proprietary QVV Metric
    • Kokoro v1.0 Neural Synthesis: Ultra-lightweight local acoustic generation running on AMD/ROCm hardware.
    • Accelerated Parallel Streams: Real-time 1.8x atempo compression and multi-channel injection directly into CUDA/ROCm Faster-Whisper.
    • Proprietary QVV Engine:
        ◦ Number & entity normalization (automatic equivalence of digits and spoken words).
        ◦ Independent punctuation channels (commas do not invalidate adjacent business tokens).
        ◦ Heavy penalties for semantic hallucinations while tolerating phonetic liaison.
(Inserisci qui lo screenshot del terminale con la tabella finale: 96.0% QVV)

Slide 4: LLM-as-a-Judge & Enterprise Scalability
    • Google DeepMind Gemma Integration: Gemma 4 acts as a forensic semantic arbiter for disputed lines (QVV < 100%), confirming whether business meaning is preserved.
    • Multi-Tier Edge-to-Cloud Serving:
        ◦ High-density local execution via AMD ROCm.
        ◦ REST API cloud fallback for corporate scalability.
    • Cost Efficiency: Slashes third-party telephony API costs by up to 90% while guaranteeing enterprise verification.

Support: Gemini 3,1 pro for SOTA intelligence study, Geminu 3,8 flash for compatibility/AMD adapt,
open libraries we used for the project:
faster-whisper
jiwer
rapidfuzz
num2words
soundfile
kokoro-onnx
Ollama
Gemma 4
