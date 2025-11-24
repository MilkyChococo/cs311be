from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from datetime import datetime
from typing import Optional, List
from collections import Counter

import torch
import whisper
import numpy as np
import soundfile as sf
import librosa

# ==== EMOTION IMPORTS ====
import cv2
from deepface import DeepFace

# ================== CẤU HÌNH WHISPER ==================

MODEL_NAME = "large-v3"      # hoặc "large", "medium" nếu muốn nhẹ hơn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Whisper model: {MODEL_NAME} on {DEVICE} ...")
whisper_model = whisper.load_model(MODEL_NAME, device=DEVICE)


def load_audio_no_ffmpeg(path, target_sr=16000):
    """
    Đọc file WAV trực tiếp, không thông qua ffmpeg.
    - Đảm bảo mono
    - Resample về 16kHz
    - Trả về np.float32
    """
    audio, sr = sf.read(path)  # audio: (n,) hoặc (n, channels)

    # nếu stereo → lấy trung bình hai kênh thành mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # resample về target_sr nếu khác
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    audio = audio.astype(np.float32)
    return audio, target_sr


def transcribe_with_whisper(audio_path, language="en"):
    """
    Chạy Whisper để lấy segments + word-level timestamps.
    Dùng global whisper_model đã load sẵn.
    """
    audio, sr = load_audio_no_ffmpeg(audio_path, target_sr=16000)

    result = whisper_model.transcribe(
        audio,
        word_timestamps=True,   # bật word-level timestamps
        verbose=False,
        language=language,      # tiếng Anh
        task="transcribe",
    )
    return result


def extract_words(result):
    """
    Từ output của Whisper, flatten thành list words:
    mỗi phần tử: {"word": str, "start": float, "end": float}
    """
    words = []
    for seg in result["segments"]:
        # Mỗi segment có thể có trường "words"
        if "words" in seg and seg["words"] is not None:
            for w in seg["words"]:
                # w có dạng {"word": " Hello", "start": 0.0, "end": 0.5, ...}
                words.append({
                    "word": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"])
                })
        else:
            # Fallback: nếu model không trả word-level,
            # dùng nguyên segment (ít chính xác hơn)
            words.append({
                "word": seg["text"],
                "start": float(seg["start"]),
                "end": float(seg["end"])
            })

    # sort theo thời gian phòng trường hợp lộn xộn
    words.sort(key=lambda x: x["start"])
    return words


def build_text_with_pauses(words, pause_threshold=0.5, round_ndigits=2):
    """
    Nhận list words + timestamps, chèn [PAUSE x.xx s]
    mỗi khi khoảng cách giữa hai từ liên tiếp >= pause_threshold.
    Ví dụ: Hello, [PAUSE 0.55s] my name is Fu.
    """
    if not words:
        return ""

    parts = []
    prev_end = words[0]["end"]
    parts.append(words[0]["word"])

    for i in range(1, len(words)):
        w = words[i]
        gap = w["start"] - prev_end

        # Nếu khoảng im lặng đủ lớn → chèn PAUSE
        if gap >= pause_threshold:
            pause_sec = round(gap, round_ndigits)
            parts.append(f"[PAUSE {pause_sec:.2f}s]")

        parts.append(w["word"])
        prev_end = w["end"]

    # ghép lại, rồi xử lý khoảng trắng
    text = " ".join(parts)
    # Whisper hay cho space trước dấu câu → chỉnh lại
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    text = text.replace(" ’", "’")
    return text.strip()


def group_into_segments(words, pause_threshold=0.8):
    """
    Chia thành các ĐOẠN NÓI (chunks) cách nhau bởi khoảng im lặng dài (pause_threshold).
    Trả về:
      - segments: list dict {"text", "start", "end"}
      - pauses: list pause_seconds giữa các segment
    """
    if not words:
        return [], []

    segments = []
    pauses = []

    current_words = [words[0]["word"]]
    current_start = words[0]["start"]
    prev_end = words[0]["end"]

    for i in range(1, len(words)):
        w = words[i]
        gap = w["start"] - prev_end

        if gap >= pause_threshold:
            # kết thúc segment hiện tại
            segment_text = " ".join(current_words)
            segments.append({
                "text": segment_text.strip(),
                "start": current_start,
                "end": prev_end
            })
            pauses.append(gap)

            # bắt đầu segment mới
            current_words = [w["word"]]
            current_start = w["start"]
        else:
            current_words.append(w["word"])

        prev_end = w["end"]

    # thêm segment cuối
    if current_words:
        segment_text = " ".join(current_words)
        segments.append({
            "text": segment_text.strip(),
            "start": current_start,
            "end": prev_end
        })

    return segments, pauses


def format_segments_with_pauses(segments, pauses, round_ndigits=2):
    """
    Định dạng kiểu: Đoạn nói [PAUSE x.xx s] Đoạn nói ...
    """
    parts = []
    for i, seg in enumerate(segments):
        seg_text = seg["text"]
        parts.append(seg_text)
        if i < len(pauses):
            pause_sec = round(pauses[i], round_ndigits)
            parts.append(f"[PAUSE {pause_sec:.2f}s]")
    return " ".join(parts).strip()


# ================== EMOTION MODEL (DeepFace) ==================

print("Loading Emotion model (DeepFace)...")
emotion_labels = ['angry','disgust','fear','happy','sad','surprise','neutral']
emo_model = DeepFace.build_model("Emotion", "facial_attribute")
inner_model = getattr(emo_model, "model", None)
if inner_model is None:
    raise RuntimeError("Không lấy được inner Keras model từ Emotion model")

# Haar cascade để detect face
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def predict_emotion_from_bgr(face_bgr: np.ndarray):
    """
    Nhận diện emotion từ ảnh khuôn mặt (BGR), trả về (label, probs).
    """
    if face_bgr is None or face_bgr.size == 0:
        return None, None

    face_bgr = np.ascontiguousarray(face_bgr)
    g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (48, 48), interpolation=cv2.INTER_AREA)
    g = g.astype("float32") / 255.0
    g = g.reshape(1, 48, 48, 1)

    probs = inner_model.predict(g, verbose=0)[0]  # (7,)
    idx = int(np.argmax(probs))
    return emotion_labels[idx], probs


def analyze_video_emotions(
    video_path: str,
    segments: List[dict],
    sample_every_n_frames: int = 3
):
    """
    Phân tích emotion từ video:
      - Mở video
      - Lặp qua frame, lấy timestamp t = frame_idx / fps
      - Đưa frame vào segment có start <= t <= end
      - Detect mặt + emotion, append vào emotions_by_segment[segment_idx]
    Trả về: list các list nhãn emotion theo từng segment.
    """
    if not segments:
        return []

    cap = cv2.VideoCapture(video_path)


    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # fallback

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    emotions_by_segment: List[List[str]] = [[] for _ in segments]

    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps  # thời gian (giây) của frame

        # tìm segment chứa thời điểm t
        seg_idx = None
        for i, seg in enumerate(segments):
            if seg["start"] <= t <= seg["end"]:
                seg_idx = i
                break

        if seg_idx is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                if w >= 20 and h >= 20:
                    face_bgr = frame[y:y+h, x:x+w]
                    emo_label, _ = predict_emotion_from_bgr(face_bgr)
                    if emo_label is not None:
                        emotions_by_segment[seg_idx].append(emo_label)

        frame_idx += sample_every_n_frames

    cap.release()
    return emotions_by_segment


# ================== FASTAPI APP ==================

app = FastAPI()

origins = [
    "http://localhost:3001",
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # hoặc ["*"] khi dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RECORD_DIR = "recordings"
OUTPUT_DIR = "transcripts"
os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== (1) /chat/chatDomain ======

class ChatRequest(BaseModel):
    room_id: str
    query: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat/chatDomain", response_model=ChatResponse)
async def chat_domain(req: ChatRequest):
    reply = f"You said: {req.query}"
    return ChatResponse(response=reply)


# ====== (2) /upload/audio (+ optional video + emotion) ======

@app.post("/upload/audio")
async def upload_audio(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    video_file: Optional[UploadFile] = File(None)
):
    """
    Nhận file audio (WAV) từ frontend, lưu vào server,
    chạy Whisper + đánh dấu [PAUSE Xs],
    NẾU có video_file thì phân tích thêm emotion theo từng segment
    và lưu ra file txt.
    """
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"interview-{session_id}-{time_str}"

    # 1) Lưu file audio (giả sử frontend gửi .wav)
    wav_path = os.path.join(RECORD_DIR, base_name + ".wav")
    with open(wav_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2) Whisper: nhận dạng + timestamps
    result = transcribe_with_whisper(wav_path, language="en")

    # raw text từ Whisper
    raw_text = (result.get("text") or "").strip()

    # list word-level (cho PAUSE chi tiết)
    words = extract_words(result)

    # 3A) Annotated text chi tiết: chèn [PAUSE] giữa các cụm từ
    word_level_annotated = build_text_with_pauses(
        words,
        pause_threshold=0.5  # ví dụ: >= 0.5s thì coi là pause
    )

    # 3B) Chia thành ĐOẠN NÓI (câu) + [PAUSE] giữa các đoạn
    segments, pauses = group_into_segments(
        words,
        pause_threshold=0.8  # pause dài hơn → ngắt thành đoạn
    )
    segment_level_annotated = format_segments_with_pauses(segments, pauses)

    # 4) Lưu transcript annotated (ở đây mình lưu bản segment-level)
    txt_name = f"{base_name}.txt"
    out_path = os.path.join(OUTPUT_DIR, txt_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(segment_level_annotated)

    # 5) Nếu có video_file → phân tích emotion, lưu riêng ra txt
    emotions_by_segment = None
    emotion_txt_path = None

    if video_file is not None:
        video_ext = os.path.splitext(video_file.filename or "")[1] or ".webm"
        video_path = os.path.join(RECORD_DIR, base_name + video_ext)
        with open(video_path, "wb") as vf:
            shutil.copyfileobj(video_file.file, vf)

        # phân tích emotion theo segment
        try:
            emotions_by_segment = analyze_video_emotions(video_path, segments)
        except Exception as e:
            print(f"[ERROR] analyze_video_emotions: {e}")
            emotions_by_segment = [[] for _ in segments]

        # lưu emotion ra txt
        emo_txt_name = f"{base_name}_emotion.txt"
        emotion_txt_path = os.path.join(OUTPUT_DIR, emo_txt_name)
        with open(emotion_txt_path, "w", encoding="utf-8") as ef:
            for i, seg in enumerate(segments):
                ef.write(f"Segment {i+1}: {seg['start']:.2f}-{seg['end']:.2f}s\n")
                ef.write(f"Text: {seg['text']}\n")
                if emotions_by_segment and emotions_by_segment[i]:
                    labels = emotions_by_segment[i]
                    counts = Counter(labels)
                    majority, cnt = counts.most_common(1)[0]
                    ef.write(f"Emotions: {', '.join(labels)}\n")
                    ef.write(f"Majority: {majority} (x{cnt})\n")
                else:
                    ef.write("Emotions: (no face detected)\n")
                ef.write("\n")

    return {
        "status": "ok",
        "session_id": session_id,
        "audio_path": wav_path,
        "transcript_path": out_path,
        "raw_text": raw_text,
        "word_level_annotated": word_level_annotated,       # Hello [PAUSE...] my name...
        "segment_level_annotated": segment_level_annotated, # Đoạn nói [PAUSE...] Đoạn nói...
        "segments": segments,   # list {text, start, end}
        "pauses": pauses,       # list float (giây)
        "emotions_by_segment": emotions_by_segment,
        "emotion_txt_path": emotion_txt_path,
    }
