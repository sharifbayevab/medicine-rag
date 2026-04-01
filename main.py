import os
import re
import json
import base64
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv

from agent.speech_to_text import YandexSpeechRecognizer, SttStreamingSession
from agent.text_to_speech import YandexStreamingSynthesizer, TtsStreamingSession
from agent.llm import OpenAIClient
from agent.face_encoder import FaceEncoder
from agent.face_store import FaceVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

FOLDER_ID = os.getenv("YANDEX_CATALOG_ID")
API_KEY = os.getenv("YANDEX_API_KEY")
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.62"))
FACE_LOG_ALL_COMPARISONS = os.getenv("FACE_LOG_ALL_COMPARISONS", "false").lower() == "true"

if not API_KEY or not FOLDER_ID:
    logger.error("YANDEX_API_KEY or YANDEX_CATALOG_ID is not set in .env")

app = FastAPI()

os.makedirs("static", exist_ok=True)
os.makedirs("images", exist_ok=True)
REGISTER_FACES_DIR = Path("data/register_faces")
REGISTER_FACES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

IMAGES_DIR = Path("images")
REGISTER_FACES_JSON = REGISTER_FACES_DIR / "registry.json"
face_store = FaceVectorStore()
face_encoder = None


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/")
async def get():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/faces")
async def save_face_snapshot(payload: dict):
    image_data = payload.get("image")
    if not image_data:
        return JSONResponse({"ok": False, "error": "image is required"}, status_code=400)

    try:
        filename = save_base64_image(image_data)
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid image"}, status_code=400)
    return JSONResponse({"ok": True, "path": filename})


def flush_sentence_buffer(buf: str) -> tuple[str | None, str]:
    m = re.search(r'[.!?…\n]\s', buf)
    if m:
        split_pos = m.end()
        return buf[:split_pos].strip(), buf[split_pos:]
    if len(buf) > 150:
        last_space = buf.rfind(' ', 0, 150)
        if last_space > 30:
            return buf[:last_space].strip(), buf[last_space:]
        return buf.strip(), ""
    return None, buf


def save_base64_image(image_data: str) -> str:
    if "," in image_data:
        _, image_data = image_data.split(",", 1)
    image_bytes = base64.b64decode(image_data)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = IMAGES_DIR / f"face_{timestamp}.jpg"
    filename.write_bytes(image_bytes)
    return str(filename)


def build_system_prompt(
    current_person: dict | None,
    onboarding: bool,
    current_lang: str,
) -> str:
    if current_lang == "uz-UZ":
        language_instruction = (
            "Speak only in Uzbek using Latin script. Keep it natural, simple, spoken, and respectful. Do not switch languages unless the visitor asks."
        )
    elif current_lang == "en-US":
        language_instruction = (
            "Speak only in English. Keep it natural, simple, spoken, and respectful. Do not switch languages unless the visitor asks."
        )
    else:
        language_instruction = (
            "Speak only in Russian. Keep it natural, simple, spoken, and respectful. Do not switch languages unless the visitor asks."
        )
    base = (
        "You are Medical Assistant"
        "You were prepared as part of the grant project 'Yuqumli kasalliklar shifoxonasi uchun aqlli robot yaratish', "
        "led by Azimov Bunyod Raximjonovich at Tashkent University of Information Technologies. "
        "Right now you are standing at the Oliygoh kubogi taqdirlash marosimi. "
        "Introduce yourself as a medical reception agent when it fits naturally. "
        "Keep every reply very short, lively, friendly, and clear so you do not make people sleepy. "
        "Use a spoken style with light humor and positive energy. "
        "Usually stay within 1 short sentences unless a longer answer is truly necessary. "
        "If you ask a question, ask only one simple question at a time. "
        "Always address the visitor formally and respectfully. "
        "In Russian, always use 'Вы'. In Uzbek, use a respectful formal style. In English, use polite formal phrasing. "
        f"{language_instruction}"
    )
    if onboarding:
        return (
            base
            + " You are meeting a new person. Briefly introduce yourself and ask for their first name and last name."
        )
    if current_person:
        full_name = f'{current_person.get("first_name", "")} {current_person.get("last_name", "")}'.strip()
        metadata = current_person.get("metadata") or {}
        metadata_context = ""
        if metadata:
            metadata_context = f" Extra known details about this person: {json.dumps(metadata, ensure_ascii=False)}."
        return (
            base
            + f" You already know this person: {full_name}. You may use their name in the first greeting, but after that speak naturally and do not repeat their name in every reply."
            + metadata_context
        )
    return base


def get_face_encoder() -> FaceEncoder:
    global face_encoder
    if face_encoder is None:
        face_encoder = FaceEncoder()
    return face_encoder


def identify_face_embedding(embedding: list[float], snapshot_path: str | None = None) -> dict:
    match = face_store.identify(embedding, threshold=FACE_MATCH_THRESHOLD)
    if match:
        comparisons = match.get("comparisons", [])
        comparisons_to_log = comparisons if FACE_LOG_ALL_COMPARISONS else comparisons[:3]
        for comparison in comparisons_to_log:
            logger.info(
                "Face compare item: threshold=%.2f candidate_id=%s full_name=%s score=%.4f",
                FACE_MATCH_THRESHOLD,
                comparison.get("person_id"),
                comparison.get("full_name"),
                comparison.get("score", 0.0),
            )
        logger.info(
            "Face compare best: threshold=%.2f score=%.4f matched=%s person_id=%s full_name=%s snapshot=%s candidates=%s logged=%s all_logging=%s",
            FACE_MATCH_THRESHOLD,
            match.get("score", 0.0),
            match.get("matched", False),
            match.get("person_id"),
            match.get("full_name"),
            snapshot_path or "",
            len(comparisons),
            len(comparisons_to_log),
            FACE_LOG_ALL_COMPARISONS,
        )

    if match and match.get("matched"):
        if snapshot_path:
            face_store.add_snapshot(match["person_id"], snapshot_path)
        return {
            "status": "known",
            "snapshot_path": snapshot_path,
            "person": match,
        }

    return {
        "status": "unknown",
        "snapshot_path": snapshot_path,
        "embedding": embedding,
    }


def load_registered_faces() -> tuple[int, int]:
    if not REGISTER_FACES_JSON.exists():
        logger.info("Register faces bootstrap skipped: %s not found", REGISTER_FACES_JSON)
        return 0, 0

    try:
        data = json.loads(REGISTER_FACES_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to read %s: %s", REGISTER_FACES_JSON, exc)
        return 0, 0

    items = data.get("faces", []) if isinstance(data, dict) else []
    loaded = 0
    skipped = 0

    for item in items:
        file_name = item.get("file")
        person_id = item.get("person_id")
        first_name = item.get("first_name", "")
        last_name = item.get("last_name", "")
        metadata = item.get("metadata", {})

        if not file_name or not first_name:
            skipped += 1
            continue

        image_path = (REGISTER_FACES_DIR / file_name).resolve()
        if not image_path.exists():
            skipped += 1
            continue

        snapshot_path = str(image_path)
        existing = face_store.get_person(person_id) if person_id else None
        if existing:
            if snapshot_path:
                face_store.add_snapshot(person_id, snapshot_path)
            loaded += 1
            continue

        try:
            embedding = get_face_encoder().extract_embedding_from_path(snapshot_path)
            face_store.register(
                embedding=embedding,
                first_name=first_name,
                last_name=last_name,
                snapshot_path=snapshot_path,
                metadata=metadata,
                person_id=person_id,
            )
            logger.info(
                "Bootstrapped face: file=%s person_id=%s full_name=%s %s threshold=%.2f",
                file_name,
                person_id,
                f"{first_name} {last_name}".strip(),
                f"metadata={json.dumps(metadata, ensure_ascii=False)}" if metadata else "metadata={}",
                FACE_MATCH_THRESHOLD,
            )
            loaded += 1
        except Exception as exc:
            logger.error("Failed to bootstrap face %s: %s", file_name, exc)
            skipped += 1

    logger.info("Register faces bootstrap finished: loaded=%s skipped=%s", loaded, skipped)
    return loaded, skipped


@app.on_event("startup")
async def startup_bootstrap_faces():
    load_registered_faces()


@app.post("/api/faces/identify")
async def identify_faces(payload: dict):
    faces = payload.get("faces", [])
    results = []

    for face in faces:
        image_data = face.get("image")
        if not image_data:
            continue
        try:
            snapshot_path = save_base64_image(image_data)
            embedding = get_face_encoder().extract_embedding_from_base64(image_data)
        except Exception:
            continue

        results.append(identify_face_embedding(embedding, snapshot_path))

    return JSONResponse({"faces": results})


@app.post("/api/frame/analyze")
async def analyze_frame(payload: dict):
    frame = payload.get("frame")
    max_faces = int(payload.get("max_faces", 5))
    if not frame:
        return JSONResponse({"faces": []})

    try:
        detected_faces = get_face_encoder().analyze_frame_base64(frame, max_faces=max_faces)
    except Exception as exc:
        logger.error("Failed to analyze frame: %s", exc)
        return JSONResponse({"faces": []})

    results = []
    for face in detected_faces:
        crop_base64 = face.get("crop_base64")
        if not crop_base64:
            continue

        identified = identify_face_embedding(face["embedding"])
        if identified["status"] == "unknown":
            try:
                snapshot_path = save_base64_image(crop_base64)
            except Exception:
                continue
            identified["snapshot_path"] = snapshot_path
            identified["embedding"] = face["embedding"]

        identified["box"] = face["box"]
        identified["det_score"] = face.get("det_score", 0.0)
        results.append(identified)

    return JSONResponse({"faces": results})


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    recognizer = YandexSpeechRecognizer(folder_id=FOLDER_ID, iam_token=API_KEY)
    synthesizer = YandexStreamingSynthesizer(folder_id=FOLDER_ID, iam_token=API_KEY)
    llm = OpenAIClient()

    loop = asyncio.get_running_loop()

    current_lang = "uz-UZ"
    current_voice = "yulduz"

    stt_session = None
    partial_stt_task = None

    # Настройки
    live_mode = False
    allow_interrupt = False

    # Состояние
    is_responding = False
    response_cancelled = asyncio.Event()
    # Ссылка на текущую TTS сессию для принудительной остановки
    active_tts_session = None
    active_response_task = None
    response_generation = 0
    current_person = None
    pending_registration = None
    last_face_person_id = None
    last_face_snapshot_path = None

    def start_stt():
        nonlocal stt_session, partial_stt_task
        if stt_session:
            try:
                stt_session._chunks.put(None)
            except Exception:
                pass
            stt_session = None

        if partial_stt_task:
            partial_stt_task.cancel()
            partial_stt_task = None

        stt_partial_queue = asyncio.Queue()
        stt_session = SttStreamingSession(recognizer, 16000, loop, stt_partial_queue, current_lang)
        stt_session.start()
        logger.info("STT session started")

        async def send_partial_stt():
            try:
                while True:
                    text = await stt_partial_queue.get()
                    if text is None:
                        break
                    await websocket.send_json({"type": "stt_partial", "text": text})
            except (asyncio.CancelledError, Exception):
                pass

        partial_stt_task = asyncio.create_task(send_partial_stt())

    def stop_stt():
        nonlocal stt_session, partial_stt_task
        if stt_session:
            try:
                stt_session._chunks.put(None)
            except Exception:
                pass
            stt_session = None
        if partial_stt_task:
            partial_stt_task.cancel()
            partial_stt_task = None

    def force_cancel_response():
        """Немедленно останавливает текущий ответ: LLM + TTS."""
        nonlocal is_responding, active_tts_session, active_response_task, response_generation
        response_generation += 1
        response_cancelled.set()
        if active_tts_session is not None:
            active_tts_session.cancel()
            active_tts_session = None
        if active_response_task and not active_response_task.done():
            active_response_task.cancel()
            active_response_task = None
        is_responding = False
        logger.info("Response force-cancelled")

    def refresh_system_prompt(clear_history: bool = False):
        nonlocal messages
        system_message = {
            "role": "system",
            "content": build_system_prompt(
                current_person,
                onboarding=bool(pending_registration),
                current_lang=current_lang,
            ),
        }

        if clear_history or not messages:
            messages = [system_message]
            return

        if messages[0].get("role") == "system":
            messages[0] = system_message
        else:
            messages.insert(0, system_message)

    messages = []
    refresh_system_prompt(clear_history=True)

    async def process_response(user_text: str, generation: int):
        nonlocal is_responding, active_tts_session, active_response_task
        is_responding = True
        response_cancelled.clear()
        refresh_system_prompt(clear_history=False)

        tts_audio_queue = asyncio.Queue()
        tts_session = TtsStreamingSession(
            synthesizer, 1.1, 48000, loop, tts_audio_queue, current_voice
        )
        active_tts_session = tts_session
        tts_session.start()

        await websocket.send_json({"type": "response_started", "response_id": generation})

        async def send_tts_audio():
            try:
                while not response_cancelled.is_set() and generation == response_generation:
                    try:
                        audio_chunk = await asyncio.wait_for(tts_audio_queue.get(), timeout=0.02)
                    except asyncio.TimeoutError:
                        continue
                    if audio_chunk is None:
                        break
                    if response_cancelled.is_set() or generation != response_generation:
                        break
                    await websocket.send_json(
                        {
                            "type": "tts_chunk",
                            "response_id": generation,
                            "audio": base64.b64encode(audio_chunk).decode("ascii"),
                        }
                    )
                # Drain remaining chunks from queue (don't send them)
                while not tts_audio_queue.empty():
                    try:
                        tts_audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                if generation == response_generation and not response_cancelled.is_set():
                    await websocket.send_json({"type": "tts_end", "response_id": generation})
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error sending TTS audio: {e}")

        tts_task = asyncio.create_task(send_tts_audio())

        full_llm_response = ""
        sentence_buf = ""

        try:
            async for llm_chunk in llm.get_response_stream(messages):
                if response_cancelled.is_set() or generation != response_generation:
                    break
                full_llm_response += llm_chunk
                await websocket.send_json(
                    {"type": "llm_partial", "text": llm_chunk, "response_id": generation}
                )

                sentence_buf += llm_chunk
                sentence, sentence_buf = flush_sentence_buffer(sentence_buf)
                if sentence:
                    tts_session.feed(sentence)

            if (
                sentence_buf.strip()
                and not response_cancelled.is_set()
                and generation == response_generation
            ):
                tts_session.feed(sentence_buf.strip())
        except asyncio.CancelledError:
            logger.info("Response task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error during LLM streaming: {e}")

        # Завершаем TTS (если ещё не принудительно остановлен)
        try:
            await tts_session.finish()
        except (asyncio.CancelledError, Exception):
            pass

        was_cancelled = response_cancelled.is_set() or generation != response_generation

        if full_llm_response and not was_cancelled:
            messages.append({"role": "assistant", "content": full_llm_response})

        try:
            await tts_task
        except asyncio.CancelledError:
            pass

        if active_tts_session is tts_session:
            active_tts_session = None
        if active_response_task is asyncio.current_task():
            active_response_task = None
        if generation == response_generation:
            is_responding = False
        logger.info(f"Response completed (cancelled={was_cancelled})")

        # В живом режиме — снова слушаем, только если не прервано
        # (при прерывании фронт уже сам начал слушать)
        if live_mode and not was_cancelled:
            await websocket.send_json({"type": "ready_to_listen"})

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                # Аудио от микрофона
                # Barge-in: аудио пришло пока агент отвечает
                if is_responding and allow_interrupt:
                    logger.info("Barge-in: audio received during response")
                    force_cancel_response()
                    await websocket.send_json({"type": "interrupt"})
                    start_stt()

                if stt_session:
                    stt_session.feed(data["bytes"])

            elif "text" in data:
                msg = json.loads(data["text"])
                msg_type = msg.get("type")

                if msg_type == "set_language":
                    current_lang = msg.get("lang", "uz-UZ")
                    if current_lang == "uz-UZ":
                        current_voice = "yulduz"
                    elif current_lang == "en-US":
                        current_voice = "john"
                    else:
                        current_voice = "yulduz_ru"
                    refresh_system_prompt(clear_history=False)
                    logger.info(f"Language set to {current_lang}, voice {current_voice}")

                elif msg_type == "set_settings":
                    live_mode = msg.get("live_mode", False)
                    allow_interrupt = msg.get("allow_interrupt", False)
                    logger.info(f"Settings: live_mode={live_mode}, allow_interrupt={allow_interrupt}")

                elif msg_type == "start_speech":
                    logger.info("start_speech received")
                    # Если агент ещё отвечает и разрешено прерывать — прерываем
                    if is_responding and allow_interrupt:
                        force_cancel_response()
                        await websocket.send_json({"type": "interrupt"})
                    start_stt()

                elif msg_type == "interrupt":
                    if is_responding:
                        logger.info("Explicit interrupt")
                        force_cancel_response()
                        await websocket.send_json({"type": "interrupt"})

                elif msg_type == "face_identity":
                    incoming_person = msg.get("person")
                    incoming_pending = msg.get("pending_registration")
                    incoming_person_id = incoming_person.get("person_id") if incoming_person else None
                    incoming_snapshot = incoming_pending.get("snapshot_path") if incoming_pending else None

                    face_changed = (
                        incoming_person_id != last_face_person_id
                        or incoming_snapshot != last_face_snapshot_path
                        or bool(incoming_pending) != bool(pending_registration)
                    )

                    current_person = incoming_person
                    pending_registration = incoming_pending
                    last_face_person_id = incoming_person_id
                    last_face_snapshot_path = incoming_snapshot
                    refresh_system_prompt(clear_history=True)
                    logger.info(
                        "Face identity received: person=%s pending=%s",
                        current_person.get("full_name") if current_person else None,
                        bool(pending_registration),
                    )

                    if face_changed and is_responding:
                        force_cancel_response()
                        await websocket.send_json({"type": "interrupt"})

                    if face_changed and not is_responding:
                        if current_person:
                            greeting_prompt = (
                                f"{current_person.get('first_name', '').strip()} is in front of you. "
                                "Greet this person as someone you already know and briefly offer help."
                            )
                        elif pending_registration:
                            greeting_prompt = (
                                "A new person is in front of you. Greet them and briefly ask for their first name and last name."
                            )
                        else:
                            greeting_prompt = None

                        if greeting_prompt:
                            messages.append({"role": "user", "content": greeting_prompt})
                            response_generation += 1
                            active_response_task = asyncio.create_task(
                                process_response(greeting_prompt, response_generation)
                            )

                elif msg_type == "face_missing":
                    current_person = None
                    pending_registration = None
                    last_face_person_id = None
                    last_face_snapshot_path = None
                    refresh_system_prompt(clear_history=True)
                    logger.info("Face missing received: identity cleared and prompt reset")

                elif msg_type == "end_speech":
                    if not stt_session:
                        continue
                    final_text = await stt_session.finish()
                    stt_session = None
                    logger.info(f"Final STT: '{final_text}'")

                    if not final_text:
                        if live_mode:
                            await websocket.send_json({"type": "ready_to_listen"})
                        continue

                    if pending_registration:
                        extracted = await llm.extract_person_name(final_text)
                        if extracted and extracted.get("is_confident"):
                            current_person = face_store.register(
                                embedding=pending_registration["embedding"],
                                first_name=extracted.get("first_name", ""),
                                last_name=extracted.get("last_name", ""),
                                snapshot_path=pending_registration.get("snapshot_path"),
                                metadata={},
                            )
                            pending_registration = None
                            last_face_person_id = current_person["person_id"]
                            last_face_snapshot_path = None
                            refresh_system_prompt(clear_history=True)

                            await websocket.send_json({"type": "stt_final", "text": final_text})
                            messages.append({"role": "user", "content": final_text})

                            registration_prompt = (
                                f"Теперь ты знаешь человека: {current_person['full_name']}. "
                                "Коротко поприветствуй его по имени и скажи, что рад знакомству."
                            )
                            messages.append({"role": "user", "content": registration_prompt})
                            response_generation += 1
                            active_response_task = asyncio.create_task(
                                process_response(registration_prompt, response_generation)
                            )
                            continue

                    # Отменить предыдущий ответ если ещё идёт
                    if is_responding:
                        force_cancel_response()

                    await websocket.send_json({"type": "stt_final", "text": final_text})
                    messages.append({"role": "user", "content": final_text})
                    response_generation += 1
                    active_response_task = asyncio.create_task(
                        process_response(final_text, response_generation)
                    )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except RuntimeError as e:
        if "Cannot call" in str(e) and "receive" in str(e):
            logger.info("WebSocket connection closed by client")
        else:
            logger.error(f"RuntimeError in websocket: {e}")
    except Exception as e:
        logger.error(f"Error in websocket: {e}")
    finally:
        stop_stt()
        if is_responding or active_response_task:
            force_cancel_response()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
