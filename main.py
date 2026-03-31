import os
import re
import json
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv

from agent.speech_to_text import YandexSpeechRecognizer, SttStreamingSession
from agent.text_to_speech import YandexStreamingSynthesizer, TtsStreamingSession
from agent.llm import OpenAIClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

FOLDER_ID = os.getenv("YANDEX_CATALOG_ID")
API_KEY = os.getenv("YANDEX_API_KEY")

if not API_KEY or not FOLDER_ID:
    logger.error("YANDEX_API_KEY or YANDEX_CATALOG_ID is not set in .env")

app = FastAPI()

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/")
async def get():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


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


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    recognizer = YandexSpeechRecognizer(folder_id=FOLDER_ID, iam_token=API_KEY)
    synthesizer = YandexStreamingSynthesizer(folder_id=FOLDER_ID, iam_token=API_KEY)
    llm = OpenAIClient()

    loop = asyncio.get_running_loop()

    current_lang = "ru-RU"
    current_voice = "anton"

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
        nonlocal is_responding
        response_cancelled.set()
        if active_tts_session is not None:
            # Завершаем gRPC TTS поток (генератор перестаёт yield'ить)
            try:
                active_tts_session._text_queue.put(None)
            except Exception:
                pass
            # Отменяем asyncio task (gRPC thread wrapper)
            if active_tts_session._task and not active_tts_session._task.done():
                active_tts_session._task.cancel()
        is_responding = False
        logger.info("Response force-cancelled")

    messages = [
        {
            "role": "system",
            "content": (
                "Вы — медицинский гид (Medik Git). Помогаете пользователям с медицинскими вопросами. "
                "Отвечайте кратко (2-4 предложения) и профессионально на языке пользователя (русский или узбекский). "
                "Используйте разговорный стиль, как будто говорите вживую."
            ),
        }
    ]

    async def process_response(user_text: str):
        nonlocal is_responding, active_tts_session
        is_responding = True
        response_cancelled.clear()

        tts_audio_queue = asyncio.Queue()
        tts_session = TtsStreamingSession(
            synthesizer, 1.1, 48000, loop, tts_audio_queue, current_voice
        )
        active_tts_session = tts_session
        tts_session.start()

        async def send_tts_audio():
            try:
                while not response_cancelled.is_set():
                    try:
                        audio_chunk = await asyncio.wait_for(tts_audio_queue.get(), timeout=0.02)
                    except asyncio.TimeoutError:
                        continue
                    if audio_chunk is None:
                        break
                    if response_cancelled.is_set():
                        break
                    await websocket.send_bytes(audio_chunk)
                # Drain remaining chunks from queue (don't send them)
                while not tts_audio_queue.empty():
                    try:
                        tts_audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                await websocket.send_json({"type": "tts_end"})
            except Exception as e:
                logger.error(f"Error sending TTS audio: {e}")

        tts_task = asyncio.create_task(send_tts_audio())

        full_llm_response = ""
        sentence_buf = ""

        try:
            async for llm_chunk in llm.get_response_stream(messages):
                if response_cancelled.is_set():
                    break
                full_llm_response += llm_chunk
                await websocket.send_json({"type": "llm_partial", "text": llm_chunk})

                sentence_buf += llm_chunk
                sentence, sentence_buf = flush_sentence_buffer(sentence_buf)
                if sentence:
                    tts_session.feed(sentence)

            if sentence_buf.strip() and not response_cancelled.is_set():
                tts_session.feed(sentence_buf.strip())
        except Exception as e:
            logger.error(f"Error during LLM streaming: {e}")

        # Завершаем TTS (если ещё не принудительно остановлен)
        try:
            await tts_session.finish()
        except (asyncio.CancelledError, Exception):
            pass

        if full_llm_response:
            messages.append({"role": "assistant", "content": full_llm_response})

        await tts_task

        active_tts_session = None
        is_responding = False
        was_cancelled = response_cancelled.is_set()
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
                    current_lang = msg.get("lang", "ru-RU")
                    current_voice = "nigora" if current_lang == "uz-UZ" else "anton"
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

                    # Отменить предыдущий ответ если ещё идёт
                    if is_responding:
                        force_cancel_response()
                        await asyncio.sleep(0.1)

                    await websocket.send_json({"type": "stt_final", "text": final_text})
                    messages.append({"role": "user", "content": final_text})
                    asyncio.create_task(process_response(final_text))

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
        if is_responding:
            force_cancel_response()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
