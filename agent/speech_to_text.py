import asyncio
import queue as _queue
import grpc
import logging
import numpy as np
import speech_recognition as sr
import soundfile as sf
from scipy.signal import resample_poly
from pydub import AudioSegment, utils
from tempfile import NamedTemporaryFile
from pathlib import Path
import os

import yandex.cloud.ai.stt.v2.stt_service_pb2 as stt_pb2_v2
import yandex.cloud.ai.stt.v2.stt_service_pb2_grpc as stt_pb2_grpc_v2


class YandexSpeechRecognizer:
    CHUNK_SIZE = 4096
    SUPPORTED_RATES = {8000, 16000, 48000}

    def __init__(self, folder_id: str, iam_token: str):
        self.folder_id = folder_id
        self.iam_token = iam_token
        self.stub = self._create_stub()

    def _create_stub(self):
        try:
            creds = grpc.ssl_channel_credentials()
            channel = grpc.secure_channel('stt.api.cloud.yandex.net:443', creds)
            return stt_pb2_grpc_v2.SttServiceStub(channel)
        except Exception as e:
            logging.error(f"Failed to connect to Yandex STT: {e}")
            return None

    def _generate_requests(self, audio_data: sr.AudioData, partial_results: bool = False):
        spec = stt_pb2_v2.RecognitionSpec(
            language_code='ru-RU',
            profanity_filter=True,
            model='general',
            partial_results=partial_results,
            audio_encoding='LINEAR16_PCM',
            sample_rate_hertz=audio_data.sample_rate
        )
        config = stt_pb2_v2.RecognitionConfig(
            specification=spec,
            folder_id=self.folder_id
        )
        yield stt_pb2_v2.StreamingRecognitionRequest(config=config)

        raw = audio_data.get_raw_data()
        for i in range(0, len(raw), self.CHUNK_SIZE):
            yield stt_pb2_v2.StreamingRecognitionRequest(audio_content=raw[i:i + self.CHUNK_SIZE])

    def _recognize_audio_data(self, audio_data: sr.AudioData, on_partial=None) -> str:
        """
        Распознаёт аудио. Если передан on_partial(text), включает partial_results
        и вызывает callback для промежуточных результатов.
        """
        if not self.stub:
            logging.error("Stub not initialized; cannot recognize.")
            return ""
        try:
            responses = self.stub.StreamingRecognize(
                self._generate_requests(audio_data, partial_results=on_partial is not None),
                metadata=[('authorization', f'Api-Key {self.iam_token}')]
            )
            texts = []
            for resp in responses:
                for chunk in getattr(resp, 'chunks', []):
                    if chunk.final:
                        texts.extend([alt.text for alt in chunk.alternatives])
                    elif on_partial:
                        for alt in chunk.alternatives:
                            if alt.text:
                                on_partial(alt.text)
            return ' '.join(texts).strip()
        except grpc.RpcError as e:
            logging.error(f"gRPC error: {e.code()} — {e.details()}")
            return ""
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return ""

    def transcribe_bytes(self, wav_bytes: bytes, on_partial=None) -> str:
        """
        Транскрибирует сырые WAV-байты (без записи в файл).
        Если передан on_partial(text), будет вызываться для промежуточных результатов.
        """
        from io import BytesIO
        buf = BytesIO(wav_bytes)
        data, sr_rate = sf.read(buf, dtype='int16')
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.int16)
        raw = data.tobytes()

        target_sr = min(self.SUPPORTED_RATES, key=lambda r: abs(r - sr_rate))
        if sr_rate not in self.SUPPORTED_RATES:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max
            arr_rs = resample_poly(arr, target_sr, sr_rate)
            raw = (arr_rs * np.iinfo(np.int16).max).astype(np.int16).tobytes()
            logging.info(f"transcribe_bytes: resampled {sr_rate}→{target_sr} Hz")

        audio_data = sr.AudioData(raw, target_sr, 2)
        return self._recognize_audio_data(audio_data, on_partial=on_partial)

    def load_audio_data(self, filepath: str) -> sr.AudioData:
        """
        Reads .wav/.ogg/.flac via soundfile, others via pydub, converts to mono & resamples.
        """
        ext = Path(filepath).suffix.lower()
        if ext in {'.wav', '.ogg', '.flac'}:
            data, sr_rate = sf.read(filepath, dtype='int16')
            if data.ndim > 1:
                data = data.mean(axis=1).astype(np.int16)
            raw = data.tobytes()
            sample_width = 2
        else:
            seg = AudioSegment.from_file(filepath)
            seg = seg.set_channels(1)
            raw = seg.raw_data
            sr_rate = seg.frame_rate
            sample_width = seg.sample_width

        target_sr = min(self.SUPPORTED_RATES, key=lambda r: abs(r - sr_rate))
        if sr_rate not in self.SUPPORTED_RATES:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max
            arr_rs = resample_poly(arr, target_sr, sr_rate)
            raw = (arr_rs * np.iinfo(np.int16).max).astype(np.int16).tobytes()
            logging.info(f"Resampled from {sr_rate}→{target_sr} Hz")

        return sr.AudioData(raw, target_sr, sample_width)

    def _generate_streaming_requests(self, sample_rate: int, chunk_queue: _queue.Queue, language_code: str = 'ru-RU'):
        """gRPC request generator: config first, then raw int16 PCM chunks from queue (None = end)."""
        spec = stt_pb2_v2.RecognitionSpec(
            language_code=language_code,
            profanity_filter=True,
            model='general',
            partial_results=True,
            audio_encoding='LINEAR16_PCM',
            sample_rate_hertz=sample_rate,
        )
        yield stt_pb2_v2.StreamingRecognitionRequest(
            config=stt_pb2_v2.RecognitionConfig(specification=spec, folder_id=self.folder_id)
        )
        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                break
            yield stt_pb2_v2.StreamingRecognitionRequest(audio_content=chunk)

    def recognize_streaming(self, sample_rate: int, chunk_queue: _queue.Queue, on_partial=None, language_code: str = 'ru-RU') -> str:
        """Runs in thread. Streams raw int16 PCM chunks from queue to Yandex STT."""
        if not self.stub:
            logging.error("Stub not initialized; cannot recognize.")
            return ""
        try:
            responses = self.stub.StreamingRecognize(
                self._generate_streaming_requests(sample_rate, chunk_queue, language_code),
                metadata=[('authorization', f'Api-Key {self.iam_token}')]
            )
            texts = []
            for resp in responses:
                for chunk in getattr(resp, 'chunks', []):
                    if chunk.final:
                        texts.extend([alt.text for alt in chunk.alternatives])
                    elif on_partial:
                        for alt in chunk.alternatives:
                            if alt.text:
                                on_partial(alt.text)
            return ' '.join(texts).strip()
        except grpc.RpcError as e:
            if "you should send at least one audio fragment" in e.details():
                # Ignore this specific error, it just means the stream was closed before audio arrived.
                pass
            else:
                logging.error(f"gRPC streaming error: {e.code()} — {e.details()}")
            return ""
        except Exception as e:
            logging.error(f"Unexpected error in recognize_streaming: {e}")
            return ""

    def transcribe(self, filepath: str) -> str:
        """
        Splits long audio into <=5min chunks, streams each through STT, and merges text.
        """
        # Load full audio as mono
        audio = AudioSegment.from_file(filepath)
        audio = audio.set_channels(1)

        chunk_ms = 1 * 60 * 1000
        chunks = utils.make_chunks(audio, chunk_ms)

        all_texts = []
        for idx, chunk in enumerate(chunks):
            with NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                chunk.export(tmp.name, format='wav')
                tmp_path = tmp.name

            # Convert to AudioData
            audio_data = self.load_audio_data(tmp_path)
            os.remove(tmp_path)

            # Recognize chunk
            part_text = self._recognize_audio_data(audio_data)
            if part_text:
                all_texts.append(part_text)

        return ' '.join(all_texts)


class SttStreamingSession:
    """
    Per-turn streaming STT session.
    Feeds raw int16 PCM chunks into a live gRPC StreamingRecognize call
    as the user speaks, so partial results arrive in real time.
    """

    def __init__(
        self,
        recognizer: YandexSpeechRecognizer,
        sample_rate: int,
        loop: asyncio.AbstractEventLoop,
        partial_queue: asyncio.Queue,
        language_code: str = 'ru-RU'
    ):
        self._rec = recognizer
        self._sr = min(recognizer.SUPPORTED_RATES, key=lambda r: abs(r - sample_rate))
        self._loop = loop
        self._partial_queue = partial_queue
        self._chunks: _queue.Queue = _queue.Queue()
        self._task: asyncio.Task | None = None
        self._lang = language_code
        self._fed_audio = False

    def start(self):
        """Start gRPC recognition in a background thread (non-blocking)."""
        self._task = self._loop.create_task(
            asyncio.to_thread(
                self._rec.recognize_streaming,
                self._sr,
                self._chunks,
                self._on_partial,
                self._lang
            )
        )

    def _on_partial(self, text: str):
        asyncio.run_coroutine_threadsafe(self._partial_queue.put(text), self._loop)

    def feed(self, pcm_bytes: bytes):
        """Push a raw int16 PCM chunk into the stream."""
        self._fed_audio = True
        self._chunks.put(pcm_bytes)

    async def finish(self) -> str:
        """Signal end-of-stream and await final transcription."""
        if not self._fed_audio:
            # Prevent gRPC INVALID_ARGUMENT if no audio was sent
            self._chunks.put(None)
            if self._task:
                # wait for task to finish to clean up
                await self._task
            return ""

        self._chunks.put(None)
        return await self._task