import asyncio
import re
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import Protocol, TypeVar
import logging

import numpy as np
import torch
from numpy.typing import NDArray

from fastrtc.utils import async_aggregate_bytes_to_16bit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TTSOptions:
    pass


T = TypeVar("T", bound=TTSOptions, contravariant=True)


class TTSModel(Protocol[T]):
    def tts(
        self, text: str, options: T | None = None
    ) -> tuple[int, NDArray[np.float32] | NDArray[np.int16]]: ...

    def stream_tts(
        self, text: str, options: T | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32] | NDArray[np.int16]], None]: ...

    def stream_tts_sync(
        self, text: str, options: T | None = None
    ) -> Generator[tuple[int, NDArray[np.float32] | NDArray[np.int16]], None, None]: ...


@dataclass
class KokoroV11ZhTTSOptions(TTSOptions):
    voice: str = "zf_001"
    speed: float = 1.0
    lang: str = "zh"
    repo_id: str = "hexgrad/Kokoro-82M-v1.1-zh"
    sample_rate: int = 24000
    silence_between_paragraphs: int = 5000
    join_sentences: bool = True


class KokoroV11ZhTTSModel(TTSModel):
    
    def __init__(self, repo_id: str = "hexgrad/Kokoro-82M-v1.1-zh"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.repo_id = repo_id
        self._initialize_model()

    def _initialize_model(self):
        logger.info(f"Initializing model with repo_id: {self.repo_id}")
        try:
            from kokoro import KModel, KPipeline
            
            self._model = KModel(repo_id=self.repo_id).to(self.device).eval()
            
            self._en_pipeline = KPipeline(lang_code='a', repo_id=self.repo_id, model=False)
            
            def en_callable(text):
                if text == 'Kokoro':
                    return 'kˈOkəɹO'
                elif text == 'Sol':
                    return 'sˈOl'
                return next(self._en_pipeline(text)).phonemes
            
            self._zh_pipeline = KPipeline(
                lang_code='z', 
                repo_id=self.repo_id, 
                model=self._model, 
                en_callable=en_callable
            )
            
            logger.info(f"Model initialization completed, device: {self.device}")
            
            logger.info("Warming up model with test text...")
            test_result = next(self._zh_pipeline("测试", voice="zf_001", speed=1.0))
            logger.info("Model warmup completed")
        except ImportError as e:
            raise RuntimeError(
                "kokoro library is not installed. Please install it using 'pip install kokoro>=0.8.1 \"misaki[zh]>=0.8.1\"'."
            ) from e
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}, continuing anyway")
    
    def _speed_callable(self, len_ps: int) -> float:
        speed = 0.8
        if len_ps <= 83:
            speed = 1
        elif len_ps < 183:
            speed = 1 - (len_ps - 83) / 500
        return speed * 1.1
    
    def _split_text_into_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'[。！？.!?]+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def tts(
        self, text: str, options: KokoroV11ZhTTSOptions | None = None
    ) -> tuple[int, NDArray[np.float32]]:
        options = options or KokoroV11ZhTTSOptions()
        
        sentences = self._split_text_into_sentences(text)
        audio_chunks = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            generator = self._zh_pipeline(
                sentence, 
                voice=options.voice, 
                speed=self._speed_callable
            )
            
            result = next(generator)
            wav = result.audio
            
            if torch.is_tensor(wav):
                wav = wav.cpu().numpy()
            
            if i > 0 and options.silence_between_paragraphs > 0:
                silence = np.zeros(options.silence_between_paragraphs, dtype=np.float32)
                wav = np.concatenate([silence, wav])
            
            audio_chunks.append(wav)
        
        if audio_chunks:
            combined_audio = np.concatenate(audio_chunks, axis=0)
            return options.sample_rate, combined_audio.astype(np.float32)
        else:
            return options.sample_rate, np.array([], dtype=np.float32)
    
    async def stream_tts(
        self, text: str, options: KokoroV11ZhTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        options = options or KokoroV11ZhTTSOptions()
        
        sentences = self._split_text_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            generator = self._zh_pipeline(
                sentence, 
                voice=options.voice, 
                speed=self._speed_callable
            )
            
            result = next(generator)
            wav = result.audio
            
            if torch.is_tensor(wav):
                wav = wav.cpu().numpy()
            
            yield (options.sample_rate, wav.astype(np.float32))
            
            await asyncio.sleep(0.01)
    
    def stream_tts_sync(
        self, text: str, options: KokoroV11ZhTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        options = options or KokoroV11ZhTTSOptions()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            iterator = self.stream_tts(text, options).__aiter__()
            while True:
                try:
                    audio_chunk = loop.run_until_complete(iterator.__anext__())
                    yield (options.sample_rate, audio_chunk)
                except StopAsyncIteration:
                    break
        finally:
            loop.close()


def get_kokoro_v11_zh_model() -> KokoroV11ZhTTSModel:
    return KokoroV11ZhTTSModel()

