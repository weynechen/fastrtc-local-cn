import numpy as np
from typing import Protocol, Tuple
import logging
import torch
import tempfile
import wave

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from funasr import AutoModel


class STTModel(Protocol):
    def stt(self, audio: Tuple[int, np.ndarray]) -> str: ...


class LocalFunASR:
    
    def __init__(self):
        self.model = None
        
        if torch.cuda.is_available():
            device = "cuda:0"
            logger.info("CUDA device detected, using GPU acceleration")
        else:
            device = "cpu"
            logger.info("No CUDA device detected, using CPU")
        
        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_kwargs={"max_single_segment_time": 30000},
            hub="ms",
            device=device,
            disable_update=True,
        )
    
    def stt(self, audio: Tuple[int, np.ndarray]) -> str:
        sample_rate, audio_array = audio
        logger.info(f"Received audio data: sample_rate={sample_rate}, data_type={type(audio_array)}, shape={getattr(audio_array, 'shape', 'N/A')}")

        if self.model is None:
            raise RuntimeError("ASR model not loaded, please initialize model first")
        
        try:
            if audio_array.dtype == np.float32:
                audio_array = (audio_array * 32767).astype(np.int16)
            elif audio_array.dtype != np.int16:
                audio_array = audio_array.astype(np.int16)
            
            if audio_array.ndim > 1:
                audio_array = audio_array[0] if audio_array.shape[0] < audio_array.shape[1] else audio_array[:, 0]
            
            logger.info(f"Processed audio length: {len(audio_array)}")
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                
                with wave.open(tmp_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_array.tobytes())
            
            result = self.model.generate(
                input=tmp_path,
                cache={},
                language="zh",
                use_itn=True,
                batch_size_s=60,
            )
            
            
            text = result[0]["text"] if result and len(result) > 0 else ""
            
            parsed_text = self._parse_funasr_output(text)
            
            return parsed_text if parsed_text else ""
            
        except Exception as e:
            logger.error(f"Speech transcription error: {e}")
            return f"Transcription failed: {str(e)}" 
    
    def _parse_funasr_output(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        
        try:
            import re
            pattern = r'<\|([^|]+)\|><\|([^|]+)\|><\|([^|]+)\|><\|([^|]+)\|>(.*)'
            match = re.match(pattern, raw_text)
            
            if match:
                language = match.group(1)
                emotion = match.group(2)
                speech_type = match.group(3)
                processing = match.group(4)
                actual_text = match.group(5)
                
                logger.debug(f"Parse result - language: {language}, emotion: {emotion}, type: {speech_type}, processing: {processing}")
                logger.debug(f"Extracted text: {actual_text}")
                
                return actual_text.strip()
            else:
                logger.warning(f"Could not parse funasr output format, returning original text: {raw_text}")
                return raw_text
                
        except Exception as e:
            logger.error(f"Error parsing funasr output: {e}")
            return raw_text