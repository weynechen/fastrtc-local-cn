from fastrtc import (
    AdditionalOutputs,
    Stream,
    ReplyOnPause,
    AlgoOptions
)
import numpy as np
from numpy.typing import NDArray
from fastapi import FastAPI
import gradio as gr
from stt_adapter import LocalFunASR
from tts_adapter import KokoroV11ZhTTSModel
from litellm import completion
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


stt_model = LocalFunASR()
tts_model = KokoroV11ZhTTSModel()


def extract_complete_sentences(text: str) -> tuple[list[str], str]:
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5\w\s。！？；，：、.,!?;:()]', '', text)
    
    sentence_endings = []
    for match in re.finditer(r'[。！？.!?]', cleaned_text):
        sentence_endings.append(match.end())
    
    if not sentence_endings:
        return [], cleaned_text
    
    sentences = []
    start = 0
    
    for end_pos in sentence_endings:
        sentence = cleaned_text[start:end_pos].strip()
        if len(sentence) > 8:
            sentences.append(sentence)
        start = end_pos
    
    remaining_text = cleaned_text[sentence_endings[-1]:].strip() if sentence_endings else cleaned_text
    return sentences, remaining_text


def response(
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chatbot: list[dict] | None = None,
):
    chatbot = chatbot or []
    system_message = {
        "role": "system",
        "content": "You are a helpful AI assistant. Please provide clear, accurate and concise responses. Keep your tone friendly and professional. Respond in Chinese.Very short and concise."
    }
    messages = [system_message] 
    messages = [{"role": d["role"], "content": d["content"]} for d in chatbot]
    
    sample_rate, audio_data = audio
    if audio_data.ndim > 1:
        audio_data = audio_data[0] if audio_data.shape[0] < audio_data.shape[1] else audio_data[:, 0]

    duration = len(audio_data) / sample_rate
    if duration < 2:
        logger.info(f"Audio duration {duration:.2f}s is less than 2 seconds, filtered")
        return

    text = stt_model.stt(audio)
    if text:
        chatbot.append({"role": "user", "content": text})
        yield AdditionalOutputs(chatbot)

    messages.append({"role": "user", "content": text})

    response = completion(
        model="ollama/qwen2.5:latest", 
        messages=messages,
        api_base="http://localhost:11434",
        stream=True
    )
    
    accumulated_text = ""
    processed_sentences = []
    
    for chunk in response:
        if chunk['choices'][0]['delta'].content:
            accumulated_text += chunk['choices'][0]['delta'].content
            
            complete_sentences, remaining_text = extract_complete_sentences(accumulated_text)
            
            for sentence in complete_sentences:
                if sentence not in processed_sentences:
                    processed_sentences.append(sentence)
                    logger.info(f"Extracted complete sentence: {sentence}")
                    
                    for audio_chunk in tts_model.stream_tts_sync(sentence):
                        sample_rate, audio_data = audio_chunk
                        chatbot.append({"role": "assistant", "content": sentence})
                        yield audio_data, AdditionalOutputs(chatbot)
            
            if complete_sentences:
                accumulated_text = remaining_text
    
    final_remaining_text = ""
    if accumulated_text.strip() and len(accumulated_text.strip()) > 5:
        final_remaining_text = accumulated_text.strip()
        logger.info(f"Processing final remaining text: {final_remaining_text}")
        for audio_chunk in tts_model.stream_tts_sync(final_remaining_text):
            sample_rate, audio_data = audio_chunk
            chatbot.append({"role": "assistant", "content": final_remaining_text})
            yield audio_data, AdditionalOutputs(chatbot)
    

    
    logger.info("response end")

chatbot = gr.Chatbot(type="messages")
stream = Stream(
    handler= ReplyOnPause(response,
                          can_interrupt=True,
                          input_sample_rate=16000,
                          output_sample_rate=24000,
                          algo_options=AlgoOptions(
                            audio_chunk_duration=1.0,
                            started_talking_threshold=0.2,
                            speech_threshold=0.1,
                 )
                          ),
    modality="audio",
    mode="send-receive",
    additional_outputs_handler=lambda *args: args[-1],
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
)

app = FastAPI()

stream.mount(app)

stream.ui.launch()