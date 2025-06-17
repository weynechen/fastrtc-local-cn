# FastRTC 本地中文语音对话演示

基于 FastRTC 构建的实时语音对话系统，支持中文语音识别(STT)和语音合成(TTS)功能，完全本地运行。

## 功能特性

- **实时语音交互**: 基于流式音频处理，支持对话打断功能
- **中文语音识别**: 采用 FunASR 的 SenseVoiceSmall 模型
- **中文语音合成**: 使用 Kokoro v1.1 中文 TTS 模型
- **本地大语言模型集成**: 兼容 Ollama 本地语言模型推理
- **Web 用户界面**: 基于 Gradio 的易用界面
- **GPU 加速**: 自动检测 CUDA 设备以提升性能

## 系统要求

- Python 3.12+
- CUDA 兼容的 GPU（可选，用于加速）
- [Ollama](https://ollama.com/) 在本地 11434 端口运行
- 已安装的 Qwen2.5 模型

## 安装部署

1. 克隆代码仓库：
```bash
git clone https://github.com/weynechen/fastrtc-local-cn
cd fastrtc-local-cn
```

2. 使用 uv 安装依赖（推荐）：
```bash
pip install uv
uv sync
```

3. 设置 Ollama 并安装 Qwen2.5：
```bash
# 安装 Ollama（请访问 https://ollama.com/ 获取安装说明）
ollama pull qwen2.5:latest
ollama serve
```

## 使用方法
1. 国内环境访问hf有问题，需先设置环境变量
Linux:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
Windows CMD
```
set HF_ENDPOINT=https://hf-mirror.com
```
Windows Powershell
```
$env:HF_ENDPOINT = "https://hf-mirror.com"
```
2. 启动应用程序：
```bash
uv run main.py
```
初次启动需要等待较久的时间，需要下载模型。

3. 打开浏览器并访问显示的网址（通常是 `http://localhost:7860`）

4. 点击麦克风按钮开始语音对话

5. 用中文说话 - 系统将：
   - 使用 FunASR 将语音转换为文本
   - 通过 Qwen2.5 大语言模型处理文本
   - 使用 Kokoro TTS 生成音频回复
   - 实时流式播放音频响应

## 系统架构

### 核心组件

- **main.py**: 核心应用程序，包含 FastRTC 流处理和 Gradio 界面
- **stt_adapter.py**: 语音转文本适配器，使用 FunASR SenseVoiceSmall
- **tts_adapter.py**: 文本转语音适配器，使用 Kokoro v1.1 中文模型

### 音频处理流水线

1. **音频输入**: 麦克风以 16kHz 采样率捕获语音
2. **语音活动检测**: FastRTC 内置 VAD 检测语音活动
3. **语音识别**: FunASR 将语音转换为中文文本
4. **语言处理**: Ollama/Qwen2.5 生成智能回复
5. **语音合成**: Kokoro 以 24kHz 采样率合成自然中文语音
6. **音频输出**: 实时流式播放，支持打断功能

### 关键特性

- **流式 TTS**: 逐句处理文本以获得更快的响应速度
- **打断功能**: 用户可以打断 AI 的回复
- **智能句子分割**: 自动检测完整句子
- **GPU 优化**: 自动利用 CUDA 加速

## 配置说明

### STT 配置
- 模型: `iic/SenseVoiceSmall`
- 语言: 中文 (zh)
- VAD: 单段最大时长 30 秒
- 格式: 16-bit PCM, 16kHz

### TTS 配置
- 模型: `hexgrad/Kokoro-82M-v1.1-zh`
- 音色: `zf_001` (女声)
- 采样率: 24kHz
- 根据文本长度动态调整语速

### LLM 配置
- 模型: `ollama/qwen2.5:latest`
- API 地址: `http://localhost:11434`
- 系统提示: 配置为提供有用、简洁的中文回复

## 问题排查

### 常见问题

1. **模型加载错误**:
   - 确保有足够的磁盘空间用于模型下载
   - 检查网络连接以进行初始模型下载

2. **CUDA 问题**:
   - 验证 CUDA 安装: `nvidia-smi`
   - 检查 PyTorch CUDA 兼容性: `torch.cuda.is_available()`

3. **Ollama 连接问题**:
   - 验证 Ollama 是否运行: `curl http://localhost:11434/api/tags`
   - 确保 Qwen2.5 模型已安装: `ollama list`

4. **音频问题**:
   - 检查麦克风权限
   - 验证音频设备兼容性
   - 使用较短的音频片段进行测试（需要 >2 秒）

### 性能优化

- 在可用时使用 GPU 加速
- 调整 `audio_chunk_duration` 以平衡延迟和质量
- 根据您的环境调整 VAD 阈值

## 依赖项

核心依赖包括：
- `fastrtc[stt,tts,vad]>=0.0.28`: 实时通信框架
- `funasr>=1.2.6`: 语音识别
- `kokoro>=0.9.4`: 语音合成
- `litellm>=1.72.6`: 大语言模型集成
- `torch>=2.7.1`: 深度学习框架
- `gradio`: Web 界面

## 许可证

MIT

## 技术支持

如有问题和疑问：
1. 查看上述问题排查部分
2. 查看日志获取详细错误信息
3. 确保所有先决条件都已正确安装
4. 验证模型兼容性和版本

## 开发说明

### 项目结构
```
fastrtc_ex/
├── main.py              # 主应用程序入口
├── stt_adapter.py       # 语音转文本适配器
├── tts_adapter.py       # 文本转语音适配器
├── pyproject.toml       # 项目配置文件
├── README.md            # 英文说明文档
├── README_CN.md         # 中文说明文档
└── uv.lock             # 依赖锁文件
```

### 自定义配置

您可以通过修改以下参数来自定义系统行为：

- **VAD 阈值**: 在 `main.py` 中调整 `started_talking_threshold` 和 `speech_threshold`
- **音频块持续时间**: 修改 `audio_chunk_duration` 以优化延迟
- **TTS 语速**: 在 `tts_adapter.py` 中调整 `_speed_callable` 函数
- **系统提示**: 在 `main.py` 中修改 `system_message` 内容

### 扩展功能

系统设计为模块化，您可以轻松：
- 替换 STT 模型（实现 `STTModel` 协议）
- 更换 TTS 引擎（实现 `TTSModel` 协议）
- 集成不同的语言模型
- 添加新的音频处理功能 