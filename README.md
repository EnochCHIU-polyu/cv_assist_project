# cv_assist_project

这是一个计算机视觉助手项目，整合了 **OWL-ViT 目标检测**、**MediaPipe 手部追踪**、以及 **MiDaS 深度估计**，用于演示多模型协同工作实现智能视觉分析与交互。

---

## 🛠 技术栈

- **语言**：Python 3.8+
- **主要依赖**：
  - `torch`, `torchvision`（深度学习模型）
  - `opencv-python`（图像处理）
  - `mediapipe`（手部关键点检测）
  - `midas`（深度估计模型）
  - `owl-vit`（视觉-语言检测模型）
- **项目结构**：
  ```
  cv_assist_project/
  ├─ config.py          # 配置参数
  ├─ main.py            # 程序入口
  ├─ detectors/         # 各检测模块
  ├─ core/              # 系统集成与引导逻辑
  ├─ utils/             # 共用工具函数
   ├─ tests/             # 测试脚本目录
  └─ README.md          # 当前文档
  ```

---

## 🚀 启动项目

1. **环境准备**  
   ```bash
   cd cv_assist_project
   # 可选：创建 Python 虚拟环境并激活
   python -m venv .venv           # 仅第一次或当需要隔离依赖时
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # Windows cmd
   call .venv\Scripts\activate
   # Linux / macOS
   source .venv/bin/activate

   pip install -r requirements.txt
   ```
   > 🔧 如果你希望脚本自动完成以上步骤，可使用仓库根目录下的 `run.sh`（bash）或新增的 `run.bat`（Windows）。

2. **配置**  
   - 编辑 `config.py` 修改输入源、模型路径等参数（如需）。

3. **运行**  
   ```bash
   # 直接执行
   python main.py
   # 或使用运行脚本（见上文）
   ./run.sh       # Linux/macOS
   run.bat        # Windows cmd
   ```
   程序将加载检测器，打开摄像头/视频流并在画面上叠加检测结果与深度信息。

   启动后会看到类似下面的输出：
   ```
   ============================================================
    CV 视觉辅助系统
   ============================================================
    OWL-ViT: google/owlvit-base-patch32
    MiDaS: MiDaS_small
    设备: cpu
    FP16: False
   ============================================================
   …
   控制：
     q - 退出
     d - 切换深度显示
   检测目标: ['a cup', 'a bottle']
   ```
   - 按 `d` 切换深度渲染，按 `q` 退出窗口。

4. **测试**  
   验证功能是否正常：
   ```bash
   python tests/test_all.py
   ```


---

## 🧩 功能概览

- **目标检测** 使用 OWL-ViT 模型识别图像中物体。
- **手部追踪** MediaPipe 提供掌心和手指关键点坐标。
- **深度估计** MiDaS 生成场景深度图，用于测距或增强理解。
- **🎤 语音识别 (ASR)** 使用 OpenAI Whisper 将用户语音转换为文本指令。
- **🔊 文本转语音 (TTS)** 将系统引导指令通过自然语言播放，帮助视觉障碍用户。
- **模块化设计**：各 detector 互相独立，可扩展新算法或模型。
- **简单的系统逻辑** 位于 `core/`，协调各模块输入输出，进行可视化展示。

---

## 🎤 音频功能（ASR & TTS）

本系统现已集成语音识别和文本转语音功能，让视觉障碍用户可以：
- 通过 **语音命令** 指定要寻找的目标物品
- 通过 **语音播报** 接收系统的引导指令

### 快速开始

1. **安装音频依赖**
   ```bash
   pip install openai-whisper pyttsx3 sounddevice scipy
   ```

2. **测试音频功能**
   ```bash
   python tests/test_audio.py
   ```

3. **启用语音运行**
   ```bash
   # 使用语音配置（同时启用 ASR 和 TTS）
   python main.py --config voice
   
   # 仅启用 TTS（ASR 关闭）
   python main.py --config tts
   ```

4. **使用语音控制**
   - 按 `v` 键开始语音输入
   - 说出指令，如"找到杯子"或"where is the cup"
   - 系统会自动识别并开始寻找目标
   - 引导指令会通过语音播放（如"向左移动"、"向上移动"等）

### 更多信息

详细的音频功能使用指南请参考：[audio/README.md](audio/README.md)

---

## 🤝 协作建议

- **分支规范**：每个新功能以 feature/xxx 命名，提交说明清晰。
- **代码风格**：遵循 PEP8，建议安装并使用 `flake8` 或 `black`。
- **文档**：新增功能或改动请更新此 README 或 `doc/summary.md`。
- **测试**：每次改动后运行 `tests/test_all.py`，确保模块兼容无误。

---

## 📦 其他

- 可通过修改 `requirements.txt` 添加更多依赖。
- 若需在 Linux/macOS 使用，请参考 `install.sh` 或 `run.sh`，命令类似。

---