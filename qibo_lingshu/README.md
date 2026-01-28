# 岐伯灵枢：中医四诊AI辅助分析平台

## 项目简介

岐伯灵枢是一个基于Jetson Nano开发板的中医四诊AI辅助分析平台，融合传统中医智慧与现代人工智能技术，通过望、闻、问、切四种诊断方法，为用户提供专业的健康分析服务。

### 项目特点

- **硬件优化**：专为Jetson Nano开发板优化，充分利用其GPU加速能力
- **四诊合一**：集成望诊（舌苔分析）、闻诊（健康数据采集）、问诊（语音输入）、切诊（脉象分析）
- **AI驱动**：结合规则引擎与大模型，提供准确的中医辨证分析
- **用户友好**：简洁直观的Web界面，操作简单便捷
- **模块化设计**：各功能模块独立，便于维护和扩展

## 系统架构

```
岐伯灵枢：中医四诊AI辅助分析平台
├── 望诊模块（舌苔分析）
│   ├── 图像采集
│   ├── 图像预处理
│   └── 舌苔特征分析
├── 闻诊模块（健康数据采集）
│   ├── 蓝牙连接
│   ├── 生命体征采集
│   └── 异常检测
├── 问诊模块（语音输入）
│   ├── 语音录制
│   ├── 语音转文字
│   └── 症状提取
├── 切诊模块（脉象分析）
│   ├── 脉搏数据处理
│   └── 脉象特征分析
└── 中医分析引擎
    ├── 规则引擎
    ├── AI模型集成
    └── 综合辨证分析
```

## 硬件要求

### 主控设备
- **Jetson Nano开发板**（推荐B01版本）
- **MicroSD卡**：64GB以上，Class 10或更高
- **电源适配器**：5V 4A直流电源

### 摄像头
- **USB摄像头**或**CSI摄像头**
- 分辨率：640×480以上
- 帧率：15fps以上

### 健康监测设备
- **智能手环**（支持蓝牙连接）
- 推荐型号：小米手环、华为手环等
- 需支持：心率、血氧、血压、体温监测

### 音频设备
- **USB麦克风**或**3.5mm麦克风**
- 支持语音识别

### 可选配件
- **散热风扇**：确保长时间运行稳定
- **外壳**：保护开发板和连接线
- **显示屏**：HDMI或DP接口，用于本地调试

## 软件环境

### 操作系统
- **Ubuntu 18.04**或**Ubuntu 20.04**
- **JetPack SDK 4.6**或更高版本

### Python环境
- **Python 3.6**或更高版本
- **pip**包管理器
- **虚拟环境**（推荐）

### 依赖库
详细依赖列表请参考`requirements.txt`文件，主要包括：
- Flask：Web框架
- OpenCV：图像处理
- PyTorch/TensorFlow：深度学习框架
- SpeechRecognition：语音识别
- Bleak：蓝牙通信
- 其他辅助库

## 安装指南

### 1. 系统准备

#### 1.1 刷写JetPack系统
1. 下载JetPack SDK镜像
2. 使用Etcher等工具将镜像刷写到MicroSD卡
3. 将MicroSD卡插入Jetson Nano并启动

#### 1.2 系统配置
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装必要的系统工具
sudo apt install -y python3-pip python3-dev python3-setuptools
sudo apt install -y git curl wget vim
sudo apt install -y build-essential cmake pkg-config

# 安装摄像头支持
sudo apt install -y python3-gst-1.0 gstreamer1.0-plugins-*
sudo apt install -y v4l-utils

# 安装音频支持
sudo apt install -y alsa-utils pulseaudio
sudo apt install -y python3-pyaudio

# 安装蓝牙支持
sudo apt install -y bluetooth bluez blueman
sudo systemctl enable bluetooth
sudo systemctl start bluetooth
```

### 2. 项目部署

#### 2.1 克隆项目
```bash
# 克隆项目到本地
git clone https://github.com/yourusername/qibo_lingshu.git
cd qibo_lingshu
```

#### 2.2 创建虚拟环境
```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
pip install --upgrade pip
```

#### 2.3 安装依赖
```bash
# 安装Python依赖
pip install -r requirements.txt

# 注意：对于Jetson Nano，某些包需要特殊安装方式
# 安装TensorFlow（Jetson Nano兼容版本）
sudo pip3 install --pre tensorflow==2.13.0

# 安装PyTorch（Jetson Nano兼容版本）
sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装系统监控工具
sudo pip3 install jetson-stats
```

#### 2.4 配置项目
```bash
# 复制配置文件模板
cp config.py.example config.py

# 编辑配置文件
vim config.py
```

### 3. 硬件配置

#### 3.1 摄像头配置
```bash
# 检查摄像头设备
ls /dev/video*

# 测试摄像头
cheese  # 图形界面测试
# 或
v4l2-ctl --list-devices
```

#### 3.2 蓝牙配置
```bash
# 启动蓝牙服务
sudo systemctl start bluetooth

# 扫描蓝牙设备
bluetoothctl
scan on
# 查找您的智能手环MAC地址
```

#### 3.3 音频配置
```bash
# 检查音频设备
arecord -l

# 测试录音
arecord -d 5 -f cd test.wav
aplay test.wav
```

## 使用指南

### 1. 启动应用

```bash
# 激活虚拟环境
source venv/bin/activate

# 启动应用
python app.py
```

### 2. 访问界面

打开浏览器，访问：`http://jetson-nano-ip:5000`

### 3. 四诊采集流程

#### 3.1 望诊（舌苔分析）
1. 点击"采集舌苔图像"按钮
2. 将舌头伸出，保持自然状态
3. 等待图像采集和分析完成
4. 查看舌苔分析结果

#### 3.2 闻诊（健康数据采集）
1. 输入智能手环MAC地址
2. 点击"采集健康数据"按钮
3. 等待蓝牙连接和数据采集
4. 查看健康数据分析结果

#### 3.3 问诊（语音输入）
1. 点击"开始录音"按钮
2. 描述您的症状和身体状况
3. 再次点击按钮结束录音
4. 查看语音转文字结果和症状提取

#### 3.4 切诊（脉象分析）
1. 脉象分析基于健康数据自动进行
2. 系统会根据心率等数据分析脉象特征
3. 在分析结果中查看脉象分析

### 4. 查看分析结果

1. 完成四诊信息采集后，点击"开始中医分析"
2. 等待系统综合分析
3. 查看中医诊断结果和治疗建议
4. 可以保存分析结果供后续参考

## 项目结构

```
qibo_lingshu/
├── app.py                      # 主应用入口
├── config.py                   # 配置文件
├── requirements.txt            # 依赖包列表
├── README.md                   # 项目文档
├── modules/                    # 功能模块
│   ├── __init__.py
│   ├── tongue_analyzer.py      # 舌苔分析模块
│   ├── health_collector.py     # 健康数据采集模块
│   ├── voice_input.py          # 语音输入模块
│   └── tcm_engine.py           # 中医分析引擎
├── templates/                  # HTML模板
│   └── index.html              # 主页面
├── static/                     # 静态资源
│   ├── css/                    # CSS样式
│   ├── js/                     # JavaScript脚本
│   └── images/                 # 图片资源
├── uploads/                    # 上传文件存储
├── audio/                      # 音频文件存储
├── models/                     # 模型文件
│   └── tcm_knowledge/          # 中医知识库
│       └── tcm_knowledge.json  # 中医知识数据
└── logs/                       # 日志文件
```

## 核心模块详解

### 1. 舌苔分析模块 (tongue_analyzer.py)

负责舌苔图像的采集、预处理和分析，主要功能包括：

- **图像采集**：通过摄像头采集舌苔图像
- **图像预处理**：调整亮度、对比度，裁剪感兴趣区域
- **特征提取**：提取舌苔颜色、形状、纹理等特征
- **症状分析**：基于舌苔特征分析可能的健康问题

#### 关键函数
```python
# 采集舌苔图像
capture_image(camera_index=None)

# 分析舌苔图像
analyze_tongue(image_path)

# 提取症状
extract_symptoms(analysis_result)
```

### 2. 健康数据采集模块 (health_collector.py)

负责通过蓝牙连接智能手环，采集健康数据，主要功能包括：

- **设备连接**：通过蓝牙连接智能手环
- **数据采集**：采集心率、血氧、血压、体温等数据
- **异常检测**：检测健康数据中的异常指标
- **脉象分析**：基于脉搏数据分析脉象特征

#### 关键函数
```python
# 连接智能手环
async def connect_device(device_mac)

# 采集生命体征
async def collect_vitals()

# 检测异常指标
detect_abnormalities(vitals)

# 分析脉象
analyze_pulse(vitals)
```

### 3. 语音输入模块 (voice_input.py)

负责语音录制和语音转文字，主要功能包括：

- **语音录制**：通过麦克风录制患者症状描述
- **语音转文字**：将语音转换为文字
- **症状提取**：从文字描述中提取症状关键词
- **音频管理**：保存和管理音频文件

#### 关键函数
```python
# 录制语音
capture_voice(duration=None)

# 语音转文字
convert_to_text(audio_data)

# 提取症状
extract_symptoms(text)

# 保存转录文本
save_transcription(text, audio_path=None)
```

### 4. 中医分析引擎 (tcm_engine.py)

负责综合分析四诊信息，提供中医诊断和治疗建议，主要功能包括：

- **知识库管理**：加载和管理中医知识库
- **规则引擎**：基于中医理论进行规则分析
- **AI模型集成**：集成大模型进行智能分析
- **辨证分析**：综合四诊信息进行中医辨证

#### 关键函数
```python
# 综合分析四诊信息
analyze_symptoms(tongue_image_path, vitals, patient_description)

# 基于规则的分析
_rule_based_analysis(tongue_image_path, vitals, patient_description)

# 识别证型
_identify_syndrome_pattern(symptoms, tongue_analysis, pulse_analysis)

# 获取治疗建议
_get_treatment_recommendation(syndrome_pattern)
```

## 配置说明

### 1. 基本配置 (config.py)

```python
class Config:
    # Flask配置
    SECRET_KEY = 'your_secret_key_here'
    
    # 摄像头配置
    CAMERA_INDEX = 0
    CAMERA_RESOLUTION = (640, 480)
    CAMERA_FRAMERATE = 15
    
    # Jetson Nano性能优化配置
    THREAD_POOL_SIZE = 2
    GPU_MEMORY_FRACTION = 0.5
    ENABLE_GPU_ACCELERATION = True
    
    # 中医诊断配置
    TONGUE_ANALYSIS_CONFIDENCE_THRESHOLD = 0.6
    VITALS_NORMAL_RANGES = {
        "heart_rate": {"min": 60, "max": 100},
        "blood_oxygen": {"min": 95, "max": 100},
        "blood_pressure_systolic": {"min": 90, "max": 140},
        "blood_pressure_diastolic": {"min": 60, "max": 90},
        "temperature": {"min": 36.0, "max": 37.3}
    }
    
    # AI模型配置
    AI_API_KEY = 'your_api_key_here'
    AI_MODEL = 'gpt-4-vision-preview'
    AI_TIMEOUT = 30
    AI_MAX_TOKENS = 1500
```

### 2. 日志配置

```python
# 日志级别
LOG_LEVEL = logging.INFO

# 日志文件路径
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'qibo_lingshu.log')

# 日志格式
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```

### 3. 蓝牙设备配置

```python
# 默认蓝牙设备MAC地址
DEFAULT_DEVICE_MAC = '00:00:00:00:00:00'

# 蓝牙连接超时时间
BLUETOOTH_TIMEOUT = 10

# 蓝牙重试次数
BLUETOOTH_RETRY_COUNT = 3
```

## 故障排除

### 1. 摄像头问题

#### 问题描述
摄像头无法打开或图像采集失败

#### 解决方案
```bash
# 检查摄像头设备
ls /dev/video*

# 检查摄像头权限
sudo usermod -a -G video $USER

# 重启系统
sudo reboot
```

### 2. 蓝牙连接问题

#### 问题描述
无法连接智能手环或数据采集失败

#### 解决方案
```bash
# 检查蓝牙服务状态
sudo systemctl status bluetooth

# 重启蓝牙服务
sudo systemctl restart bluetooth

# 重新配对设备
bluetoothctl
remove <device_mac>
scan on
pair <device_mac>
connect <device_mac>
```

### 3. 语音识别问题

#### 问题描述
语音转文字失败或识别准确率低

#### 解决方案
```bash
# 检查麦克风设备
arecord -l

# 测试麦克风
arecord -d 5 -f cd test.wav
aplay test.wav

# 调整麦克风音量
alsamixer
```

### 4. 性能问题

#### 问题描述
系统运行缓慢或响应延迟

#### 解决方案
```bash
# 检查系统资源使用
htop

# 检查GPU使用
tegrastats

# 优化系统性能
sudo nvpmodel -m 0  # 最大功率模式
sudo jetson_clocks  # 最大频率
```

## 性能优化

### 1. Jetson Nano优化

```bash
# 设置最大功率模式
sudo nvpmodel -m 0

# 设置最大频率
sudo jetson_clocks

# 检查当前模式
sudo nvpmodel -q
```

### 2. 应用优化

- **降低图像分辨率**：根据实际需求调整摄像头分辨率
- **减少并发任务**：限制同时运行的任务数量
- **使用GPU加速**：确保深度学习模型使用GPU
- **优化内存使用**：定期清理不需要的变量和缓存

### 3. 模型优化

- **模型量化**：使用量化后的模型减少内存占用
- **模型剪枝**：移除不必要的模型参数
- **批处理优化**：合理设置批处理大小

## 扩展开发

### 1. 添加新的诊断方法

1. 在`modules`目录下创建新模块
2. 实现必要的接口函数
3. 在`app.py`中添加路由
4. 在前端界面中添加相应UI

### 2. 集成新的AI模型

1. 在`tcm_engine.py`中添加新的模型调用函数
2. 更新配置文件添加模型参数
3. 修改分析流程以支持新模型

### 3. 扩展中医知识库

1. 更新`models/tcm_knowledge/tcm_knowledge.json`文件
2. 添加新的证型、中药、穴位等信息
3. 更新分析引擎以支持新知识

## 安全注意事项

1. **数据隐私**：确保患者数据的安全存储和传输
2. **访问控制**：实施适当的用户认证和授权机制
3. **输入验证**：对所有用户输入进行验证和清理
4. **错误处理**：妥善处理错误，避免泄露敏感信息
5. **日志管理**：定期清理日志文件，避免占用过多空间

## 常见问题

### Q: 如何提高舌苔分析的准确性？

A: 可以通过以下方式提高准确性：
1. 确保良好的光照条件
2. 使用高质量的摄像头
3. 保持舌头自然伸出状态
4. 定期校准分析模型

### Q: 智能手环连接失败怎么办？

A: 尝试以下解决方案：
1. 确认手环蓝牙已开启
2. 检查MAC地址是否正确
3. 重启蓝牙服务
4. 重新配对手环

### Q: 语音识别准确率低如何解决？

A: 可以尝试以下方法：
1. 使用高质量的麦克风
2. 确保环境安静
3. 说话清晰、语速适中
4. 调整麦克风位置和音量

### Q: 系统运行缓慢如何优化？

A: 可以尝试以下优化措施：
1. 设置Jetson Nano为最大功率模式
2. 降低图像分辨率
3. 减少同时运行的任务
4. 使用GPU加速

## 技术支持

如果您在使用过程中遇到问题，可以通过以下方式获取支持：

1. **GitHub Issues**：提交问题到项目GitHub仓库
2. **邮件联系**：发送邮件至support@qibolingshu.com
3. **技术论坛**：访问项目技术论坛获取帮助

## 许可证

本项目采用MIT许可证，详情请参考LICENSE文件。

## 贡献指南

欢迎为项目贡献代码！请遵循以下步骤：

1. Fork项目到您的GitHub账户
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -am 'Add some feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 提交Pull Request

## 致谢

感谢以下开源项目和社区的支持：

- OpenCV：图像处理库
- TensorFlow/PyTorch：深度学习框架
- Flask：Web框架
- Bleak：蓝牙通信库
- SpeechRecognition：语音识别库

## 更新日志

### v1.0.0 (2023-10-15)
- 初始版本发布
- 实现四诊基本功能
- 集成AI模型分析
- 优化Jetson Nano性能

---

**岐伯灵枢：中医四诊AI辅助分析平台**

融合传统中医智慧与现代AI技术，为您的健康保驾护航！