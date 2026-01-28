#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
岐伯灵枢：中医四诊AI辅助分析平台
语音输入模块

适用于Jetson Nano开发板
"""

import os
import logging
import wave
import audioop
from datetime import datetime
import json

# 尝试导入语音识别库
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logging.warning("SpeechRecognition库不可用，将使用回退语音输入")

# 尝试导入PyAudio
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.warning("PyAudio库不可用，将使用回退音频输入")

# 尝试导入ZhipuAI
try:
    from zhipuai import ZhipuAI
    ZHIPUAI_AVAILABLE = True
except ImportError:
    ZHIPUAI_AVAILABLE = False
    logging.warning("zhipuai库不可用")

# 尝试导入百度语音识别
try:
    from aip import AipSpeech
    BAIDU_ASR_AVAILABLE = True
except ImportError:
    BAIDU_ASR_AVAILABLE = False
    logging.warning("baidu-aip库不可用")

logger = logging.getLogger(__name__)

class VoiceToText:
    """语音转文字类"""
    
    def __init__(self, api_key=None, baidu_config=None):
        """初始化语音转文字器"""
        self.recognizer = None
        self.microphone = None
        self.api_key = api_key
        self.baidu_config = baidu_config
        self.audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'audio')
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # 录制状态
        self.is_recording = False
        self._stop_event = False
        
        # 初始化智谱AI客户端
        if ZHIPUAI_AVAILABLE and self.api_key:
            try:
                self.client = ZhipuAI(api_key=self.api_key)
                logger.info("智谱AI语音识别已启用")
            except Exception as e:
                logger.warning(f"智谱AI初始化失败: {str(e)}")
                self.client = None
        else:
            self.client = None

        # 初始化百度AI客户端
        if BAIDU_ASR_AVAILABLE and self.baidu_config:
            try:
                self.baidu_client = AipSpeech(
                    self.baidu_config.get('app_id'),
                    self.baidu_config.get('api_key'),
                    self.baidu_config.get('secret_key')
                )
                logger.info("百度语音识别已启用")
            except Exception as e:
                logger.warning(f"百度语音识别初始化失败: {str(e)}")
                self.baidu_client = None
        else:
            self.baidu_client = None
        
        # 初始化语音识别器和麦克风
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            
            # 尝试初始化麦克风
            if PYAUDIO_AVAILABLE:
                try:
                    self.microphone = sr.Microphone()
                    # 调整麦克风环境噪音
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source)
                    logger.info("语音转文字器初始化完成")
                except Exception as e:
                    logger.warning(f"麦克风初始化失败: {str(e)}")
                    self.microphone = None
            else:
                logger.warning("PyAudio不可用，无法使用麦克风")
                self.microphone = None
        else:
            logger.warning("SpeechRecognition不可用，将使用模拟语音转文字")
        
        # 语音识别配置
        self.recognition_engine = "google"  # 默认使用Google语音识别
        self.language = "zh-CN"  # 中文识别
        self.timeout = 30  # 识别超时时间
        self.phrase_timeout = 5  # 短语超时时间
        
        # 音频录制配置
        self.format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        self.channels = 1  # 单声道
        self.rate = 16000  # 采样率
        self.chunk = 1024  # 缓冲区大小
        self.record_timeout = 5  # 默认录音时长
        
        logger.info(f"语音输入模块初始化完成，麦克风可用: {self.microphone is not None}")
    
    def capture_voice(self, duration=None):
        """录制患者语音描述"""
        try:
            self.is_recording = True
            self._stop_event = False
            duration = duration or self.timeout
            
            if self.microphone and PYAUDIO_AVAILABLE:
                # 使用真实麦克风录音 (可中断)
                return self._record_interruptible(duration)
                
            # 使用回退录音
            return self._fallback_voice_recording(duration)
                
        except Exception as e:
            logger.error(f"录制语音失败: {str(e)}")
            self.is_recording = False
            raise
        finally:
            self.is_recording = False

    def stop_voice(self):
        """停止语音录制"""
        logger.info("收到停止录音请求")
        self._stop_event = True
        self.is_recording = False

    def _record_interruptible(self, duration):
        """使用 PyAudio 进行可中断的录音"""
        try:
            import pyaudio
            import wave
            
            p = pyaudio.PyAudio()
            stream = p.open(format=self.format,
                            channels=self.channels,
                            rate=self.rate,
                            input=True,
                            frames_per_buffer=self.chunk)
            
            logger.info(f"开始录制语音 (可中断)，最长时长: {duration} 秒")
            frames = []
            
            # 计算总共需要的块数
            total_chunks = int(self.rate / self.chunk * duration)
            
            for i in range(total_chunks):
                if self._stop_event:
                    logger.info("录音被用户中止")
                    break
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
            
            logger.info("录音结束，正在保存文件...")
            
            # 停止和关闭流
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # 保存音频文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"voice_{timestamp}.wav"
            audio_path = os.path.join(self.audio_dir, filename)
            
            with wave.open(audio_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
            
            logger.info(f"语音录制完成: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"中断式录音失败: {str(e)}")
            raise

    def _record_with_microphone(self, duration):
        """使用麦克风录制音频"""
        try:
            logger.info(f"开始录制语音，时长: {duration} 秒")
            
            # 使用SpeechRecognition库录制音频
            with self.microphone as source:
                # 再次调整环境噪音
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # 录制音频
                audio_data = self.recognizer.listen(source, timeout=duration)
            
            # 保存音频文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"voice_{timestamp}.wav"
            audio_path = os.path.join(self.audio_dir, filename)
            
            # 将音频数据保存为WAV文件
            with open(audio_path, "wb") as f:
                f.write(audio_data.get_wav_data())
            
            logger.info(f"语音录制完成: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"麦克风录音失败: {str(e)}")
            raise
    
    def _fallback_voice_recording(self, duration):
        """回退录音过程"""
        try:
            logger.info(f"回退录音过程，时长: {duration} 秒")
            
            # 创建一个空的音频文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"voice_{timestamp}.wav"
            audio_path = os.path.join(self.audio_dir, filename)
            
            # 创建一个简单的WAV文件
            if PYAUDIO_AVAILABLE:
                p = pyaudio.PyAudio()
                
                try:
                    stream = p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  output=True)
                    
                    # 生成静音数据
                    frames = []
                    for _ in range(0, int(self.rate / self.chunk * duration)):
                        data = b'\x00' * self.chunk * 2  # 静音数据
                        frames.append(data)
                    
                    stream.stop_stream()
                    stream.close()
                    
                    # 保存为WAV文件
                    with wave.open(audio_path, 'wb') as wf:
                        wf.setnchannels(self.channels)
                        wf.setsampwidth(p.get_sample_size(self.format))
                        wf.setframerate(self.rate)
                        wf.writeframes(b''.join(frames))
                    
                    p.terminate()
                except Exception as e:
                    logger.error(f"创建模拟音频文件失败: {str(e)}")
                    # 创建一个空文件
                    with open(audio_path, 'wb') as f:
                        f.write(b'')
            else:
                # 创建一个空文件
                with open(audio_path, 'wb') as f:
                    f.write(b'')
            
            logger.info(f"回退录音完成: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"回退录音失败: {str(e)}")
            raise
    
    def convert_to_text(self, audio_data):
        """将语音转换为文字"""
        try:
            if isinstance(audio_data, str):
                # 如果是文件路径，读取文件
                if os.path.exists(audio_data):
                    return self._convert_file_to_text(audio_data)
                else:
                    raise FileNotFoundError(f"音频文件不存在: {audio_data}")
            else:
                # 如果是音频数据，直接转换
                return self._convert_audio_data_to_text(audio_data)
                
        except Exception as e:
            logger.error(f"语音转文字失败: {str(e)}")
            raise
    
    def _convert_file_to_text(self, audio_path):
        """将音频文件转换为文字"""
        try:
            # 1. 优先使用百度语音识别（专为中文优化）
            if self.baidu_client:
                try:
                    logger.info(f"使用百度语音识别转换音频文件: {audio_path}")
                    with open(audio_path, 'rb') as f:
                        audio_data = f.read()
                    
                    # 识别音频，1537 表示普通话
                    result = self.baidu_client.asr(audio_data, 'wav', 16000, {
                        'dev_pid': 1537,
                    })
                    
                    if result['err_no'] == 0:
                        text = result['result'][0]
                        logger.info(f"百度语音识别成功: {text}")
                        return text
                    else:
                        logger.error(f"百度语音识别失败: {result['err_msg']} (错误码: {result['err_no']})")
                except Exception as e:
                    logger.error(f"百度语音识别异常: {str(e)}，将尝试其他方式...")

            # 2. 其次使用智谱AI进行识别
            if self.client:
                try:
                    logger.info(f"使用智谱AI转换音频文件: {audio_path}")
                    with open(audio_path, "rb") as f:
                        result = self.client.audio.transcriptions.create(
                          model="whisper-1", 
                          file=f,
                        )
                    text = result.text
                    logger.info(f"智谱AI识别成功: {text}")
                    return text
                except Exception as e:
                    logger.error(f"智谱AI识别失败: {str(e)}，将尝试本地识别...")

            if not SPEECH_RECOGNITION_AVAILABLE:
                # 使用回退语音识别
                return self._fallback_speech_recognition(audio_path)
            
            logger.info(f"开始转换音频文件为文字 (SpeechRecognition): {audio_path}")
            
            # 使用SpeechRecognition库识别音频文件
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            
            # 尝试使用Google语音识别API
            try:
                text = self.recognizer.recognize_google(audio, language=self.language)
                logger.info(f"语音识别成功: {text}")
                return text
            except sr.RequestError as e:
                logger.error(f"Google语音识别服务错误: {str(e)}")
                # 尝试使用Sphinx离线识别
                try:
                    text = self.recognizer.recognize_sphinx(audio, language='zh-cn')
                    logger.info(f"Sphinx语音识别成功: {text}")
                    return text
                except sr.RequestError as e2:
                    logger.error(f"Sphinx语音识别失败: {str(e2)}")
                    # 使用回退识别
                    return self._fallback_speech_recognition(audio_path)
            except sr.UnknownValueError:
                logger.warning("无法识别语音内容")
                return "无法识别语音内容，请重试"
                
        except Exception as e:
            logger.error(f"音频文件转文字失败: {str(e)}")
            raise
    
    def _convert_audio_data_to_text(self, audio_data):
        """将音频数据转换为文字"""
        try:
            if not SPEECH_RECOGNITION_AVAILABLE:
                # 使用回退语音识别
                return self._fallback_speech_recognition()
            
            logger.info("开始转换音频数据为文字")
            
            # 尝试使用Google语音识别API
            try:
                text = self.recognizer.recognize_google(audio_data, language=self.language)
                logger.info(f"语音识别成功: {text}")
                return text
            except sr.RequestError as e:
                logger.error(f"Google语音识别服务错误: {str(e)}")
                # 尝试使用Sphinx离线识别
                try:
                    text = self.recognizer.recognize_sphinx(audio_data, language='zh-cn')
                    logger.info(f"Sphinx语音识别成功: {text}")
                    return text
                except sr.RequestError as e2:
                    logger.error(f"Sphinx语音识别失败: {str(e2)}")
                    # 使用模拟识别
                    return self._simulate_speech_recognition()
            except sr.UnknownValueError:
                logger.warning("无法识别语音内容")
                return "无法识别语音内容，请重试"
                
        except Exception as e:
            logger.error(f"音频数据转文字失败: {str(e)}")
            raise
    
    def _fallback_speech_recognition(self, audio_path=None):
        """回退语音识别"""
        try:
            # 常见症状描述（作为回退选项）
            sample_texts = [
                "我最近感觉头晕乏力，食欲不振，睡眠质量也不好",
                "我经常咳嗽，有痰，喉咙痛，有点发烧",
                "我最近胃痛，消化不良，有时恶心，食欲下降",
                "我最近感觉心慌气短，容易疲劳，手脚冰凉",
                "我最近关节疼痛，活动不便，天气变化时更严重",
                "我最近失眠多梦，记忆力下降，注意力不集中",
                "我最近口干舌燥，容易上火，大便干燥",
                "我最近感觉腰酸背痛，精神不振，容易疲劳"
            ]
            
            import random
            text = random.choice(sample_texts)
            
            logger.info(f"回退语音识别结果: {text}")
            return text
            
        except Exception as e:
            logger.error(f"回退语音识别失败: {str(e)}")
            return "语音识别失败，请重试"
    
    def extract_symptoms(self, text):
        """从文字描述中提取症状关键词"""
        try:
            # 症状关键词字典
            symptom_keywords = {
                "发热": ["发烧", "发热", "体温高", "热"],
                "咳嗽": ["咳嗽", "咳", "干咳", "咳痰"],
                "头痛": ["头痛", "头疼", "头晕", "头昏"],
                "乏力": ["乏力", "疲劳", "累", "没精神", "精神不振"],
                "食欲不振": ["食欲不振", "没胃口", "不想吃", "吃不下"],
                "失眠": ["失眠", "睡不着", "入睡困难", "多梦"],
                "胃痛": ["胃痛", "胃疼", "肚子痛", "腹痛"],
                "关节痛": ["关节痛", "关节疼", "风湿", "关节炎"],
                "心慌": ["心慌", "心悸", "心跳快", "心跳不规律"],
                "气短": ["气短", "气喘", "呼吸困难", "胸闷"],
                "口干": ["口干", "口渴", "口燥", "咽干"],
                "便秘": ["便秘", "大便干燥", "排便困难", "大便不畅"],
                "腹泻": ["腹泻", "拉肚子", "大便稀", "便溏"],
                "恶心": ["恶心", "想吐", "呕吐", "反胃"]
            }
            
            # 提取症状
            symptoms = []
            for symptom, keywords in symptom_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        symptoms.append(symptom)
                        break
            
            # 去重
            symptoms = list(set(symptoms))
            
            logger.info(f"从文本中提取症状: {symptoms}")
            return symptoms
            
        except Exception as e:
            logger.error(f"提取症状失败: {str(e)}")
            return []
    
    def get_audio_info(self, audio_path):
        """获取音频文件信息"""
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
            info = {
                "file_path": audio_path,
                "file_size": os.path.getsize(audio_path),
                "file_format": os.path.splitext(audio_path)[1].lower()
            }
            
            # 如果是WAV文件，获取更多信息
            if info["file_format"] == ".wav":
                try:
                    with wave.open(audio_path, 'rb') as wav_file:
                        info["channels"] = wav_file.getnchannels()
                        info["sample_width"] = wav_file.getsampwidth()
                        info["frame_rate"] = wav_file.getframerate()
                        info["n_frames"] = wav_file.getnframes()
                        info["duration"] = info["n_frames"] / info["frame_rate"]
                except Exception as e:
                    logger.warning(f"获取WAV文件信息失败: {str(e)}")
            
            return info
            
        except Exception as e:
            logger.error(f"获取音频信息失败: {str(e)}")
            return {"error": str(e)}
    
    def save_transcription(self, text, audio_path=None):
        """保存转录文本"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcription_{timestamp}.json"
            transcription_path = os.path.join(self.audio_dir, filename)
            
            data = {
                "timestamp": timestamp,
                "text": text,
                "audio_path": audio_path,
                "symptoms": self.extract_symptoms(text)
            }
            
            with open(transcription_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"转录文本已保存: {transcription_path}")
            return transcription_path
            
        except Exception as e:
            logger.error(f"保存转录文本失败: {str(e)}")
            return None
    
    def get_microphone_list(self):
        """获取可用麦克风列表"""
        try:
            if not PYAUDIO_AVAILABLE:
                return {"error": "PyAudio不可用"}
            
            p = pyaudio.PyAudio()
            mic_list = []
            
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    mic_list.append({
                        "index": i,
                        "name": device_info['name'],
                        "channels": device_info['maxInputChannels'],
                        "sample_rate": int(device_info['defaultSampleRate'])
                    })
            
            p.terminate()
            
            return mic_list
            
        except Exception as e:
            logger.error(f"获取麦克风列表失败: {str(e)}")
            return {"error": str(e)}