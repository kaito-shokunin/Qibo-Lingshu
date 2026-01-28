#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
岐伯灵枢：中医四诊AI辅助分析平台
配置文件

适用于Jetson Nano开发板
"""

import os
from datetime import timedelta

class Config:
    """应用基础配置"""
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'qibo_lingshu_secret_key_for_jetson_nano'
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB最大上传大小
    
    # 摄像头配置 - 适用于Jetson Nano
    CAMERA_INDEX = 0  # 默认摄像头索引，CSI摄像头通常为0，USB摄像头可能为1或更高
    CAMERA_RESOLUTION = (640, 480)  # 降低分辨率以适应Jetson Nano性能
    CAMERA_FRAMERATE = 15  # 降低帧率以减少处理负担
    
    # 图像保存路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    AUDIO_FOLDER = os.path.join(BASE_DIR, 'audio')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # 模型路径
    TONGUE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'tongue_analysis')
    TCM_KNOWLEDGE_PATH = os.path.join(BASE_DIR, 'models', 'tcm_knowledge')
    
    # 蓝牙配置 - 适用于Jetson Nano
    BLUETOOTH_ADAPTER = "hci0"  # Jetson Nano默认蓝牙适配器
    BLUETOOTH_TIMEOUT = 10  # 蓝牙连接超时时间(秒)
    
    # 大模型API配置 (推荐使用 智谱AI 或 DeepSeek)
    AI_MODEL_API_KEY = os.environ.get('AI_MODEL_API_KEY') or 'f17fa3f9e1b548a09f877f7ba606fec6.WjX5rEDFvPOt6w42'
    # 智谱AI URL: https://open.bigmodel.cn/api/paas/v4/
    # DeepSeek URL: https://api.deepseek.com
    # 本地 Ollama URL: http://localhost:11434/v1
    AI_MODEL_BASE_URL = os.environ.get('AI_MODEL_BASE_URL') or "https://open.bigmodel.cn/api/paas/v4/"
    AI_MODEL_NAME = os.environ.get('AI_MODEL_NAME') or "glm-4-air" # 使用资源包支持的 glm-4-air
    AI_MODEL_TIMEOUT = 60  # API请求超时时间(秒)
    AI_MODEL_MAX_TOKENS = 2000  # 最大生成token数
    
    # 语音识别配置
    SPEECH_RECOGNITION_ENGINE = "google"  # 使用Google语音识别API
    SPEECH_RECOGNITION_LANGUAGE = "zh-CN"  # 中文识别
    AUDIO_RECORD_DURATION = 5  # 默认录音时长(秒)
    AUDIO_SAMPLE_RATE = 16000  # 音频采样率
    AUDIO_CHANNELS = 1  # 单声道
    
    # Jetson Nano性能优化配置
    THREAD_POOL_SIZE = 2  # 线程池大小，根据Jetson Nano核心数调整
    GPU_MEMORY_FRACTION = 0.5  # 分配给TensorFlow的GPU内存比例
    ENABLE_GPU_ACCELERATION = True  # 启用GPU加速
    
    # 中医诊断配置
    TONGUE_ANALYSIS_CONFIDENCE_THRESHOLD = 0.6  # 舌苔分析置信度阈值
    VITALS_NORMAL_RANGES = {
        "heart_rate": {"min": 60, "max": 100},  # 心率正常范围
        "blood_oxygen": {"min": 95, "max": 100},  # 血氧正常范围
        "blood_pressure_systolic": {"min": 90, "max": 140},  # 收缩压正常范围
        "blood_pressure_diastolic": {"min": 60, "max": 90},  # 舒张压正常范围
        "temperature": {"min": 36.0, "max": 37.3}  # 体温正常范围
    }
    
    # 会话配置
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    
    # 日志配置
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or "INFO"
    LOG_FILE = os.path.join(BASE_DIR, 'qibo_lingshu.log')
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # 安全配置
    WTF_CSRF_ENABLED = False  # 简化开发，生产环境建议启用
    CORS_ORIGINS = ["*"]  # 允许所有来源，生产环境应限制
    
    # 智能手环兼容配置
    SUPPORTED_DEVICE_TYPES = {
        "xiaomi": ["Mi Band", "Mi Smart Band"],
        "huawei": ["Huawei Band", "Honor Band"],
        "generic": ["BLE Heart Rate Monitor", "BLE Fitness Tracker"]
    }
    
    # 缓存配置
    CACHE_TYPE = "simple"  # 简单内存缓存
    CACHE_DEFAULT_TIMEOUT = 300  # 默认缓存超时时间(秒)

    # Neo4j 知识图谱配置
    NEO4J_URI = os.environ.get('NEO4J_URI') or "bolt://localhost:7687"
    NEO4J_USER = os.environ.get('NEO4J_USER') or "neo4j"
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD') or "12345678"

    # 百度语音识别配置
    BAIDU_APP_ID = '120447611'
    BAIDU_API_KEY = 't0lYNSg0ofXMOEUe6Sff6R5M'
    BAIDU_SECRET_KEY = 'b8KjJMyprTMAYFiMlV6iet4ZUi4hpsqB'

    # 开发/生产环境标识
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1', 'yes']
    TESTING = os.environ.get('TESTING', 'False').lower() in ['true', '1', 'yes']

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    LOG_LEVEL = "WARNING"
    CORS_ORIGINS = ["http://localhost:5000", "http://127.0.0.1:5000"]

class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    DEBUG = True
    WTF_CSRF_ENABLED = False

# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}