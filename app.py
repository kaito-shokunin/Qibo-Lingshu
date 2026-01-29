#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
岐伯灵枢：中医四诊AI辅助分析平台
主应用入口文件

适用于Jetson Nano开发板
"""

import os
import sys
import logging
from datetime import datetime

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import traceback

# 导入自定义模块
from modules.tongue_analyzer import TongueAnalyzer
from modules.health_collector import HealthDataCollector
from modules.voice_input import VoiceToText
from modules.tcm_engine import TCMAnalysisEngine
from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qibo_lingshu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)  # 允许跨域请求

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio'), exist_ok=True)

# 初始化各模块
try:
    logger.info("初始化舌苔分析模块...")
    tongue_analyzer = TongueAnalyzer()
    
    logger.info("初始化健康数据采集模块...")
    health_collector = HealthDataCollector()
    
    logger.info("初始化语音输入模块...")
    baidu_config = {
        'app_id': app.config['BAIDU_APP_ID'],
        'api_key': app.config['BAIDU_API_KEY'],
        'secret_key': app.config['BAIDU_SECRET_KEY']
    }
    voice_input = VoiceToText(
        api_key=app.config['AI_MODEL_API_KEY'],
        baidu_config=baidu_config
    )
    
    logger.info("初始化中医分析引擎...")
    neo4j_config = {
        'uri': app.config['NEO4J_URI'],
        'user': app.config['NEO4J_USER'],
        'password': app.config['NEO4J_PASSWORD']
    }
    tcm_engine = TCMAnalysisEngine(
        api_key=app.config['AI_MODEL_API_KEY'],
        neo4j_config=neo4j_config,
        base_url=app.config['AI_MODEL_BASE_URL'],
        model=app.config['AI_MODEL_NAME'],
        tongue_analyzer=tongue_analyzer
    )
    
    logger.info("所有模块初始化完成")
except Exception as e:
    logger.error(f"模块初始化失败: {str(e)}")
    logger.error(traceback.format_exc())

import cv2
import numpy as np
import threading
import time

class VideoCamera:
    def __init__(self):
        self.camera = None
        self.last_frame = None
        self.keep_running = False
        self.lock = threading.Lock()
        self.thread = None
        self.backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY

    def start(self):
        with self.lock:
            if self.keep_running:
                return True
            
            logger.info("正在开启摄像头...")
            self.camera = cv2.VideoCapture(app.config['CAMERA_INDEX'], self.backend)
            
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(app.config['CAMERA_INDEX'])
                
            if not self.camera.isOpened():
                logger.error("无法打开摄像头")
                return False
                
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.keep_running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            logger.info("摄像头已开启")
            return True

    def _update(self):
        while self.keep_running:
            if self.camera is None or not self.camera.isOpened():
                break
            success, frame = self.camera.read()
            if success and frame is not None:
                with self.lock:
                    self.last_frame = frame.copy()
            else:
                logger.warning("摄像头读取失败，尝试重连...")
                with self.lock:
                    self.last_frame = None
                self.camera.release()
                time.sleep(1)
                self.camera = cv2.VideoCapture(app.config['CAMERA_INDEX'], self.backend)
            time.sleep(0.03)

    def get_frame(self):
        with self.lock:
            if not self.keep_running:
                return False, None
            if self.last_frame is not None:
                return True, self.last_frame.copy()
            return False, None

    def stop(self):
        with self.lock:
            logger.info("正在关闭摄像头...")
            self.keep_running = False
            if self.camera:
                self.camera.release()
                self.camera = None
            self.last_frame = None
            logger.info("摄像头已关闭")

# 全局摄像头对象
video_camera = VideoCamera()

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """开启摄像头"""
    if video_camera.start():
        return jsonify({"status": "success", "message": "摄像头已开启"})
    return jsonify({"status": "error", "message": "无法开启摄像头"}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """关闭摄像头"""
    video_camera.stop()
    return jsonify({"status": "success", "message": "摄像头已关闭"})

@app.route('/camera_status')
def camera_status():
    """获取摄像头状态"""
    return jsonify({"active": video_camera.keep_running})

# 创建Flask应用
@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

def gen_frames():
    """视频流生成器"""
    while True:
        if not video_camera.keep_running:
            # 如果摄像头没开启，返回一个“摄像头未开启”的占位图
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera Closed", (180, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1) # 降低占位图刷新频率
            continue

        success, frame = video_camera.get_frame()
        if not success:
            time.sleep(0.1)
            continue
        else:
            # 在预览中添加时间戳
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_tongue', methods=['POST'])
def capture_tongue():
    """采集舌苔图像"""
    try:
        logger.info(">>> 收到舌苔采集请求")
        
        # 确保摄像头已开启
        if not video_camera.keep_running:
            if not video_camera.start():
                raise Exception("无法自动开启摄像头，请手动开启")
            time.sleep(1) # 等待摄像头稳定
            
        # 使用当前视频流的最后一帧
        success, frame = video_camera.get_frame()
        
        if not success:
            logger.error("无法获取摄像头帧")
            raise Exception("无法从摄像头读取画面，请确保摄像头已连接且未被其他程序占用")
            
        logger.info("成功获取画面帧，开始保存和分析...")
        image_path = tongue_analyzer.capture_image(frame=frame)
        analysis_result = tongue_analyzer.analyze_tongue(image_path)
        
        # 尝试通过知识图谱丰富分析结果
        predicted_label = analysis_result.get("predicted_label")
        if predicted_label and hasattr(tcm_engine, 'kg') and tcm_engine.kg:
            kg_interpretation = tcm_engine.kg.get_coating_interpretation(predicted_label)
            if kg_interpretation:
                logger.info(f"从知识图谱获取到临床意义: {predicted_label}")
                analysis_result["interpretation"] = kg_interpretation
        
        symptoms = tongue_analyzer.extract_symptoms(analysis_result)
        
        logger.info(f"舌苔图像采集成功: {image_path}")
        return jsonify({
            "status": "success",
            "image_path": image_path,
            "symptoms": symptoms,
            "analysis": analysis_result
        })
    except Exception as e:
        logger.error(f"舌苔图像采集失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})

@app.route('/tongue_feedback', methods=['POST'])
def tongue_feedback():
    """处理舌苔分析反馈（自我学习）"""
    try:
        data = request.json
        image_path = data.get('image_path')
        corrected_result = data.get('corrected_result')
        
        if not image_path or not corrected_result:
            return jsonify({"status": "error", "message": "参数缺失"}), 400
            
        # 1. 本地模型自我学习 (更新阈值和保存JSON)
        success = tongue_analyzer.add_feedback(image_path, corrected_result)
        
        # 2. 知识图谱动态更新
        if success and hasattr(tcm_engine, 'kg') and tcm_engine.kg:
            # 提取特征用于知识图谱存储
            try:
                import cv2
                import numpy as np
                from PIL import Image
                img = Image.open(image_path).convert('RGB')
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
                features = {
                    "mean_hsv": np.mean(hsv, axis=(0, 1)).tolist(),
                    "mean_lab": np.mean(lab, axis=(0, 1)).tolist()
                }
                tcm_engine.kg.record_feedback(image_path, corrected_result, features)
            except Exception as e:
                logger.error(f"知识图谱更新失败: {str(e)}")
        
        if success:
            return jsonify({"status": "success", "message": "感谢您的反馈，系统已完成自我学习并更新知识库"})
        else:
            return jsonify({"status": "error", "message": "反馈处理失败"}), 500
    except Exception as e:
        logger.error(f"反馈处理异常: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/collect_health_data', methods=['POST'])
def collect_health_data():
    """采集健康数据"""
    try:
        device_mac = request.json.get('device_mac')
        logger.info(f"开始采集健康数据，设备MAC: {device_mac}")
        
        import asyncio
        
        async def run_collection():
            try:
                # 1. 连接设备
                connected = await health_collector.connect_device(device_mac)
                if not connected:
                    return {"status": "error", "message": "无法连接到设备"}
                
                # 2. 采集数据
                vitals = await health_collector.collect_vitals()
                
                # 3. 异常检测
                abnormalities = health_collector.detect_abnormalities(vitals)
                
                return {
                    "status": "success",
                    "vitals": vitals,
                    "abnormalities": abnormalities
                }
            finally:
                # 4. 无论成功失败，都断开连接以释放资源
                await health_collector.disconnect()

        # 使用新的事件循环运行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_collection())
            logger.info(f"健康数据采集任务完成: {result}")
            return jsonify(result)
        finally:
            # 在关闭循环前，先给一点时间让 Bleak 的后台任务清理
            loop.run_until_complete(asyncio.sleep(0.5))
            loop.close()
            
    except Exception as e:
        logger.error(f"健康数据采集失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})

@app.route('/voice_input', methods=['POST'])
def voice_input_handler():
    """处理语音输入"""
    try:
        logger.info("开始处理语音输入")
        
        # 检查是否有文件上传
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename != '':
                # 保存音频文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"voice_{timestamp}.wav"
                audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio', filename)
                audio_file.save(audio_path)
                
                # 转换为文字
                text = voice_input.convert_to_text(audio_path)
                
                logger.info(f"语音转文字成功: {text}")
                return jsonify({
                    "status": "success",
                    "text": text,
                    "audio_path": audio_path
                })
        
        # 如果没有文件上传，使用麦克风录音
        audio_path = voice_input.capture_voice()
        text = voice_input.convert_to_text(audio_path)
        
        logger.info(f"语音转文字成功: {text}")
        return jsonify({
            "status": "success",
            "text": text,
            "audio_path": audio_path
        })
    except Exception as e:
        logger.error(f"语音输入处理失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop_voice', methods=['POST'])
def stop_voice():
    """停止语音录制"""
    voice_input.stop_voice()
    return jsonify({"status": "success", "message": "已发送停止录音信号"})

@app.route('/analyze', methods=['POST'])
def analyze():
    """综合分析四诊信息"""
    try:
        data = request.json
        tongue_image = data.get('tongue_image')
        vitals = data.get('vitals')
        patient_description = data.get('patient_description')
        
        logger.info("开始综合分析四诊信息")
        
        result = tcm_engine.analyze_symptoms(tongue_image, vitals, patient_description)
        
        logger.info("综合分析完成")
        return jsonify({
            "status": "success",
            "result": result
        })
    except Exception as e:
        logger.error(f"综合分析失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传文件的访问"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/audio/<filename>')
def audio_file(filename):
    """提供音频文件的访问"""
    return send_from_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio'), filename)

@app.route('/history', methods=['GET'])
def get_history():
    """获取诊断历史记录"""
    try:
        return jsonify({
            "status": "success",
            "history": tcm_engine.analysis_history
        })
    except Exception as e:
        logger.error(f"获取历史记录失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/system_info')
def system_info():
    """获取系统信息"""
    try:
        import platform
        import psutil
        
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
        
        return jsonify({
            "status": "success",
            "info": info
        })
    except Exception as e:
        logger.error(f"获取系统信息失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "页面未找到"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"服务器内部错误: {str(error)}")
    return jsonify({"status": "error", "message": "服务器内部错误"}), 500

if __name__ == '__main__':
    logger.info("启动岐伯灵枢：中医四诊AI辅助分析平台")
    logger.info(f"上传目录: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"模型路径: {app.config['TONGUE_MODEL_PATH']}")
    
    # 在Jetson Nano上，建议使用0.0.0.0以便从其他设备访问
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)