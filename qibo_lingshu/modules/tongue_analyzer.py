#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
岐伯灵枢：中医四诊AI辅助分析平台
舌苔分析模块

适用于Jetson Nano开发板
"""

import cv2
import os
import numpy as np
from PIL import Image
import logging
from datetime import datetime
import json
import base64

# 尝试导入PyTorch，如果不可用则使用传统的计算机视觉和专家规则进行分析
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    import logging
    logging.warning(f"PyTorch不可用或加载失败 ({str(e)})，将使用传统的计算机视觉和专家规则进行舌苔分析")

logger = logging.getLogger(__name__)

class TongueAnalyzer:
    """舌苔分析类"""
    
    def __init__(self):
        """初始化舌苔分析器"""
        self.model = None
        self.transform = None
        self.tongue_knowledge = self._load_tongue_knowledge()
        self.thresholds = self._load_thresholds()
        
        # 定义本项目特有的 6 种舌苔分类
        self.coating_labels = [
            "black tongue coating",
            "map tongue coating",
            "purple tongue coating",
            "red tongue yellow fur thick greasy fur",
            "The red tongue is thick and greasy",
            "The white tongue is thick and greasy"
        ]
        
        # 初始化模型
        if TORCH_AVAILABLE:
            self.model = self._load_tongue_model()
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        logger.info("舌苔分析器初始化完成")

    def _load_thresholds(self):
        """加载诊断阈值（支持自我学习）"""
        try:
            threshold_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         'models', 'tongue_analysis', 'thresholds.json')
            
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 默认阈值（从代码逻辑提取）
                default_thresholds = {
                    "tongue_color": {
                        "淡白舌": {"s_max": 30, "v_min": 150},
                        "红舌": {"h_ranges": [[0, 10], [170, 180]], "s_min": 50, "a_min": 20},
                        "紫舌": {"h_range": [140, 165]}
                    },
                    "tongue_coating": {
                        "黑苔": {"v_max": 60},
                        "黄厚苔": {"h_max": 30, "s_min": 40, "v_min": 100},
                        "黄苔": {"h_max": 30, "s_min": 40},
                        "白厚苔": {"v_min": 160, "s_max": 30}
                    },
                    "stats": {
                        "sample_count": 0,
                        "last_updated": datetime.now().isoformat()
                    }
                }
                return default_thresholds
        except Exception as e:
            logger.error(f"加载阈值失败: {str(e)}")
            return {}

    def add_feedback(self, image_path, corrected_result):
        """
        添加用户反馈以进行自我学习
        :param image_path: 图像路径
        :param corrected_result: 用户修正后的诊断结果字典
        """
        try:
            if not os.path.exists(image_path):
                return False
                
            image = Image.open(image_path).convert('RGB')
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            
            mean_hsv = np.mean(hsv, axis=(0, 1))
            mean_lab = np.mean(lab, axis=(0, 1))
            
            # 更新阈值统计
            self._update_thresholds(corrected_result, mean_hsv, mean_lab)
            
            # 保存到反馈数据集
            feedback_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        'models', 'tongue_analysis', 'feedback_data')
            os.makedirs(feedback_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feedback_file = os.path.join(feedback_dir, f"feedback_{timestamp}.json")
            
            feedback_data = {
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
                "corrected_result": corrected_result,
                "features": {
                    "hsv": mean_hsv.tolist(),
                    "lab": mean_lab.tolist()
                }
            }
            
            with open(feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"成功记录用户反馈，系统已学习新样本: {image_path}")
            return True
        except Exception as e:
            logger.error(f"记录反馈失败: {str(e)}")
            return False

    def _update_thresholds(self, result, hsv, lab):
        """根据反馈数据动态调整阈值（简单的移动平均学习逻辑）"""
        try:
            h, s, v = hsv
            l, a, b = lab
            
            # 学习速率 (随着样本增加逐渐减小，保证稳定性)
            count = self.thresholds.get("stats", {}).get("sample_count", 0)
            alpha = 1.0 / (count + 10.0) # 初始学习率较大
            
            # 更新舌头颜色阈值
            color = result.get("tongue_color")
            if color == "淡白舌":
                self.thresholds["tongue_color"]["淡白舌"]["s_max"] = (1-alpha) * self.thresholds["tongue_color"]["淡白舌"]["s_max"] + alpha * s
                self.thresholds["tongue_color"]["淡白舌"]["v_min"] = (1-alpha) * self.thresholds["tongue_color"]["淡白舌"]["v_min"] + alpha * v
            elif color == "红舌":
                self.thresholds["tongue_color"]["红舌"]["s_min"] = (1-alpha) * self.thresholds["tongue_color"]["红舌"]["s_min"] + alpha * s
                self.thresholds["tongue_color"]["红舌"]["a_min"] = (1-alpha) * self.thresholds["tongue_color"]["红舌"]["a_min"] + alpha * a
            
            # 更新舌苔阈值
            coating = result.get("tongue_coating")
            if coating:
                if coating == "黑苔":
                    self.thresholds["tongue_coating"]["黑苔"]["v_max"] = (1-alpha) * self.thresholds["tongue_coating"]["黑苔"]["v_max"] + alpha * v
                elif "黄" in coating:
                    self.thresholds["tongue_coating"]["黄苔"]["h_max"] = (1-alpha) * self.thresholds["tongue_coating"]["黄苔"]["h_max"] + alpha * h
                    self.thresholds["tongue_coating"]["黄苔"]["s_min"] = (1-alpha) * self.thresholds["tongue_coating"]["黄苔"]["s_min"] + alpha * s
            
            # 更新统计信息
            self.thresholds["stats"]["sample_count"] = count + 1
            self.thresholds["stats"]["last_updated"] = datetime.now().isoformat()
            
            # 保存更新后的阈值
            threshold_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         'models', 'tongue_analysis', 'thresholds.json')
            with open(threshold_path, 'w', encoding='utf-8') as f:
                json.dump(self.thresholds, f, ensure_ascii=False, indent=2)
                
            logger.info(f"阈值已根据反馈自动优化 (样本总数: {count + 1})")
        except Exception as e:
            logger.error(f"更新阈值失败: {str(e)}")
        
    def _load_tongue_model(self):
        """加载舌苔分析模型"""
        try:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      'models', 'tongue_analysis', 'tongue_model.pth')
            
            if os.path.exists(model_path):
                logger.info(f"正在从 {model_path} 加载 PyTorch 舌苔模型...")
                
                # 导入 ResNet50 架构 (与 train_tongue.py 同步)
                from torchvision import models
                import torch.nn as nn
                
                # 初始化 ResNet50
                model = models.resnet50(pretrained=False)
                num_classes = 6
                num_ftrs = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_ftrs, num_classes)
                )
                
                # 加载权重
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    # 同时同步类别名称
                    if 'classes' in checkpoint:
                        self.coating_labels = checkpoint['classes']
                        logger.info(f"已同步类别名称: {self.coating_labels}")
                    elif 'class_names' in checkpoint:
                        self.coating_labels = checkpoint['class_names']
                        logger.info(f"已同步类别名称: {self.coating_labels}")
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                
                # 如果 GPU 可用且系统支持，移动到 GPU
                if torch.cuda.is_available():
                    model = model.cuda()
                    logger.info("舌苔分析模型已加载到 GPU")
                else:
                    logger.info("舌苔分析模型已加载到 CPU")
                    
                return model
            else:
                logger.warning(f"未找到舌苔模型文件: {model_path}，将使用规则引擎分析")
                return None
        except Exception as e:
            logger.error(f"加载舌苔模型失败: {str(e)}")
            return None
    
    def _load_tongue_knowledge(self):
        """加载舌苔知识库"""
        try:
            knowledge_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         'models', 'tcm_knowledge', 'tongue_diagnosis.json')
            
            if os.path.exists(knowledge_path):
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 默认舌苔知识库
                default_knowledge = {
                    "tongue_color": {
                        "淡红舌": "气血调和，常见于健康人或轻症",
                        "红舌": "热证，常见于实热或虚热",
                        "淡白舌": "气血两虚，常见于虚证",
                        "紫舌": "血瘀，常见于血瘀证",
                        "绛舌": "热盛伤津，常见于热入营血"
                    },
                    "tongue_coating": {
                        "薄白苔": "表证或寒证",
                        "白厚苔": "寒湿或痰湿",
                        "黄苔": "热证",
                        "黄厚苔": "里热炽盛",
                        "少苔或无苔": "阴虚或气血两虚"
                    },
                    "tongue_shape": {
                        "正常": "健康状态",
                        "胖大舌": "脾肾阳虚或水湿内停",
                        "齿痕舌": "脾虚或湿盛",
                        "瘦薄舌": "气血两虚或阴虚火旺",
                        "裂纹舌": "阴液亏虚或血虚不润"
                    },
                    "tongue_moisture": {
                        "适中": "津液正常",
                        "湿润": "寒湿或痰湿",
                        "干燥": "热盛伤津或阴虚",
                        "光剥": "胃阴枯竭"
                    }
                }
                
                # 确保目录存在
                os.makedirs(os.path.dirname(knowledge_path), exist_ok=True)
                
                # 保存默认知识库
                with open(knowledge_path, 'w', encoding='utf-8') as f:
                    json.dump(default_knowledge, f, ensure_ascii=False, indent=2)
                
                return default_knowledge
        except Exception as e:
            logger.error(f"加载舌苔知识库失败: {str(e)}")
            return {}
    
    def _safe_imwrite(self, path, img):
        """安全地保存图像，支持中文路径"""
        try:
            ext = os.path.splitext(path)[1]
            result, nparray = cv2.imencode(ext, img)
            if result:
                with open(path, 'wb') as f:
                    f.write(nparray.tobytes())
                return True
            return False
        except Exception as e:
            logger.error(f"安全保存图像失败: {str(e)}")
            return False

    def capture_image(self, camera_index=None, frame=None):
        """采集舌苔图像 (支持直接传入帧或通过摄像头采集)"""
        try:
            if frame is None:
                # 使用配置中的摄像头索引或传入的索引
                cam_index = camera_index or 0
                
                # 初始化摄像头
                cap = cv2.VideoCapture(cam_index)
                
                if not cap.isOpened():
                    # 尝试其他摄像头索引
                    for i in range(1, 5):
                        cap = cv2.VideoCapture(i)
                        if cap.isOpened():
                            logger.info(f"使用摄像头索引 {i}")
                            break
                    else:
                        raise Exception("无法打开任何摄像头")
                
                # 设置摄像头参数
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                
                # 读取多帧以确保图像稳定
                for _ in range(5):
                    ret, frame = cap.read()
                
                if not ret:
                    raise Exception("无法捕获图像")
                
                cap.release()
            
            # 保存图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tongue_{timestamp}.jpg"
            upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, filename)
            
            # 图像预处理
            processed_frame = self._preprocess_image(frame)
            
            # 检查处理后的图像是否有效
            if processed_frame is None or processed_frame.size == 0:
                logger.warning("预处理后图像为空，使用原始帧")
                processed_frame = frame
                
            success = self._safe_imwrite(image_path, processed_frame)
            if not success:
                # 尝试使用原始文件名，避免路径中可能的字符问题
                alt_filename = f"t_{timestamp}.jpg"
                image_path = os.path.join(upload_dir, alt_filename)
                success = self._safe_imwrite(image_path, processed_frame)
                
            if not success:
                raise Exception(f"无法保存图像到: {image_path}")
            
            logger.info(f"舌苔图像已保存: {image_path}")
            return image_path
        except Exception as e:
            logger.error(f"采集舌苔图像失败: {str(e)}")
            raise
    
    def _preprocess_image(self, frame):
        """预处理图像以提高分析质量"""
        try:
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 定义舌头的HSV范围
            lower_tongue = np.array([0, 20, 50], dtype=np.uint8)
            upper_tongue = np.array([20, 255, 255], dtype=np.uint8)
            
            # 创建掩码
            mask = cv2.inRange(hsv, lower_tongue, upper_tongue)
            
            # 应用形态学操作去除噪声
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 找到最大轮廓（假设为舌头）
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 找到最大轮廓
                max_contour = max(contours, key=cv2.contourArea)
                
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(max_contour)
                
                # 稍微扩大边界矩形以确保包含整个舌头
                margin = 20
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(frame.shape[1] - x, w + 2 * margin)
                h = min(frame.shape[0] - y, h + 2 * margin)
                
                # 裁剪舌头区域
                tongue_region = frame[y:y+h, x:x+w]
                
                return tongue_region
            
            # 如果没有找到舌头，返回原始图像
            return frame
        except Exception as e:
            logger.error(f"图像预处理失败: {str(e)}")
            return frame
    
    def analyze_tongue(self, image_path):
        """分析舌苔图像"""
        try:
            # 增加重试机制，防止磁盘IO延迟导致文件检测不到
            import time
            for _ in range(3):
                if os.path.exists(image_path):
                    break
                time.sleep(0.2)
                
            if not os.path.exists(image_path):
                # 记录详细的目录信息以便排查
                upload_dir = os.path.dirname(image_path)
                dir_exists = os.path.exists(upload_dir)
                dir_content = os.listdir(upload_dir) if dir_exists else "Dir not exists"
                logger.error(f"文件缺失详情 - 路径: {image_path}, 目录是否存在: {dir_exists}, 目录内容: {dir_content}")
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            
            # 如果有模型，使用模型分析
            if self.model and TORCH_AVAILABLE:
                return self._analyze_with_model(image)
            else:
                # 使用传统的计算机视觉和专家规则进行分析
                return self._analyze_with_rules(image)
        except Exception as e:
            logger.error(f"舌苔分析失败: {str(e)}")
            raise
    
    def _analyze_with_model(self, image):
        """使用深度学习模型分析舌苔"""
        try:
            # 1. 首先运行传统规则引擎获取基础特征（如形状、水分等模型可能覆盖不全的细节）
            # 这确保了即使模型主要关注分类，细微的特征检测也不会丢失
            base_result = self._analyze_with_rules(image)
            
            # 2. 准备模型推理
            image_tensor = self.transform(image).unsqueeze(0)
            
            # 如果有GPU可用，使用GPU
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
                self.model.cuda()
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(image_tensor)
            
            # 3. 解析模型输出
            model_result = self._parse_model_output(outputs)

            # 4. 清理显存 (针对 Jetson Nano 等边缘设备优化)
            if torch.cuda.is_available():
                del image_tensor
                del outputs
                torch.cuda.empty_cache()
            
            # 5. 融合结果：专门训练的 ResNet50 模型在分类上更鲁棒，将其作为主要判断依据
            combined_result = base_result.copy()
            
            # 更新模型预测的核心分类和置信度
            combined_result.update({
                "predicted_label": model_result["predicted_label"],
                "confidence": model_result["confidence"],
                "analysis_method": "ResNet50深度学习模型+专家规则融合"
            })
            
            # 如果模型给出了具体的颜色、苔质等映射，则覆盖规则引擎的结果
            # 逻辑：模型在分类（颜色、苔质）上更准；在细节特征（形状、水分）上，如果模型给出了非默认值，则信任模型，否则保留规则引擎的检测结果
            for key in ["tongue_color", "tongue_coating", "tongue_shape", "tongue_moisture", "interpretation"]:
                if key in model_result and model_result[key]:
                    if key in ["tongue_shape", "tongue_moisture"]:
                        # 只有当模型给出了明确的异常判断时才覆盖规则引擎的结果
                        if model_result[key] not in ["正常", "适中", "分析中..."]:
                             combined_result[key] = model_result[key]
                    else:
                        # 对于颜色和苔质，优先信任模型预测
                        combined_result[key] = model_result[key]
            
            logger.info(f"融合诊断完成: 分类={combined_result['predicted_label']}, 置信度={combined_result['confidence']:.4f}")
            return combined_result
        except Exception as e:
            logger.error(f"模型分析失败: {str(e)}")
            # 回退到规则分析
            return self._analyze_with_rules(image)
    
    def _apply_awb(self, img):
        """应用自动白平衡校准 (灰度世界算法)"""
        try:
            # 灰度世界算法假设图像的平均颜色是灰色的
            # 这在舌诊图像中非常有用，可以抵消环境光的影响
            b, g, r = cv2.split(img)
            m_b = np.mean(b)
            m_g = np.mean(g)
            m_r = np.mean(r)
            m_gray = (m_b + m_g + m_r) / 3
            
            # 避免除以零
            if m_b == 0 or m_g == 0 or m_r == 0:
                return img
                
            k_b = m_gray / m_b
            k_g = m_gray / m_g
            k_r = m_gray / m_r
            
            b = cv2.convertScaleAbs(b, alpha=k_b)
            g = cv2.convertScaleAbs(g, alpha=k_g)
            r = cv2.convertScaleAbs(r, alpha=k_r)
            
            return cv2.merge([b, g, r])
        except Exception as e:
            logger.error(f"自动白平衡校准失败: {str(e)}")
            return img

    def _analyze_with_rules(self, image):
        """使用图像处理和规则分析舌苔"""
        try:
            # 转换为OpenCV格式
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 应用自动白平衡校准
            cv_image = self._apply_awb(cv_image)
            
            # 转换为HSV和LAB颜色空间进行分析
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            
            # 计算平均颜色
            mean_hsv = np.mean(hsv, axis=(0, 1))
            mean_lab = np.mean(lab, axis=(0, 1))
            
            # 分析舌头颜色
            tongue_color = self._analyze_tongue_color(mean_hsv, mean_lab)
            
            # 分析舌苔
            tongue_coating = self._analyze_tongue_coating(hsv)
            
            # 分析舌头形状
            tongue_shape = self._analyze_tongue_shape(cv_image)
            
            # 分析舌头湿润度
            tongue_moisture = self._analyze_tongue_moisture(lab)
            
            # 映射到 Neo4j 中的 6 种分类标签
            predicted_label = "The white tongue is thick and greasy" # 默认值
            
            if tongue_color == "紫舌":
                predicted_label = "purple tongue coating"
            elif tongue_coating == "黑苔":
                predicted_label = "black tongue coating"
            elif tongue_moisture == "光剥":
                predicted_label = "map tongue coating"
            elif tongue_color == "红舌":
                if "黄" in tongue_coating:
                    predicted_label = "red tongue yellow fur thick greasy fur"
                else:
                    predicted_label = "The red tongue is thick and greasy"
            
            # 执行本体映射，获取标准中医术语
            standard_info = self._map_label_to_standard(predicted_label)
            
            # 获取对应的临床意义（优先从本地知识库获取）
            interpretation = self.tongue_knowledge.get("tongue_coating", {}).get(tongue_coating, "舌象正常")
            
            return {
                "tongue_color": standard_info["color"] if standard_info["color"] != "未知" else tongue_color,
                "tongue_coating": standard_info["coating"] if standard_info["coating"] != "未知" else tongue_coating,
                "tongue_shape": tongue_shape,
                "tongue_moisture": tongue_moisture,
                "predicted_label": predicted_label,
                "kg_node": standard_info["kg_node"],
                "interpretation": interpretation,
                "confidence": 0.65  # 规则引擎基础置信度
            }
        except Exception as e:
            logger.error(f"规则分析失败: {str(e)}")
            # 返回默认分析结果
            return {
                "tongue_color": "淡红舌",
                "tongue_coating": "薄白苔",
                "tongue_shape": "正常",
                "tongue_moisture": "适中",
                "predicted_label": "The white tongue is thick and greasy",
                "interpretation": "舌淡红，苔薄白，为健康舌象",
                "confidence": 0.4
            }

    def _analyze_tongue_color(self, mean_hsv, mean_lab):
        """分析舌头颜色 (基于动态阈值)"""
        h, s, v = mean_hsv
        l, a, b = mean_lab
        
        tc = self.thresholds.get("tongue_color", {})
        
        # 1. 淡白舌判断
        dw = tc.get("淡白舌", {"s_max": 30, "v_min": 150})
        if s < dw.get("s_max", 30) and v > dw.get("v_min", 150):
            return "淡白舌"
            
        # 2. 红舌判断
        hs = tc.get("红舌", {"h_ranges": [[0, 10], [170, 180]], "s_min": 50, "a_min": 20})
        is_red_h = any(r[0] <= h <= r[1] for r in hs.get("h_ranges", [[0, 10], [170, 180]]))
        if (is_red_h and s > hs.get("s_min", 50)) or a > hs.get("a_min", 20):
            return "红舌"
            
        # 3. 紫舌判断
        zs = tc.get("紫舌", {"h_range": [140, 165]})
        hr = zs.get("h_range", [140, 165])
        if hr[0] <= h <= hr[1]:
            return "紫舌"
            
        return "淡红舌"

    def _analyze_tongue_coating(self, hsv):
        """分析舌苔 (基于动态阈值)"""
        # 计算图像亮度分布
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]
        h_channel = hsv[:, :, 0]
        
        mean_v = np.mean(v_channel)
        mean_s = np.mean(s_channel)
        mean_h = np.mean(h_channel)
        
        tk = self.thresholds.get("tongue_coating", {})
        
        # 1. 黑苔判断
        ht = tk.get("黑苔", {"v_max": 60})
        if mean_v < ht.get("v_max", 60):
            return "黑苔"
            
        # 2. 黄苔系列判断
        yt = tk.get("黄苔", {"h_max": 30, "s_min": 40})
        if mean_h < yt.get("h_max", 30) and mean_s > yt.get("s_min", 40):
            yht = tk.get("黄厚苔", {"v_min": 100})
            if mean_v > yht.get("v_min", 100):
                return "黄厚苔"
            else:
                return "黄苔"
                
        # 3. 白苔系列判断
        bt = tk.get("白厚苔", {"v_min": 160, "s_max": 30})
        if mean_v > bt.get("v_min", 160):
            if mean_s < bt.get("s_max", 30):
                return "白厚苔"
            else:
                return "薄白苔"
                
        return "薄白苔"
    
    def _analyze_tongue_shape(self, cv_image):
        """分析舌头形状"""
        # 转换为灰度图
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大轮廓
            max_contour = max(contours, key=cv2.contourArea)
            
            # 计算轮廓特征
            area = cv2.contourArea(max_contour)
            perimeter = cv2.arcLength(max_contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # 计算边界矩形
            x, y, w, h = cv2.boundingRect(max_contour)
            aspect_ratio = float(w) / h
            
            # 基于形状特征判断舌头形状
            if aspect_ratio > 1.5:
                return "胖大舌"
            elif circularity < 0.7:
                return "齿痕舌"
            elif aspect_ratio < 1.0:
                return "瘦薄舌"
            else:
                return "正常"
        else:
            return "正常"
    
    def _analyze_tongue_moisture(self, lab):
        """分析舌头湿润度"""
        # 计算LAB中的L通道（亮度）标准差
        l_channel = lab[:, :, 0]
        std_dev = np.std(l_channel)
        
        # 计算平均亮度
        mean_l = np.mean(l_channel)
        
        # 基于亮度和标准差判断湿润度
        if std_dev > 40:
            return "光剥"
        elif std_dev > 25:
            return "干燥"
        elif mean_l > 180:
            return "湿润"
        else:
            return "适中"
    
    def _map_label_to_standard(self, label):
        """将模型输出的原始标签映射为标准中医术语和知识图谱节点名"""
        mapping = {
            "black tongue coating": {
                "color": "焦紫舌", 
                "coating": "黑苔", 
                "kg_node": "black tongue coating"
            },
            "map tongue coating": {
                "color": "红舌", 
                "coating": "剥苔", 
                "kg_node": "map tongue coating"
            },
            "purple tongue coating": {
                "color": "紫舌", 
                "coating": "薄白苔", 
                "kg_node": "purple tongue coating"
            },
            "red tongue yellow fur thick greasy fur": {
                "color": "红舌", 
                "coating": "黄厚腻苔", 
                "kg_node": "red tongue yellow fur thick greasy fur"
            },
            "The red tongue is thick and greasy": {
                "color": "红舌", 
                "coating": "红舌厚腻苔", 
                "kg_node": "The red tongue is thick and greasy"
            },
            "The white tongue is thick and greasy": {
                "color": "淡红舌", 
                "coating": "白厚腻苔", 
                "kg_node": "The white tongue is thick and greasy"
            }
        }
        return mapping.get(label, {
            "color": "未知", 
            "coating": "未知", 
            "kg_node": label
        })

    def _parse_model_output(self, outputs):
        """解析模型输出并进行本体映射"""
        try:
            import torch.nn.functional as F
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # 获取原始标签
            raw_label = self.coating_labels[predicted_idx.item()]
            
            # 执行本体映射
            standard_info = self._map_label_to_standard(raw_label)
            
            res = {
                "predicted_label": raw_label,
                "kg_node": standard_info["kg_node"],
                "tongue_color": standard_info["color"],
                "tongue_coating": standard_info["coating"],
                "tongue_shape": "正常",
                "tongue_moisture": "适中",
                "confidence": confidence.item(),
                "interpretation": ""
            }
            
            # 补充详细解释
            if raw_label == "black tongue coating":
                res.update({"interpretation": "里证、寒极或热极。苔色越深，病情越重。"})
            elif raw_label == "map tongue coating":
                res.update({"tongue_moisture": "光剥", "tongue_shape": "裂纹舌", "interpretation": "剥苔的一种，多为阴虚、气血两虚。"})
            elif raw_label == "purple tongue coating":
                res.update({"interpretation": "主血瘀、寒凝或热毒。舌质紫暗，提示血液运行不畅。"})
            elif raw_label == "red tongue yellow fur thick greasy fur":
                res.update({"interpretation": "里博热证，湿热内蕴，痰饮化热。"})
            elif raw_label == "The red tongue is thick and greasy":
                res.update({"interpretation": "湿热内蕴，痰热内阻。常见于湿热体质。"})
            elif raw_label == "The white tongue is thick and greasy":
                res.update({"interpretation": "寒湿内停，痰饮内阻。提示脾失健运。"})
            
            return res
            
        except Exception as e:
            logger.error(f"解析模型输出失败: {str(e)}")
            return {
                "tongue_color": "分析中...",
                "tongue_coating": "分析中...",
                "predicted_label": "Error",
                "confidence": 0.0
            }
    
    def extract_symptoms(self, analysis_result):
        """从舌苔分析中提取中医症状名称列表"""
        try:
            symptoms = []
            
            # 提取舌头颜色
            tongue_color = analysis_result.get("tongue_color", "")
            if tongue_color and tongue_color != "淡红舌":
                symptoms.append(tongue_color)
            
            # 提取舌苔
            tongue_coating = analysis_result.get("tongue_coating", "")
            if tongue_coating and tongue_coating != "薄白苔":
                symptoms.append(tongue_coating)
            
            # 提取舌头形状
            tongue_shape = analysis_result.get("tongue_shape", "")
            if tongue_shape and tongue_shape != "正常":
                symptoms.append(tongue_shape)
            
            # 提取舌头湿润度
            tongue_moisture = analysis_result.get("tongue_moisture", "")
            if tongue_moisture and tongue_moisture != "适中":
                symptoms.append(tongue_moisture)
            
            # 如果什么都没有，至少返回一个"舌象正常"
            if not symptoms:
                symptoms.append("舌象正常")
                
            logger.info(f"提取舌苔症状: {symptoms}")
            return symptoms
        except Exception as e:
            logger.error(f"提取舌苔症状失败: {str(e)}")
            return ["分析异常"]
    
    def get_image_base64(self, image_path):
        """获取图像的base64编码"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            logger.error(f"获取图像base64编码失败: {str(e)}")
            return ""