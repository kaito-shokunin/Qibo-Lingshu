#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
岐伯灵枢：中医四诊AI辅助分析平台
中医分析引擎模块

适用于Jetson Nano开发板
"""

import os
import json
import logging
import base64
import requests
from datetime import datetime
import hashlib
from modules.case_retriever import CaseRetriever

# 导入知识图谱模块
from modules.knowledge_graph import TCMKnowledgeGraph

# 尝试导入requests库
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests库不可用，AI分析将受限")

from .tcm_standardizer import TCMStandardizer

logger = logging.getLogger(__name__)

class TCMAnalysisEngine:
    """中医分析引擎类"""
    
    def __init__(self, api_key=None, neo4j_config=None, base_url=None, model=None, tongue_analyzer=None):
        """初始化中医分析引擎"""
        from config import Config
        self.api_key = api_key or Config.AI_MODEL_API_KEY
        self.base_url = base_url or Config.AI_MODEL_BASE_URL
        self.model = model or Config.AI_MODEL_NAME
        self.timeout = Config.AI_MODEL_TIMEOUT
        self.max_tokens = Config.AI_MODEL_MAX_TOKENS
        self.tongue_analyzer = tongue_analyzer
        
        # 初始化知识图谱
        if neo4j_config:
            self.kg = TCMKnowledgeGraph(
                uri=neo4j_config.get('uri'),
                user=neo4j_config.get('user'),
                password=neo4j_config.get('password')
            )
        else:
            self.kg = None
            
        # 加载中医知识库
        self.tcm_knowledge = self._load_tcm_knowledge()
        
        # 分析历史记录
        self.history_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         'models', 'analysis_history.json')
        self.analysis_history = self._load_history()
        
        # 初始化案例检索器 (RAG)
        self.case_retriever = CaseRetriever(data_dirs=[
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'tcm_knowledge'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'TCMLLM-main', 'data')
        ])
        
        logger.info("中医分析引擎初始化完成")
    
    def _load_history(self):
        """加载历史记录"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"加载历史记录失败: {str(e)}")
            return []

    def _save_history(self):
        """保存历史记录到文件"""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存历史记录失败: {str(e)}")

    def _load_tcm_knowledge(self):
        """加载中医知识库"""
        try:
            knowledge_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         'models', 'tcm_knowledge', 'tcm_knowledge.json')
            
            if os.path.exists(knowledge_path):
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 默认中医知识库
                default_knowledge = {
                    "syndrome_patterns": {
                        "风寒感冒": {
                            "symptoms": ["发热", "恶寒", "头痛", "鼻塞", "流清涕", "咳嗽"],
                            "tongue": "舌淡红，苔薄白",
                            "pulse": "浮紧",
                            "treatment": "辛温解表，宣肺散寒",
                            "herbs": ["麻黄", "桂枝", "杏仁", "甘草", "生姜", "大枣"]
                        },
                        "风热感冒": {
                            "symptoms": ["发热", "微恶风寒", "头痛", "咽喉肿痛", "咳嗽", "痰黄"],
                            "tongue": "舌尖红，苔薄黄",
                            "pulse": "浮数",
                            "treatment": "辛凉解表，清热解毒",
                            "herbs": ["金银花", "连翘", "薄荷", "荆芥", "牛蒡子", "桔梗"]
                        },
                        "脾胃虚弱": {
                            "symptoms": ["食欲不振", "腹胀", "便溏", "乏力", "面色萎黄"],
                            "tongue": "舌淡胖，苔白腻",
                            "pulse": "缓弱",
                            "treatment": "健脾益气，和胃渗湿",
                            "herbs": ["党参", "白术", "茯苓", "甘草", "陈皮", "半夏"]
                        },
                        "肝郁气滞": {
                            "symptoms": ["情绪抑郁", "胸闷", "胁痛", "嗳气", "食欲不振"],
                            "tongue": "舌淡红，苔薄白",
                            "pulse": "弦",
                            "treatment": "疏肝理气，解郁止痛",
                            "herbs": ["柴胡", "白芍", "枳壳", "甘草", "香附", "川芎"]
                        },
                        "心肾不交": {
                            "symptoms": ["失眠", "多梦", "心烦", "头晕", "耳鸣", "腰膝酸软"],
                            "tongue": "舌红少苔",
                            "pulse": "细数",
                            "treatment": "滋阴降火，交通心肾",
                            "herbs": ["黄连", "阿胶", "黄芩", "白芍", "鸡子黄", "酸枣仁"]
                        },
                        "气血两虚": {
                            "symptoms": ["面色苍白", "乏力", "心悸", "气短", "自汗", "食欲不振"],
                            "tongue": "舌淡白，苔薄白",
                            "pulse": "细弱",
                            "treatment": "益气补血，健脾养心",
                            "herbs": ["人参", "黄芪", "白术", "茯苓", "当归", "川芎"]
                        },
                        "血瘀证": {
                            "symptoms": ["刺痛", "痛处固定", "面色晦暗", "口唇青紫"],
                            "tongue": "舌质紫暗，或有瘀斑",
                            "pulse": "涩脉",
                            "treatment": "活血化瘀，通络止痛",
                            "herbs": ["桃仁", "红花", "当归", "川芎", "赤芍", "生地黄"]
                        }
                    },
                    "dietary_recommendations": {
                        "血瘀证": ["山楂茶", "黑豆粥", "红糖汤"],
                        "风寒感冒": ["生姜红糖水", "葱白豆豉汤"],
                        "风热感冒": ["薄荷茶", "菊花茶"]
                    },
                    "herb_properties": {
                        "麻黄": {"性味": "辛、微苦，温", "归经": "肺、膀胱经", "功效": "发汗解表，宣肺平喘"},
                        "桂枝": {"性味": "辛、甘，温", "归经": "心、肺、膀胱经", "功效": "发汗解肌，温通经脉"},
                        "金银花": {"性味": "甘，寒", "归经": "肺、心、胃经", "功效": "清热解毒，疏散风热"},
                        "连翘": {"性味": "苦，微寒", "归经": "肺、心、胆经", "功效": "清热解毒，消肿散结"},
                        "柴胡": {"性味": "苦、辛，微寒", "归经": "肝、胆经", "功效": "和解少阳，疏肝解郁"},
                        "白芍": {"性味": "苦、酸、甘，微寒", "归经": "肝、脾经", "功效": "养血敛阴，柔肝止痛"},
                        "人参": {"性味": "甘、微苦，微温", "归经": "脾、肺、心经", "功效": "大补元气，复脉固脱"},
                        "黄芪": {"性味": "甘，温", "归经": "脾、肺经", "功效": "补气升阳，固表止汗"},
                        "当归": {"性味": "甘、辛，温", "归经": "肝、心、脾经", "功效": "补血调经，活血止痛"},
                        "甘草": {"性味": "甘，平", "归经": "心、肺、脾、胃经", "功效": "补脾益气，清热解毒"},
                        "桃仁": {"性味": "苦、甘，平", "归经": "心、肝、大肠经", "功效": "活血祛瘀，润肠通便"},
                        "红花": {"性味": "辛，温", "归经": "心、肝经", "功效": "活血通经，散瘀止痛"},
                        "赤芍": {"性味": "苦，微寒", "归经": "肝经", "功效": "清热凉血，散瘀止痛"},
                        "川芎": {"性味": "辛，温", "归经": "肝、胆、心包经", "功效": "活血行气，祛风止痛"}
                    },
                    "acupoints": {
                        "感冒": ["合谷", "风池", "大椎", "列缺"],
                        "咳嗽": ["肺俞", "尺泽", "列缺", "天突"],
                        "胃痛": ["中脘", "足三里", "内关", "胃俞"],
                        "失眠": ["神门", "内关", "三阴交", "安眠"],
                        "头痛": ["百会", "太阳", "风池", "合谷"],
                        "血瘀": ["血海", "膈俞", "三阴交", "内关"]
                    },
                    "dietary_therapy": {
                        "风寒感冒": ["生姜红糖水", "葱白豆豉汤", "紫苏叶茶"],
                        "风热感冒": ["薄荷茶", "菊花茶", "梨汁"],
                        "脾胃虚弱": ["山药粥", "莲子粥", "小米粥"],
                        "肝郁气滞": ["玫瑰花茶", "佛手茶", "陈皮茶"],
                        "心肾不交": ["莲子心茶", "百合粥", "枸杞子茶"],
                        "气血两虚": ["红枣桂圆茶", "当归羊肉汤", "黄芪鸡汤"],
                        "血瘀证": ["山楂茶", "黑豆粥", "红糖汤"]
                    }
                }
                
                # 确保目录存在
                os.makedirs(os.path.dirname(knowledge_path), exist_ok=True)
                
                # 保存默认知识库
                with open(knowledge_path, 'w', encoding='utf-8') as f:
                    json.dump(default_knowledge, f, ensure_ascii=False, indent=2)
                
                return default_knowledge
        except Exception as e:
            logger.error(f"加载中医知识库失败: {str(e)}")
            return {}
    
    def analyze_symptoms(self, tongue_image_path, vitals, patient_description):
        """综合分析所有四诊信息"""
        try:
            logger.info("开始综合分析四诊信息")
            
            # 1. 舌象分析
            tongue_analysis = None
            if self.tongue_analyzer and tongue_image_path:
                try:
                    tongue_analysis = self.tongue_analyzer.analyze_tongue(tongue_image_path)
                    logger.info(f"专门模型识别结果: {tongue_analysis.get('predicted_label')} (置信度: {tongue_analysis.get('confidence', 0):.2f})")
                except Exception as e:
                    logger.warning(f"专门舌苔模型分析失败: {str(e)}")

            # 2. 脉象分析 (基于生理指标)
            pulse_analysis = self._analyze_pulse_by_rules(vitals)

            # 3. 症状提取
            extracted_symptoms = self._extract_symptoms_from_text(patient_description)

            # 生成分析ID
            analysis_id = hashlib.md5(f"{tongue_image_path}{vitals}{patient_description}{datetime.now()}".encode()).hexdigest()
            
            # 构建中医分析提示词 (传入专门模型的识别结果)
            system_prompt, user_prompt, similar_cases = self._build_tcm_prompt(tongue_image_path, vitals, patient_description, tongue_analysis)
            
            # 调用大模型API
            if REQUESTS_AVAILABLE and self.api_key != "your_api_key_here":
                try:
                    response = self._call_model(system_prompt, user_prompt, tongue_image_path)
                    result = self._parse_response(response)
                    result["analysis_id"] = analysis_id
                    result["analysis_method"] = "AI模型"
                except Exception as e:
                    logger.error(f"AI模型分析失败: {str(e)}")
                    # 回退到规则分析
                    result = self._rule_based_analysis(tongue_image_path, vitals, patient_description)
                    result["analysis_id"] = analysis_id
                    result["analysis_method"] = "规则引擎"
            else:
                # 使用规则分析
                result = self._rule_based_analysis(tongue_image_path, vitals, patient_description)
                result["analysis_id"] = analysis_id
                result["analysis_method"] = "规则引擎"
            
            # --- 数据对齐与补全 (关键修复) ---
            
            # 确保基础字段存在
            if not result.get("diagnosis"):
                result["diagnosis"] = "未明确诊断"
            
            if not result.get("treatment_principle"):
                result["treatment_principle"] = "请咨询专业中医师进行辨证论治。"
                
            # 确保症状列表存在
            if not result.get("symptoms"):
                result["symptoms"] = extracted_symptoms
            
            # 确保各个推荐列表存在且为列表
            for field in ["herb_recommendation", "acupoint_recommendation", "dietary_recommendation", "lifestyle_recommendation", "follow_up_questions"]:
                if field not in result or result[field] is None:
                    result[field] = []
                elif isinstance(result[field], str):
                    result[field] = [result[field]]
            
            # 补全舌苔分析数据 (将专门模型的精细结果合入最终输出)
            if tongue_analysis:
                if "tongue_analysis" not in result or not isinstance(result["tongue_analysis"], dict):
                    result["tongue_analysis"] = {}
                
                # 翻译模型输出为易读格式
                display_tongue = {
                    "舌色": tongue_analysis.get("tongue_color", "未知"),
                    "苔质": tongue_analysis.get("tongue_coating", "未知"),
                    "舌形": tongue_analysis.get("tongue_shape", "未知"),
                    "润燥": tongue_analysis.get("tongue_moisture", "未知"),
                    "分类": tongue_analysis.get("predicted_label", "未知"),
                    "置信度": f"{tongue_analysis.get('confidence', 0):.2f}"
                }
                # 如果 LLM 有自己的描述，保留它
                if "description" in result["tongue_analysis"]:
                    display_tongue["中医解读"] = result["tongue_analysis"]["description"]
                
                result["tongue_analysis"] = display_tongue

            # 补全脉象分析数据
            if pulse_analysis and ("pulse_analysis" not in result or not result["pulse_analysis"]):
                result["pulse_analysis"] = {
                    "脉象类型": pulse_analysis.get("type", "平脉"),
                    "临床意义": pulse_analysis.get("condition", "正常"),
                    "实时频率": f"{pulse_analysis.get('rate', 0)} 次/分"
                }

            # 补齐字段
            if "syndrome" not in result and "diagnosis" in result:
                result["syndrome"] = result["diagnosis"]

            # 统一调用知识库核验（无论 AI 还是规则引擎）
            result = self._verify_with_kg(result, similar_cases)

            # 添加时间戳
            result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 保存分析历史
            self.analysis_history.append({
                "id": analysis_id,
                "timestamp": result["timestamp"],
                "method": result["analysis_method"],
                "summary": result.get("diagnosis", "未知诊断")
            })
            
            # 限制历史记录数量
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
            
            # 持久化保存
            self._save_history()
            
            logger.info(f"综合分析完成，方法: {result['analysis_method']}")
            return result
            
        except Exception as e:
            logger.error(f"综合分析失败: {str(e)}")
            raise
    
    def _build_tcm_prompt(self, tongue_image_path, vitals, patient_description, tongue_analysis=None):
        """构建发送给大模型的Prompt"""
        # 1. 获取RAG参考案例 (使用标准化后的文本)
        standardized_description = TCMStandardizer.standardize(patient_description)
        similar_cases = self.case_retriever.search(standardized_description)
        rag_context = self.case_retriever.format_cases_for_prompt(similar_cases)
        
        # 2. 整合舌诊和体征
        vitals_str = json.dumps(vitals, ensure_ascii=False)
        tongue_str = json.dumps(tongue_analysis, ensure_ascii=False) if tongue_analysis else "未知"
        
        # 强制要求 JSON 格式 (System 指令)
        system_prompt = """你是一位专业的中医师，请根据提供的四诊信息（望、闻、问、切）进行辨证论治。
你必须严格按照以下 JSON 格式返回，不要包含任何多余的解释文字、开场白或结束语。
你的响应必须是一个合法的 JSON 对象。

{
  "syndrome": "中医证型",
  "treatment_principle": "治则治法",
  "herb_recommendation": [
    {"name": "药材名", "dosage": "剂量", "properties": "性味", "meridian": "归经", "effects": "功效"}
  ],
  "acupoint_recommendation": ["穴位1", "穴位2"],
  "dietary_recommendation": ["建议1", "建议2"],
  "lifestyle_recommendation": ["建议1", "建议2"],
  "tongue_analysis": {"description": "舌象分析"},
  "pulse_analysis": {"description": "脉象分析"},
  "follow_up_questions": ["如果你觉得四诊信息不足，请提出1-2个针对性的追问，否则留空"]
}"""

        # 3. 检查缺失信息
        missing_info = TCMStandardizer.get_missing_critical_info(patient_description, vitals, tongue_analysis)
        missing_context = f"\n### 系统检测到可能缺失的信息 (请优先考虑追问):\n{', '.join(missing_info)}" if missing_info else ""

        user_prompt = f"""### 患者主诉/症状:
{patient_description} (专业术语参考: {standardized_description})

### 舌诊自动识别结果:
{tongue_str}

### 生理指标数据:
{vitals_str}
{missing_context}

### 参考临床病历案例 (RAG检索结果):
{rag_context}

请直接输出 JSON 内容:"""
        return system_prompt, user_prompt, similar_cases

    def _call_model(self, system_prompt, user_prompt, image_path=None):
        """调用大模型API"""
        try:
            # 智谱AI GLM-4 API 调用 (兼容 OpenAI SDK 格式)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 构建用户内容
            if "v" in self.model.lower() and image_path and os.path.exists(image_path):
                # 视觉模型：使用列表格式
                user_content = [{"type": "text", "text": user_prompt}]
                base64_image = self._encode_image(image_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            else:
                # 纯文本模型或无图像：使用字符串格式，兼容性更好
                user_content = user_prompt
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": 0.1  # 降低温度，增加输出稳定性
            }
            
            logger.info(f"发送请求到大模型: {self.model}, URL: {self.base_url}")
            # 确保 URL 拼接正确
            api_url = self.base_url
            if not api_url.endswith('/'):
                api_url += '/'
            api_url += "chat/completions"
            
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"API请求失败: {response.status_code}, {response.text}")
            
            return response.json()
        except Exception as e:
            logger.error(f"调用模型API失败: {str(e)}")
            raise
    
    def _encode_image(self, image_path):
        """将图像编码为base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"图像编码失败: {str(e)}")
            return ""
    
    def _verify_with_kg(self, result, similar_cases=None):
        """
        使用本地知识库验证 AI 结果，并增加安全性检查
        """
        try:
            diagnostic_basis = []
            syndrome = result.get("syndrome", "")
            
            # 1. 提取 RAG 检索证据
            if similar_cases:
                all_matched_tokens = set()
                for case in similar_cases:
                    if "_matched_tokens" in case:
                        all_matched_tokens.update(case["_matched_tokens"])
                
                if all_matched_tokens:
                    # 只选取前 5 个最重要的关键词
                    evidence_str = "、".join(list(all_matched_tokens)[:5])
                    diagnostic_basis.append(f"临床参考：在病案库中匹配到相似症状关键词“{evidence_str}”。")

            # 2. 检查证型是否存在于本地库
            if self.tcm_knowledge:
                patterns = self.tcm_knowledge.get("syndrome_patterns", {})
                matched_pattern = None
                for pattern_name, pattern_info in patterns.items():
                    if pattern_name in syndrome:
                        matched_pattern = pattern_info
                        diagnostic_basis.append(f"证型匹配：识别为“{pattern_name}”，符合本地知识库规范。")
                        
                        # 1. 补充中药推荐
                        if not result.get("herb_recommendation"):
                            herbs = pattern_info.get("herbs", [])
                            formatted_herbs = []
                            for h_name in herbs:
                                h_info = self.tcm_knowledge.get("herb_properties", {}).get(h_name, {})
                                formatted_herbs.append({
                                    "name": h_name,
                                    "dosage": "10g",
                                    "properties": h_info.get("性味", "见说明"),
                                    "meridian": h_info.get("归经", "见说明"),
                                    "effects": h_info.get("功效", "对症调理")
                                })
                            result["herb_recommendation"] = formatted_herbs
                            diagnostic_basis.append("药方补充：AI 未给出药方，已根据知识库推荐经典方剂。")
                        
                        # 2. 补充穴位/饮食
                        if not result.get("acupoint_recommendation"):
                            result["acupoint_recommendation"] = self.tcm_knowledge.get("acupoints", {}).get(pattern_name, [])
                        if not result.get("dietary_recommendation"):
                            result["dietary_recommendation"] = self.tcm_knowledge.get("dietary_therapy", {}).get(pattern_name, [])
                        
                        # 3. 校验治则匹配度
                        kg_treatment = pattern_info.get("treatment", "")
                        ai_treatment = result.get("treatment_principle", "")
                        if ai_treatment and kg_treatment:
                            # 简单关键词匹配
                            if not any(word in ai_treatment for word in kg_treatment[:2]): 
                                diagnostic_basis.append(f"治则校对：AI 建议“{ai_treatment}”，本地库建议“{kg_treatment}”，请综合参考。")
                        
                        if not ai_treatment or "咨询" in ai_treatment:
                            result["treatment_principle"] = kg_treatment
                        break
            
            # 2. 安全性检查
            herbs = result.get("herb_recommendation", [])
            if herbs:
                herb_names = [h["name"] if isinstance(h, dict) else str(h) for h in herbs]
                import re
                clean_names = [re.sub(r'\(.*?\)', '', name).strip() for name in herb_names]
                
                contraindications = self._check_contraindications(clean_names)
                if contraindications:
                    result["safety_warning"] = f"【安全警示】处方中疑似存在配伍禁忌：{', '.join(contraindications)}。请务必在执业中医师指导下使用！"
                    diagnostic_basis.append("安全警告：发现配伍禁忌，已触发预警。")

            # 3. 汇总诊断依据
            result["diagnostic_basis"] = diagnostic_basis if diagnostic_basis else ["根据 AI 模型深度分析得出。"]
            
            return result
        except Exception as e:
            logger.error(f"知识库核验失败: {str(e)}")
            return result

    def _check_contraindications(self, herb_names):
        """
        检查中药禁忌（十八反、十九畏、妊娠禁忌）
        """
        warnings = []
        herb_set = set(herb_names)

        # 1. 十八反
        eighteen_antagonisms = [
            ({"甘草"}, {"甘遂", "大戟", "海藻", "芫花"}, "十八反"),
            ({"乌头", "附子", "草乌", "川乌"}, {"半夏", "瓜蒌", "贝母", "白蔹", "白及"}, "十八反"),
            ({"藜芦"}, {"人参", "沙参", "丹参", "玄参", "细辛", "芍药"}, "十八反")
        ]
        
        # 2. 十八畏 (十九畏)
        nineteen_inhibitions = [
            ({"硫黄"}, {"芒硝"}, "十九畏"),
            ({"水银"}, {"砒霜"}, "十九畏"),
            ({"狼毒"}, {"密陀僧"}, "十九畏"),
            ({"巴豆"}, {"牵牛子"}, "十九畏"),
            ({"丁香"}, {"郁金"}, "十九畏"),
            ({"牙硝"}, {"三棱"}, "十九畏"),
            ({"川乌", "草乌"}, {"犀角"}, "十九畏"),
            ({"人参"}, {"五灵脂"}, "十九畏"),
            ({"官桂", "肉桂"}, {"石脂"}, "十九畏")
        ]

        # 3. 妊娠禁忌 (常见剧毒或破血药)
        pregnancy_contraindications = {
            "巴豆", "牵牛子", "大戟", "商陆", "麝香", "三棱", "莪术", "水蛭", "虻虫", "斑蝥", "雄黄", "砒霜"
        }

        # 检查反/畏
        for group1, group2, category in eighteen_antagonisms + nineteen_inhibitions:
            if herb_set.intersection(group1) and herb_set.intersection(group2):
                intersect1 = herb_set.intersection(group1)
                intersect2 = herb_set.intersection(group2)
                warnings.append(f"【{category}】{'/'.join(intersect1)} 与 {'/'.join(intersect2)} 不宜同用")
        
        # 检查妊娠禁忌
        intersect_preg = herb_set.intersection(pregnancy_contraindications)
        if intersect_preg:
            warnings.append(f"【妊娠禁忌】含有 {'/'.join(intersect_preg)}，孕妇禁用")
                
        return warnings

    def _parse_response(self, response):
        """解析API响应"""
        try:
            content = response["choices"][0]["message"]["content"]
            
            # 尝试从内容中提取 JSON (更加健壮的正则)
            import re
            # 查找第一个 '{' 和最后一个 '}' 之间的内容
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    # 预处理：去掉可能存在的 markdown 代码块标记
                    json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
                    json_str = re.sub(r'\s*```$', '', json_str, flags=re.MULTILINE)
                    
                    # 处理常见的 JSON 格式错误：末尾逗号
                    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                    
                    # 处理可能的转义双引号
                    if '\\"' in json_str:
                        json_str = json_str.replace('\\"', '"')
                    
                    # 处理可能包裹在引号内的整个 JSON 字符串
                    if json_str.startswith('"') and json_str.endswith('"'):
                        json_str = json_str[1:-1]
                    
                    result = json.loads(json_str)
                    
                    # --- 字段映射 (适配前端) ---
                    # 1. 证型与诊断 (确保两者一致)
                    if "syndrome" in result and "diagnosis" not in result:
                        result["diagnosis"] = result["syndrome"]
                    elif "diagnosis" in result and "syndrome" not in result:
                        result["syndrome"] = result["diagnosis"]
                    
                    # 2. 治疗原则 (处理不同可能的键名)
                    if "treatment_principles" in result and "treatment_principle" not in result:
                        result["treatment_principle"] = result["treatment_principles"]
                    
                    # 2. 穴位推荐
                    if "acupoints" in result and "acupoint_recommendation" not in result:
                        result["acupoint_recommendation"] = result["acupoints"]
                    
                    # 3. 饮食与生活建议 (处理旧格式 lifestyle_advice)
                    if "lifestyle_advice" in result:
                        advice = result["lifestyle_advice"]
                        if isinstance(advice, dict):
                            if "diet" in advice and "dietary_recommendation" not in result:
                                result["dietary_recommendation"] = [advice["diet"]] if isinstance(advice["diet"], str) else advice["diet"]
                            
                            lifestyle = []
                            if "rest" in advice: lifestyle.append(f"起居：{advice['rest']}")
                            if "emotion" in advice: lifestyle.append(f"情志：{advice['emotion']}")
                            if lifestyle and "lifestyle_recommendation" not in result:
                                result["lifestyle_recommendation"] = lifestyle
                    
                    # 确保字段是列表格式 (适配前端 v-for)
                    for list_field in ["dietary_recommendation", "lifestyle_recommendation", "acupoint_recommendation"]:
                        if list_field in result and isinstance(result[list_field], str):
                            result[list_field] = [result[list_field]]

                    # 4. 中药详情补充
                    if "herb_recommendation" in result and isinstance(result["herb_recommendation"], list):
                        formatted_herbs = []
                        for herb in result["herb_recommendation"]:
                            if not isinstance(herb, dict): 
                                # 处理字符串格式的药材
                                name = str(herb).split('(')[0].split()[0]
                                formatted_herbs.append({
                                    "name": name,
                                    "dosage": "适量",
                                    "properties": "见说明",
                                    "meridian": "见说明",
                                    "effects": "对症调理"
                                })
                                continue
                            
                            name = herb.get("name", "未知药材")
                            # 从本地知识库获取详细属性
                            herb_info = self.tcm_knowledge.get("herb_properties", {}).get(name, {})
                            
                            formatted_herbs.append({
                                "name": name,
                                "dosage": herb.get("dose", herb.get("dosage", herb.get("dose_info", "适量"))),
                                "properties": herb_info.get("性味", "见说明"),
                                "meridian": herb_info.get("归经", "见说明"),
                                "effects": herb_info.get("功效", herb.get("reason", "对症调理"))
                            })
                        result["herb_recommendation"] = formatted_herbs
                    
                    # 如果 AI 没给中药，尝试从 RAG 或本地知识库补全
                    if not result.get("herb_recommendation"):
                        syndrome = result.get("syndrome", "")
                        for key, info in self.tcm_knowledge.get("syndromes", {}).items():
                            if key in syndrome:
                                herbs = info.get("herbs", [])
                                formatted_herbs = []
                                for name in herbs:
                                    herb_info = self.tcm_knowledge.get("herb_properties", {}).get(name, {})
                                    formatted_herbs.append({
                                        "name": name,
                                        "dosage": "10g",
                                        "properties": herb_info.get("性味", "见说明"),
                                        "meridian": herb_info.get("归经", "见说明"),
                                        "effects": herb_info.get("功效", "对症调理")
                                    })
                                result["herb_recommendation"] = formatted_herbs
                                break

                    # 增加知识图谱核验步骤
                    # 注意：这里可能无法获取到 similar_cases，因为它是 build_prompt 时生成的
                    # 我们可以在 analyze_symptoms 中统一调用 verify_with_kg
                    return result
                except Exception as e:
                    logger.error(f"解析JSON失败: {str(e)}")
                    pass

            # 如果无法解析为 JSON，抛出异常以便回退到规则引擎
            logger.warning(f"无法从模型响应中提取 JSON。响应内容预览: {content[:100]}...")
            raise ValueError("模型响应格式错误，无法解析为 JSON")
        except Exception as e:
            logger.error(f"解析响应失败: {str(e)}")
            raise  # 重新抛出异常，触发 analyze_symptoms 中的 catch 块
    
    def _rule_based_analysis(self, tongue_image_path, vitals, patient_description):
        """基于规则的分析"""
        try:
            logger.info("使用规则引擎进行分析")
            
            # 提取症状关键词
            symptoms = self._extract_symptoms_from_text(patient_description)
            
            # 分析舌苔
            tongue_analysis = self._analyze_tongue_by_rules(tongue_image_path)
            
            # 尝试从 Neo4j 知识图谱补充舌苔知识
            kg_info = ""
            if self.kg and tongue_analysis.get("label"):
                label = tongue_analysis.get("label")
                desc = self.kg.get_coating_interpretation(label)
                if desc:
                    kg_info = f"\n【知识图谱提示】：{desc}"
                    tongue_analysis["kg_description"] = desc

            # 分析脉象
            pulse_analysis = self._analyze_pulse_by_rules(vitals)
            
            # 综合辨证
            syndrome_pattern = self._identify_syndrome_pattern(symptoms, tongue_analysis, pulse_analysis)
            
            # 获取治疗建议
            treatment = self._get_treatment_recommendation(syndrome_pattern)
            
            # 获取中药建议
            herbs = self._get_herb_recommendation(syndrome_pattern)
            
            # 获取穴位建议
            acupoints = self._get_acupoint_recommendation(syndrome_pattern)
            
            # 获取饮食建议
            dietary = self._get_dietary_recommendation(syndrome_pattern)
            
            # 获取生活建议
            lifestyle = self._get_lifestyle_recommendation(syndrome_pattern)
            
            result = {
                "diagnosis": syndrome_pattern,
                "symptoms": symptoms,
                "tongue_analysis": tongue_analysis,
                "pulse_analysis": pulse_analysis,
                "treatment_principle": treatment.get("principle", "") + kg_info,
                "herb_recommendation": herbs,
                "acupoint_recommendation": acupoints,
                "dietary_recommendation": dietary,
                "lifestyle_recommendation": lifestyle,
                "confidence": tongue_analysis.get("confidence", 0.0) if tongue_analysis else 0.0
            }
            
            return result
        except Exception as e:
            logger.error(f"规则分析失败: {str(e)}")
            return {
                "diagnosis": "分析失败",
                "symptoms": [],
                "tongue_analysis": {},
                "pulse_analysis": {},
                "treatment_principle": "",
                "herb_recommendation": [],
                "acupoint_recommendation": [],
                "dietary_recommendation": [],
                "lifestyle_recommendation": [],
                "error": str(e)
            }
    
    def _extract_symptoms_from_text(self, text):
        """从文本中提取症状"""
        try:
            # 症状关键词字典
            symptom_keywords = {
                "发热": ["发烧", "发热", "体温高", "身热", "发烧了"],
                "咳嗽": ["咳嗽", "咳", "干咳", "咳痰", "痰多", "百日咳"],
                "头痛": ["头痛", "头疼", "头晕", "头昏", "脑壳痛"],
                "乏力": ["乏力", "疲劳", "累", "没精神", "精神不振", "四肢无力", "身重"],
                "食欲不振": ["食欲不振", "没胃口", "不想吃", "吃不下", "纳差", "厌食"],
                "失眠": ["失眠", "睡不着", "入睡困难", "多梦", "易醒", "浅睡"],
                "胃痛": ["胃痛", "胃疼", "肚子痛", "腹痛", "胃胀", "嘈杂"],
                "关节痛": ["关节痛", "关节疼", "风湿", "关节炎", "骨痛", "腰痛", "腿痛"],
                "心慌": ["心慌", "心悸", "心跳快", "心跳不规律", "胸口乱跳"],
                "气短": ["气短", "气喘", "呼吸困难", "胸闷", "少气"],
                "口干": ["口干", "口渴", "口燥", "咽干", "想喝水"],
                "便秘": ["便秘", "大便干燥", "排便困难", "大便不畅", "几天没拉"],
                "腹泻": ["腹泻", "拉肚子", "大便稀", "便溏", "拉稀", "五更泻"],
                "恶心": ["恶心", "想吐", "呕吐", "反胃", "干呕"],
                "怕冷": ["怕冷", "畏寒", "恶寒", "身上冷"],
                "自汗": ["自汗", "容易出汗", "动则汗出"],
                "盗汗": ["盗汗", "半夜出汗", "睡觉出汗"],
                "咽痛": ["咽痛", "嗓子疼", "喉咙痛", "咽干"]
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
            
            return symptoms
        except Exception as e:
            logger.error(f"提取症状失败: {str(e)}")
            return []
    
    def _analyze_tongue_by_rules(self, tongue_image_path):
        """基于规则分析舌苔"""
        try:
            if self.tongue_analyzer:
                analysis = self.tongue_analyzer.analyze_tongue(tongue_image_path)
                return {
                    "color": analysis.get("tongue_color", "淡红舌"),
                    "coating": analysis.get("tongue_coating", "薄白苔"),
                    "shape": analysis.get("tongue_shape", "正常"),
                    "moisture": analysis.get("tongue_moisture", "适中"),
                    "label": analysis.get("predicted_label", "")
                }
            
            # 如果没有分析器，返回默认结果
            return {
                "color": "淡红舌",
                "coating": "薄白苔",
                "shape": "正常",
                "moisture": "适中"
            }
        except Exception as e:
            logger.error(f"舌苔分析失败: {str(e)}")
            return {}
    
    def _analyze_pulse_by_rules(self, vitals):
        """基于规则分析脉象"""
        try:
            heart_rate = vitals.get("heart_rate", 0)
            
            if heart_rate < 60:
                pulse_type = "迟脉"
                pulse_condition = "寒证"
            elif heart_rate > 90:
                pulse_type = "数脉"
                pulse_condition = "热证"
            else:
                pulse_type = "平脉"
                pulse_condition = "正常"
            
            return {
                "type": pulse_type,
                "condition": pulse_condition,
                "rate": heart_rate
            }
        except Exception as e:
            logger.error(f"脉象分析失败: {str(e)}")
            return {}
    
    def _identify_syndrome_pattern(self, symptoms, tongue_analysis, pulse_analysis):
        """识别证型"""
        try:
            # 根据症状、舌苔和脉象识别证型
            # 这里使用简单的规则匹配
            
            # 风寒感冒
            if ("发热" in symptoms or "头痛" in symptoms) and \
               tongue_analysis.get("coating") == "薄白苔" and \
               pulse_analysis.get("type") in ["浮紧", "迟脉"]:
                return "风寒感冒"
            
            # 风热感冒
            if ("发热" in symptoms or "咳嗽" in symptoms) and \
               tongue_analysis.get("coating") in ["薄黄苔", "黄苔"] and \
               pulse_analysis.get("type") in ["浮数", "数脉"]:
                return "风热感冒"
            
            # 脾胃虚弱
            if ("食欲不振" in symptoms or "乏力" in symptoms) and \
               tongue_analysis.get("color") in ["淡白舌", "淡胖舌"] and \
               pulse_analysis.get("type") in ["缓弱", "细脉"]:
                return "脾胃虚弱"
            
            # 肝郁气滞
            if ("胸闷" in symptoms or "胃痛" in symptoms) and \
               tongue_analysis.get("color") == "淡红舌" and \
               pulse_analysis.get("type") == "弦脉":
                return "肝郁气滞"
            
            # 心肾不交
            if ("失眠" in symptoms or "心慌" in symptoms) and \
               tongue_analysis.get("color") == "红舌" and \
               pulse_analysis.get("type") == "细数":
                return "心肾不交"
            
            # 气血两虚
            if ("乏力" in symptoms or "心慌" in symptoms) and \
               tongue_analysis.get("color") == "淡白舌" and \
               pulse_analysis.get("type") == "细弱":
                return "气血两虚"
            
            # 阴虚发热 (来自 TCMLLM 数据)
            if ("低热" in symptoms or "盗汗" in symptoms) and \
               tongue_analysis.get("color") == "红舌" and \
               tongue_analysis.get("coating") in ["少苔", "无苔"]:
                return "阴虚发热"
            
            # 痰湿阻肺 (来自 TCMLLM 数据)
            if ("咳嗽" in symptoms or "痰" in symptoms) and \
               tongue_analysis.get("coating") == "白腻苔":
                return "痰湿阻肺"
            
            # 血瘀证 (针对紫舌、暗舌)
            if ("紫" in tongue_analysis.get("color", "") or 
                "暗" in tongue_analysis.get("color", "") or
                "purple" in tongue_analysis.get("label", "").lower()):
                return "血瘀证"
            
            # 默认证型
            return "未明确诊断"
        except Exception as e:
            logger.error(f"证型识别失败: {str(e)}")
            return "未明确诊断"
    
    def _get_treatment_recommendation(self, syndrome_pattern):
        """获取治疗建议"""
        try:
            patterns = self.tcm_knowledge.get("syndrome_patterns", {})
            if syndrome_pattern in patterns:
                return {
                    "principle": patterns[syndrome_pattern].get("treatment", "")
                }
            else:
                return {
                    "principle": "请咨询专业中医师"
                }
        except Exception as e:
            logger.error(f"获取治疗建议失败: {str(e)}")
            return {"principle": ""}
    
    def _get_herb_recommendation(self, syndrome_pattern):
        """获取中药建议"""
        try:
            patterns = self.tcm_knowledge.get("syndrome_patterns", {})
            if syndrome_pattern in patterns:
                herbs = patterns[syndrome_pattern].get("herbs", [])
                herb_details = []
                
                for herb in herbs:
                    if herb in self.tcm_knowledge.get("herb_properties", {}):
                        herb_info = self.tcm_knowledge["herb_properties"][herb]
                        herb_details.append({
                            "name": herb,
                            "properties": herb_info.get("性味", ""),
                            "meridian": herb_info.get("归经", ""),
                            "effects": herb_info.get("功效", ""),
                            "dosage": "9-15g"  # 默认剂量
                        })
                    else:
                        herb_details.append({
                            "name": herb,
                            "properties": "",
                            "meridian": "",
                            "effects": "",
                            "dosage": "9-15g"
                        })
                
                return herb_details
            else:
                return []
        except Exception as e:
            logger.error(f"获取中药建议失败: {str(e)}")
            return []
    
    def _get_acupoint_recommendation(self, syndrome_pattern):
        """获取穴位建议"""
        try:
            # 简化处理，根据症状推荐穴位
            acupoints = []
            
            if "感冒" in syndrome_pattern:
                acupoints.extend(["合谷", "风池", "大椎", "列缺"])
            elif "咳嗽" in syndrome_pattern:
                acupoints.extend(["肺俞", "尺泽", "列缺", "天突"])
            elif "胃痛" in syndrome_pattern:
                acupoints.extend(["中脘", "足三里", "内关", "胃俞"])
            elif "失眠" in syndrome_pattern:
                acupoints.extend(["神门", "内关", "三阴交", "安眠"])
            elif "头痛" in syndrome_pattern:
                acupoints.extend(["百会", "太阳", "风池", "合谷"])
            
            # 去重
            return list(set(acupoints))
        except Exception as e:
            logger.error(f"获取穴位建议失败: {str(e)}")
            return []
    
    def _get_dietary_recommendation(self, syndrome_pattern):
        """获取饮食建议"""
        try:
            patterns = self.tcm_knowledge.get("dietary_therapy", {})
            if syndrome_pattern in patterns:
                return patterns[syndrome_pattern]
            else:
                return ["清淡饮食", "多喝水", "避免辛辣刺激性食物"]
        except Exception as e:
            logger.error(f"获取饮食建议失败: {str(e)}")
            return []
    
    def _get_lifestyle_recommendation(self, syndrome_pattern):
        """获取生活建议"""
        try:
            recommendations = []
            
            # 通用建议
            recommendations.extend(["保持规律作息", "适度运动", "保持心情舒畅"])
            
            # 根据证型添加特定建议
            if "感冒" in syndrome_pattern:
                recommendations.extend(["注意保暖", "避免受凉", "多休息"])
            elif "脾胃虚弱" in syndrome_pattern:
                recommendations.extend(["饮食有节", "避免生冷食物", "细嚼慢咽"])
            elif "肝郁气滞" in syndrome_pattern:
                recommendations.extend(["调节情绪", "避免压力过大", "适当放松"])
            elif "失眠" in syndrome_pattern:
                recommendations.extend(["睡前避免刺激性活动", "保持安静环境", "规律作息"])
            
            return recommendations
        except Exception as e:
            logger.error(f"获取生活建议失败: {str(e)}")
            return []
    
    def get_analysis_history(self):
        """获取分析历史"""
        try:
            return self.analysis_history
        except Exception as e:
            logger.error(f"获取分析历史失败: {str(e)}")
            return []
    
    def save_analysis_result(self, result):
        """保存分析结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.json"
            analysis_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       'models', 'tcm_knowledge', filename)
            
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分析结果已保存: {analysis_path}")
            return analysis_path
        except Exception as e:
            logger.error(f"保存分析结果失败: {str(e)}")
            return None