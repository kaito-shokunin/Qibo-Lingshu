import re

class TCMStandardizer:
    """中医术语标准化工具：将口语化描述转化为专业术语"""
    
    # 口语 -> 专业术语 映射表
    SYNONYM_MAP = {
        "肚子痛": "腹痛",
        "肚子疼": "腹痛",
        "拉肚子": "腹泻",
        "拉稀": "便溏",
        "发烧": "发热",
        "烧了": "发热",
        "没胃口": "纳呆",
        "不想吃饭": "纳差",
        "睡不着": "不寐",
        "多梦": "失眠多梦",
        "口渴": "口燥咽干",
        "怕冷": "畏寒",
        "出虚汗": "自汗",
        "半夜出汗": "盗汗",
        "嗓子疼": "咽喉肿痛",
        "心慌": "心悸",
        "气不够用": "气短",
        "没力气": "乏力",
        "浑身没劲": "肢体倦怠",
        "头晕": "眩晕",
        "大便干": "便秘",
        "小便黄": "尿黄",
        "眼睛红": "目赤"
    }
    
    @classmethod
    def standardize(cls, text):
        """对文本进行标准化处理"""
        if not text:
            return ""
        
        standardized_text = text
        for colloquial, professional in cls.SYNONYM_MAP.items():
            standardized_text = standardized_text.replace(colloquial, professional)
            
        return standardized_text

    @classmethod
    def extract_standard_terms(cls, text):
        """提取文本中包含的标准术语"""
        standard_terms = []
        # 这里可以使用更复杂的正则或知识库匹配
        # 目前先基于映射表反向提取
        for colloquial, professional in cls.SYNONYM_MAP.items():
            if colloquial in text or professional in text:
                standard_terms.append(professional)
        
        return list(set(standard_terms))

    @classmethod
    def get_missing_critical_info(cls, text, vitals, tongue_analysis):
        """检查缺失的关键信息"""
        missing = []
        
        # 1. 检查是否提及口渴、二便、睡眠等中医“十问”
        critical_categories = {
            "饮食/口渴": ["渴", "饮", "食", "纳", "饿"],
            "睡眠": ["睡", "眠", "梦"],
            "大小便": ["便", "尿", "溺"],
            "寒热": ["冷", "热", "烧", "寒"]
        }
        
        for category, keywords in critical_categories.items():
            if not any(k in text for k in keywords):
                missing.append(category)
                
        # 2. 检查生理指标（脉诊基础）
        if not vitals or vitals.get("heart_rate", 0) == 0:
            missing.append("脉率(心率)")
            
        # 3. 检查舌诊
        if not tongue_analysis:
            missing.append("舌象照片")
            
        return missing
