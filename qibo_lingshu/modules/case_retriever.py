import json
import os
import re
import logging
import math
from collections import Counter, defaultdict
from .tcm_standardizer import TCMStandardizer

logger = logging.getLogger(__name__)

class CaseRetriever:
    """轻量级中医案例检索器 (支持语义增强与术语标准化)"""
    
    def __init__(self, data_dirs=None):
        self.cases = []
        self.inverted_index = defaultdict(list)  # 倒排索引: {token: [case_ids]}
        self.idf = {}  # 逆文档频率
        if data_dirs:
            for d in data_dirs:
                self.load_from_directory(d)
        
    def load_from_directory(self, directory):
        """从目录加载 JSON/JSONL 文件"""
        if not os.path.exists(directory):
            logger.warning(f"目录不存在: {directory}")
            return
            
        start_idx = len(self.cases)
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    self._load_json(os.path.join(root, file))
                elif file.endswith('.jsonl'):
                    self._load_jsonl(os.path.join(root, file))
        
        # 加载完成后构建/更新索引
        self._build_index(start_idx)
        logger.info(f"案例库加载完成，共 {len(self.cases)} 条案例，索引构建完毕")

    def _load_json(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # 针对 JSON-LD 格式的特殊处理 (中国中医科学院数据)
                    for item in data:
                        if not isinstance(item, dict): continue
                        
                        processed_item = {}
                        # 1. 处理方剂数据
                        if "http://localhost/formula/名称" in item:
                            processed_item = {
                                "type": "formula",
                                "name": item.get("http://localhost/formula/名称", [{}])[0].get("@value", "未知方剂"),
                                "content": item.get("http://localhost/formula/主治", [{}])[0].get("@value", ""),
                                "summary": item.get("http://localhost/formula/药物组成", [{}])[0].get("@value", ""),
                                "usage": item.get("http://localhost/formula/用法用量", [{}])[0].get("@value", ""),
                                "contraindication": item.get("http://localhost/formula/用药禁忌", [{}])[0].get("@value", "")
                            }
                        # 2. 处理中药数据
                        elif any(k.startswith("http://www.example.com/tcm/herb#") for k in item.keys()):
                            # 从 @id 提取名称
                            name_uri = item.get("@id", "")
                            name = name_uri.split("#")[-1].strip() if "#" in name_uri else "未知中药"
                            processed_item = {
                                "type": "herb",
                                "name": name,
                                "content": item.get("http://www.example.com/tcm/herb#主治", [{}])[0].get("@value", ""),
                                "summary": item.get("http://www.example.com/tcm/herb#功效", [{}])[0].get("@value", ""),
                                "property": item.get("http://www.example.com/tcm/herb#药性", [{}])[0].get("@value", "")
                            }
                        
                        if processed_item:
                            self.cases.append(processed_item)
                        else:
                            # 原始格式
                            self.cases.append(item)
                else:
                    self.cases.append(data)
        except Exception as e:
            logger.error(f"加载 JSON 失败 {path}: {e}")

    def _load_jsonl(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.cases.append(json.loads(line))
        except Exception as e:
            logger.error(f"加载 JSONL 失败 {path}: {e}")

    def _get_tokens(self, text):
        """分词：提取中文关键词和常用中医术语"""
        if not text: return []
        # 提取中文词组和单字（排除标点和空格）
        tokens = re.findall(r'[\u4e00-\u9fa5]+', text)
        result = []
        for t in tokens:
            # 加入 2-4 字词组
            for i in range(len(t)-1):
                result.append(t[i:i+2])
                if i+3 <= len(t): result.append(t[i:i+3])
            # 加入单字作为补充
            result.extend(list(t))
        return result

    def _build_index(self, start_idx=0):
        """构建倒排索引和计算 IDF"""
        total_docs = len(self.cases)
        doc_freq = Counter()
        
        for i in range(start_idx, total_docs):
            case = self.cases[i]
            content = (case.get('Content', '') or case.get('content', '') or '') + \
                      (case.get('Summary', '') or case.get('summary', '') or '')
            tokens = set(self._get_tokens(content))
            for token in tokens:
                self.inverted_index[token].append(i)
                doc_freq[token] += 1
        
        # 计算 IDF
        for token, freq in doc_freq.items():
            self.idf[token] = math.log(total_docs / (freq + 1)) + 1

    def search(self, query_text, top_k=5):
        """使用倒排索引、术语标准化和 TF-IDF 进行检索"""
        if not query_text or not self.cases:
            return []
            
        # 1. 术语标准化 (痛点修复：解决口语化与专业术语不匹配问题)
        standardized_query = TCMStandardizer.standardize(query_text)
        logger.info(f"原始查询: {query_text} -> 标准化查询: {standardized_query}")
        
        # 2. 提取查询分词
        query_tokens_list = self._get_tokens(standardized_query)
        query_tokens_counter = Counter(query_tokens_list)
        if not query_tokens_counter:
            return []
            
        # 候选案例集：存储 (score, case_idx, matched_tokens)
        candidate_scores = defaultdict(float)
        matched_tokens_per_case = defaultdict(set)
        
        for token, q_tf in query_tokens_counter.items():
            if token in self.inverted_index:
                token_idf = self.idf.get(token, 1.0)
                # 赋予 3-4 字词更高的权重
                weight = 1.5 if len(token) >= 3 else 1.0
                
                for case_idx in self.inverted_index[token]:
                    candidate_scores[case_idx] += (1 + math.log(q_tf)) * token_idf * weight
                    # 记录匹配的较长词汇（2字及以上）作为证据
                    if len(token) >= 2:
                        matched_tokens_per_case[case_idx].add(token)
        
        # 排序并返回 top_k，包含匹配证据
        sorted_indices = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in sorted_indices[:top_k]:
            case = self.cases[idx].copy()
            # 将匹配证据注入 case 对象
            case['_matched_tokens'] = list(matched_tokens_per_case[idx])
            results.append(case)
            
        return results

    def format_cases_for_prompt(self, cases):
        """将检索到的案例/方剂格式化为提示词"""
        if not cases:
            return "暂无参考案例。"
            
        formatted = "参考以下类似临床病历或经典方剂：\n"
        for i, case in enumerate(cases, 1):
            case_type = case.get('type', 'case')
            if case_type == 'formula':
                name = case.get('name', '未知方剂')
                indications = case.get('content', '未知')
                ingredients = case.get('summary', '未知')
                usage = case.get('usage', '')
                contra = case.get('contraindication', '')
                formatted += f"【参考方剂 {i}】名称：{name} | 主治：{indications} | 组成：{ingredients}"
                if usage: formatted += f" | 用法：{usage}"
                if contra: formatted += f" | 禁忌：{contra}"
                formatted += "\n"
            elif case_type == 'herb':
                name = case.get('name', '未知中药')
                indications = case.get('content', '未知')
                effects = case.get('summary', '未知')
                prop = case.get('property', '')
                formatted += f"【参考中药 {i}】名称：{name} | 主治：{indications} | 功效：{effects} | 药性：{prop}\n"
            else:
                content = case.get('Content', '') or case.get('content', '')
                summary = case.get('Summary', '') or case.get('summary', '')
                formatted += f"【参考案例 {i}】内容：{content} | 总结：{summary}\n"
                
        return formatted
