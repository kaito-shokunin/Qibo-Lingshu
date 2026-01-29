#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
岐伯灵枢：中医四诊AI辅助分析平台
知识图谱查询模块

连接 Neo4j 数据库并提供舌诊相关的知识查询接口
"""

import logging
import json
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class TCMKnowledgeGraph:
    """中医知识图谱查询类"""
    
    def __init__(self, uri="bolt://localhost:7474", user="neo4j", password="your_password"):
        """初始化 Neo4j 连接"""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # 测试连接
            self.driver.verify_connectivity()
            logger.info("成功连接到 Neo4j 知识图谱数据库")
        except Exception as e:
            logger.error(f"连接 Neo4j 失败: {str(e)}")
            self.driver = None

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()

    def get_coating_interpretation(self, label):
        """获取某种舌苔的中医临床意义"""
        if not self.driver:
            return None
            
        # 定义常见术语映射 (中英文映射)
        synonyms = {
            "黑苔": "black tongue coating",
            "剥苔": "map tongue coating",
            "紫舌": "purple tongue coating",
            "黄厚腻苔": "red tongue yellow fur thick greasy fur",
            "红舌厚腻苔": "The red tongue is thick and greasy",
            "白厚腻苔": "The white tongue is thick and greasy",
            "胖大舌": "swollen tongue",
            "齿痕舌": "teeth-marked tongue",
            "瘦薄舌": "thin tongue",
            "裂纹舌": "fissured tongue"
        }
        
        # 尝试映射
        search_label = synonyms.get(label, label)
            
        # 1. 尝试精确匹配
        query_exact = """
        MATCH (c:CoatingType)
        WHERE c.name = $label
        RETURN c.description AS description
        """
        
        # 2. 如果精确匹配失败，尝试模糊匹配 (针对非标准输入)
        query_fuzzy = """
        MATCH (c:CoatingType)
        WHERE c.name CONTAINS $label OR $label CONTAINS c.name
        RETURN c.description AS description, c.name AS name
        LIMIT 1
        """
        
        try:
            with self.driver.session() as session:
                # 首先尝试精确匹配
                result = session.run(query_exact, label=search_label)
                record = result.single()
                if record:
                    return record["description"]
                
                # 精确匹配失败，尝试模糊匹配
                result = session.run(query_fuzzy, label=search_label)
                record = result.single()
                if record:
                    logger.info(f"知识图谱模糊匹配成功: {search_label} -> {record['name']}")
                    return record["description"]
                
                logger.warning(f"知识图谱中未找到舌苔类型: {label}")
                return None
        except Exception as e:
            logger.error(f"查询知识图谱失败: {str(e)}")
            return None

    def get_similar_cases(self, label, limit=5):
        """查找具有相同舌苔类型的相似案例图片路径"""
        if not self.driver:
            return []
            
        query = """
        MATCH (i:Image)-[:HAS_COATING]->(c:CoatingType {name: $label})
        RETURN i.path AS path, i.folder AS folder
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, label=label, limit=limit)
                return [{"path": record["path"], "folder": record["folder"]} for record in result]
        except Exception as e:
            logger.error(f"查询相似案例失败: {str(e)}")
            return []

    def get_all_coating_types(self):
        """获取所有已记录的舌苔类型及其描述"""
        if not self.driver:
            return {}
            
        query = """
        MATCH (c:CoatingType)
        RETURN c.name AS name, c.description AS description
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return {record["name"]: record["description"] for record in result}
        except Exception as e:
            logger.error(f"获取舌苔类型列表失败: {str(e)}")
            return {}

    def record_feedback(self, image_path, corrected_result, features=None):
        """记录用户反馈到知识图谱中
        :param image_path: 图像路径
        :param corrected_result: 修正后的结果
        :param features: 提取的图像特征(HSV/LAB等)
        """
        if not self.driver:
            return False
            
        query = """
        MERGE (i:Image {path: $path})
        SET i.last_feedback = datetime(),
            i.tongue_color = $tongue_color,
            i.tongue_coating = $tongue_coating,
            i.features = $features
        MERGE (c:CoatingType {name: $tongue_coating})
        MERGE (i)-[:HAS_COATING]->(c)
        RETURN i
        """
        
        try:
            tongue_color = corrected_result.get("tongue_color", "未知")
            tongue_coating = corrected_result.get("tongue_coating", "未知")
            features_json = json.dumps(features) if features else None
            
            with self.driver.session() as session:
                session.run(query, 
                            path=image_path, 
                            tongue_color=tongue_color, 
                            tongue_coating=tongue_coating,
                            features=features_json)
                logger.info(f"已将反馈记录到 Neo4j: {image_path}")
                return True
        except Exception as e:
            logger.error(f"记录反馈到 Neo4j 失败: {str(e)}")
            return False
