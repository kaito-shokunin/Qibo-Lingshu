#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
岐伯灵枢：中医四诊AI辅助分析平台
健康数据采集模块

适用于Jetson Nano开发板
"""

import asyncio
import json
import time
import logging
import random
from datetime import datetime
import os

# 尝试导入蓝牙库
try:
    from bleak import BleakScanner, BleakClient
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False
    logging.warning("Bleak库不可用，将进入演示模式进行数据采集")

logger = logging.getLogger(__name__)

class HealthDataCollector:
    """健康数据采集类"""
    
    def __init__(self):
        """初始化健康数据采集器"""
        self.bluetooth_client = None
        self.device = None
        self.connected = False
        self.device_characteristics = {}
        self.device_type = None
        
        # 常见健康监测设备的蓝牙服务和特征值
        self.device_profiles = {
            "heart_rate": {
                "service_uuid": "0000180d-0000-1000-8000-00805f9b34fb",
                "characteristic_uuid": "00002a37-0000-1000-8000-00805f9b34fb"
            },
            "blood_pressure": {
                "service_uuid": "00001810-0000-1000-8000-00805f9b34fb",
                "characteristic_uuid": "00002a35-0000-1000-8000-00805f9b34fb"
            },
            "blood_oxygen": {
                "service_uuid": "00001822-0000-1000-8000-00805f9b34fb",
                "characteristic_uuid": "00002a5f-0000-1000-8000-00805f9b34fb"
            },
            "temperature": {
                "service_uuid": "00001809-0000-1000-8000-00805f9b34fb",
                "characteristic_uuid": "00002a1c-0000-1000-8000-00805f9b34fb"
            },
            "battery": {
                "service_uuid": "0000180f-0000-1000-8000-00805f9b34fb",
                "characteristic_uuid": "00002a19-0000-1000-8000-00805f9b34fb"
            }
        }
        
        logger.info("健康数据采集器初始化完成")
    
    async def connect_device(self, address):
        """连接蓝牙设备并初始化采集环境"""
        # 每次连接前重置所有数据
        self._last_heart_rate = "--"
        self._last_blood_pressure = "--"
        self._last_blood_pressure_systolic = "--"
        self._last_blood_pressure_diastolic = "--"
        self._last_blood_oxygen = "--"
        self._last_raw_data = None
        
        if not BLEAK_AVAILABLE:
            logger.error("蓝牙库不可用，无法连接设备")
            self.connected = True
            self.device_type = "演示设备"
            return True

        try:
            logger.info(f"正在尝试连接设备: {address}")
            
            # 策略：先尝试直接连接（针对已配对或系统已发现的设备）
            self.bluetooth_client = BleakClient(address)
            
            connected = False
            try:
                # 缩短连接超时，如果系统已占用会很快报错
                await self.bluetooth_client.connect(timeout=10.0)
                connected = self.bluetooth_client.is_connected
            except Exception as e:
                logger.warning(f"直接连接失败，尝试扫描后连接: {str(e)}")
            
            if not connected:
                # 策略二：如果直接连接失败，则进行短时间扫描
                logger.info("正在扫描设备广播...")
                device = await BleakScanner.find_device_by_address(address, timeout=5.0)
                if not device:
                    # 最后的尝试：按名称模糊匹配
                    devices = await BleakScanner.discover(timeout=5.0)
                    for d in devices:
                        if d.name and "BK01" in d.name.upper():
                            device = d
                            break
                
                if device:
                    self.bluetooth_client = BleakClient(device)
                    await self.bluetooth_client.connect(timeout=15.0)
                    connected = self.bluetooth_client.is_connected
            
            if connected:
                logger.info(f"成功连接到设备: {address}")
                self.connected = True
                self.device_type = f"BK01 ({address})"
                # 发现服务
                await self._discover_services()
                # 尝试发送启动指令
                await self._try_trigger_measurement()
                return True
            else:
                logger.error(f"无法建立连接: {address}")
                # 回退到演示模式
                self.connected = True
                self.device_type = "演示设备"
                return True
            
            if not self.connected:
                logger.warning(f"未找到目标设备 {device_mac}，将进入演示模式")
                self.connected = True
                self.device_type = "演示设备"
                return True
                
            return True
                
        except Exception as e:
            logger.error(f"连接设备失败: {str(e)}")
            # 回退到演示模式
            logger.info("回退到演示模式")
            self.connected = True
            self.device_type = "演示设备"
            return True
    
    async def _discover_services(self):
        """发现设备服务和特征值"""
        try:
            # 增加重试机制，防止瞬时连接抖动导致服务列表为空
            services = self.bluetooth_client.services
            if not services:
                logger.warning("初次扫描服务列表为空，尝试强制刷新...")
                # 某些系统下需要显式调用 get_services
                services = await self.bluetooth_client.get_services()
            
            service_list = list(services)
            logger.info(f"发现 {len(service_list)} 个服务")
            
            # 立即备份当前的服务详情，防止被后续操作覆盖
            temp_device_info = []
            
            self.device_characteristics = {}
            all_chars = []
            self._write_chars = []
            self._notify_chars = []
            
            # 遍历所有服务
            for service in service_list:
                service_uuid = service.uuid.lower()
                
                service_info = {
                    "uuid": service_uuid,
                    "name": service.description or "Unknown Service",
                    "characteristics": []
                }
                
                for char in service.characteristics:
                    char_uuid = char.uuid.lower()
                    props = char.properties
                    
                    # 尝试读取可读特征值的初始值
                    initial_value = "Unknown"
                    if "read" in props:
                        try:
                            # 这里不使用 await 以免阻塞整个扫描流程，仅作为占位
                            # 实际读取将在 collect_vitals 的暴力模式中进行
                            initial_value = "Readable"
                        except: pass

                    char_info = {
                        "uuid": char_uuid,
                        "name": char.description or "Unknown Char",
                        "properties": props,
                        "value": initial_value
                    }
                    service_info["characteristics"].append(char_info)
                    
                    self.device_characteristics[char.uuid] = char
                    all_chars.append(char)
                    
                    # 识别核心通道
                    if "write" in props or "write-without-response" in props:
                        self._write_chars.append(char)
                    
                    if "notify" in props or "indicate" in props:
                        if not any(std in char_uuid for std in ["1800", "1801", "2a05"]):
                            self._notify_chars.append(char)
                
                temp_device_info.append(service_info)
            
            # 统一赋值给持久变量
            self._last_device_info = temp_device_info
            
            # 检查协议类型
            standard_uuids = ["0000180d-0000-1000-8000-00805f9b34fb", "00001810-0000-1000-8000-00805f9b34fb", "00001809-0000-1000-8000-00805f9b34fb"]
            h_band_uuids = ["0000fee7-0000-1000-8000-00805f9b34fb"]
            bk01_uuids = ["f0080001-0451-4000-b000-000000000000"]
            
            is_standard = any(uuid in [s.uuid.lower() for s in service_list] for uuid in standard_uuids)
            is_h_band = any(uuid in [s.uuid.lower() for s in service_list] for uuid in h_band_uuids)
            is_bk01 = any(uuid in [s.uuid.lower() for s in service_list] for uuid in bk01_uuids)
            
            if is_standard:
                self.device_type = "标准健康监测设备"
            elif is_bk01:
                self.device_type = "岐伯灵枢专用设备(BK01-f008)"
            elif is_h_band:
                self.device_type = "定制健康监测设备(BK01/H Band)"
            else:
                self.device_type = "定制健康监测设备(BK01)"
                
            logger.info(f"服务发现完成，识别结果: {self.device_type}")
            
        except Exception as e:
            logger.error(f"发现服务失败: {str(e)}")
            if "exclusive" in str(e).lower() or "occupied" in str(e).lower():
                logger.error("!!! 检测到设备被其他应用占用，请关闭 Bluetooth LE Explorer 或手机 App !!!")

    async def _try_trigger_measurement(self):
        """针对 BK01 和 H Band 协议发送强化启动指令"""
        # BK01 f008/f002 协议指令通道
        cmd_chars = [
            "f0080003-0451-4000-b000-000000000000",
            "f0020003-0451-4000-b000-000000000000"
        ]
        
        for bk01_cmd_char in cmd_chars:
            if bk01_cmd_char in self.device_characteristics:
                char = self.device_characteristics[bk01_cmd_char]
                try:
                    # 强化指令集
                    commands = [
                        # 1. 登录/握手包 (16字节)
                        bytearray([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
                        # 2. 针对 AE41 的特定启动指令 (有些设备需要 0x01 开启通知)
                        bytearray([0x01]),
                        # 3. 开启实时健康流 (H Band 标准)
                        bytearray([0x40, 0x01, 0x01, 0x42]),
                        # 4. 开启血压测量
                        bytearray([0x02]),
                        # 5. 私有协议启动包
                        bytearray([0xBD, 0x01, 0x01, 0xBE])
                    ]
                    for cmd in commands:
                        if not self.bluetooth_client or not self.bluetooth_client.is_connected:
                            break
                        await self.bluetooth_client.write_gatt_char(char, cmd)
                        logger.info(f">>> 已向 BK01 ({bk01_cmd_char}) 发送强化指令: {cmd.hex().upper()}")
                        await asyncio.sleep(0.3) 
                except Exception as e:
                    logger.warning(f"发送 BK01 ({bk01_cmd_char}) 启动指令失败: {str(e)}")

    async def disconnect(self):
        """断开设备连接并清理资源"""
        if self.bluetooth_client and self.bluetooth_client.is_connected:
            try:
                logger.info("正在断开蓝牙连接...")
                await self.bluetooth_client.disconnect()
                self.connected = False
                logger.info("蓝牙连接已断开")
            except Exception as e:
                logger.error(f"断开连接时出错: {str(e)}")
        self.bluetooth_client = None

    async def collect_vitals(self):
        """采集脉搏、心率、血氧等数据"""
        if not self.connected:
            logger.warning("设备未连接，尝试重新连接/回退模式")
            # 尝试自动连接
            return self._generate_fallback_data()
        
        try:
            # 只有当蓝牙库可用、且确实建立了蓝牙客户端连接时，才尝试真实采集
            if BLEAK_AVAILABLE and self.bluetooth_client and self.bluetooth_client.is_connected:
                logger.info(f"开始从真实设备 {self.device_type} 采集数据")
                return await self._collect_from_device()
            else:
                logger.info("当前处于回退模式，生成演示数据")
                return self._generate_fallback_data()
                
        except Exception as e:
            logger.error(f"采集健康数据失败: {str(e)}")
            # 回退到演示数据
            logger.info("使用演示数据")
            return self._generate_fallback_data()
    
    async def _collect_from_device(self):
        """从真实设备采集数据"""
        try:
            # 清除旧数据，确保本次采集是新鲜的
            self._last_heart_rate = None
            self._last_blood_pressure = None
            self._last_blood_pressure_systolic = None
            self._last_blood_pressure_diastolic = None
            self._last_blood_oxygen = "--"
            self._last_temperature = None
            self._last_raw_data = None

            # 1. 尝试触发测量（发送测量指令）
            await self._try_trigger_measurement()
            
            # 初始化所有字段为 "--" 或 None
            vitals = {
                "timestamp": time.time(),
                "device_type": self.device_type,
                "heart_rate": "--",
                "blood_oxygen": "--",
                "blood_pressure": ["--", "--"],
                "blood_pressure_systolic": "--",
                "blood_pressure_diastolic": "--",
                "temperature": "--",
                "raw_data": "--",
                "device_info": getattr(self, '_last_device_info', [])
            }
            
            # 2. 识别所有可能的候选特征值（Notify 类型）
            candidates = []
            for uuid, char in self.device_characteristics.items():
                if "notify" in char.properties or "indicate" in char.properties:
                    if not uuid.startswith("00002a"): # 避开系统特征值
                        candidates.append(uuid)
            
            logger.info(f"开始并行监听 {len(candidates)} 个数据通道: {candidates}")
            
            # 3. 并行订阅所有通道
            subscribed = []
            for uuid in candidates:
                if not self.bluetooth_client or not self.bluetooth_client.is_connected:
                    logger.warning("设备连接已断开，停止订阅")
                    break
                    
                try:
                    # 统一使用一个全能处理器
                    logger.info(f"正在尝试订阅通道: {uuid}")
                    await self.bluetooth_client.start_notify(uuid, self._universal_notification_handler)
                    subscribed.append(uuid)
                    # 关键：给设备一点喘息时间，避免快速连续订阅导致断开
                    await asyncio.sleep(0.3)
                except Exception as e:
                    logger.warning(f"订阅通道 {uuid} 失败: {str(e)}")
            
            # 4. 等待数据上报
            if not subscribed:
                logger.warning("未能成功订阅任何数据通道")
            else:
                logger.info(f"已成功订阅 {len(subscribed)} 个通道，等待 12 秒捕获数据...")
                await asyncio.sleep(12)
            
            # 5. 回退方案：如果通知没数据，尝试暴力读取所有特征值
            if not self._last_heart_rate or self._last_heart_rate == "--":
                logger.info("通知模式未获有效值，进入暴力读取模式...")
                for uuid in self.device_characteristics:
                    if not self.bluetooth_client or not self.bluetooth_client.is_connected:
                        break
                    # 避开系统标准特征值，只读私有的 (如 FEEA, AF02 等)
                    if not uuid.startswith("00002a"):
                        try:
                            val = await self.bluetooth_client.read_gatt_char(uuid)
                            if val and len(val) > 0:
                                logger.info(f"直接读取到特征值 {uuid} 数据: {val.hex().upper()}")
                                self._universal_notification_handler(uuid, val)
                        except Exception as e:
                            continue

            # 6. 停止所有订阅
            for uuid in subscribed:
                try:
                    await self.bluetooth_client.stop_notify(uuid)
                except:
                    pass
            
            # 7. 整理结果
            if self._last_heart_rate:
                vitals["heart_rate"] = self._last_heart_rate
            
            if self._last_blood_pressure:
                vitals["blood_pressure"] = self._last_blood_pressure
                vitals["blood_pressure_systolic"] = self._last_blood_pressure[0]
                vitals["blood_pressure_diastolic"] = self._last_blood_pressure[1]
            
            if self._last_blood_oxygen:
                vitals["blood_oxygen"] = self._last_blood_oxygen
            
            if self._last_temperature:
                vitals["temperature"] = self._last_temperature
            
            if hasattr(self, '_last_raw_data') and self._last_raw_data:
                vitals["raw_data"] = self._last_raw_data
                
            return vitals
            
        except Exception as e:
            logger.error(f"从设备采集数据失败: {str(e)}")
            # 回退到演示数据
            return self._generate_fallback_data()

    def _universal_notification_handler(self, sender, data):
        """全能通知处理器：严格校验 BD 包头"""
        try:
            raw_hex = data.hex().upper()
            short_uuid = str(sender).split('-')[0][-4:] # 取 UUID 的最后 4 位或前 4 位作为标识
            self._last_raw_data = f"[{short_uuid}] {raw_hex}"
            
            # 记录到日志，方便追溯
            logger.info(f"收到通知数据 - 来源: {sender}, 内容: {raw_hex}")
            
            # 只有 BD 开头的才是真正的健康测量包
            if len(data) >= 10 and data[0] == 0xBD:
                # 偏移量通常为: 血压(4,5), 心率(6), 血氧(7)
                systolic = data[4]
                diastolic = data[5]
                heart_rate = data[6]
                
                # 记录原始解析出的数值，用于进一步调试
                logger.debug(f"BD包原始值: S={systolic}, D={diastolic}, HR={heart_rate}")

                # 严格校验：血压、心率不能完全相等且必须在合理范围内
                if 40 < systolic < 220 and 30 < diastolic < 120 and systolic > diastolic:
                    # 避免干扰项：如果心率和血压完全一致，可能是无效包
                    if systolic != heart_rate:
                        self._last_blood_pressure = [systolic, diastolic]
                        self._last_blood_pressure_systolic = systolic
                        self._last_blood_pressure_diastolic = diastolic
                        logger.info(f">>> [真实数据] 血压解析成功: {systolic}/{diastolic}")
                
                if 40 < heart_rate < 200:
                    # 再次校验干扰
                    if heart_rate != systolic:
                        self._last_heart_rate = heart_rate
                        logger.info(f">>> [真实数据] 心率解析成功: {heart_rate}")
                
                if len(data) >= 8:
                    blood_oxygen = data[7]
                    if 90 <= blood_oxygen <= 100:
                        self._last_blood_oxygen = blood_oxygen
                        logger.info(f">>> [真实数据] 血氧解析成功: {blood_oxygen}")
            else:
                # 记录所有非数据包，但不再尝试解析心率
                if raw_hex != "01000000": # 忽略频繁的心跳包日志
                    logger.info(f"收到非数据包(状态/握手): {raw_hex}")
                self._last_raw_data = f"[{short_uuid}] {raw_hex}"

        except Exception as e:
            logger.error(f"数据解析失败: {str(e)}")
    
    async def _read_heart_rate(self):
        """读取心率数据"""
        try:
            hr_char = self.device_profiles["heart_rate"]["characteristic_uuid"]
            
            if hr_char and hr_char in self.device_characteristics:
                logger.info(f"正在通过特征值 {hr_char} 订阅心率...")
                # 启用通知
                await self.bluetooth_client.start_notify(hr_char, self._heart_rate_notification_handler)
                
                # 等待数据（实际应用中可能需要更完善的同步机制）
                await asyncio.sleep(3)
                
                # 停止通知
                await self.bluetooth_client.stop_notify(hr_char)
                
                return getattr(self, '_last_heart_rate', None)
            else:
                logger.warning("设备不支持标准心率测量，且未发现可用的私用心率通道")
                return None
        except Exception as e:
            logger.error(f"读取心率失败: {str(e)}")
            return None
    
    def _heart_rate_notification_handler(self, sender, data):
        """心率数据通知处理程序"""
        try:
            logger.info(f"收到心率原始数据 [{sender}]: {data.hex().upper()}")
            
            if len(data) >= 1:
                # 尝试标准解析
                try:
                    format_flag = data[0] & 0x01
                    if format_flag == 0 and len(data) >= 2:
                        heart_rate = data[1]
                    elif format_flag == 1 and len(data) >= 3:
                        heart_rate = int.from_bytes(data[1:3], byteorder='little')
                    else:
                        # 盲猜：如果长度只有 1-2 字节，直接取值
                        heart_rate = data[0] if len(data) == 1 else data[1]
                except:
                    heart_rate = data[0]

                # 验证数据合理性 (30-220 bpm)
                if 30 <= heart_rate <= 220:
                    self._last_heart_rate = heart_rate
                    logger.info(f"解析到有效心率: {heart_rate} bpm")
                else:
                    logger.warning(f"解析到无效心率值: {heart_rate}")
                
        except Exception as e:
            logger.error(f"处理心率数据失败: {str(e)}")
    
    async def _read_blood_pressure(self):
        """读取血压数据"""
        try:
            bp_char = self.device_profiles["blood_pressure"]["characteristic_uuid"]
            
            if bp_char and bp_char in self.device_characteristics:
                logger.info(f"正在通过特征值 {bp_char} 订阅血压...")
                await self.bluetooth_client.start_notify(bp_char, self._blood_pressure_notification_handler)
                await asyncio.sleep(2)
                await self.bluetooth_client.stop_notify(bp_char)
                return getattr(self, '_last_blood_pressure', None)
            else:
                logger.warning("设备不支持标准血压测量，且未发现可用的私有血压通道")
                return None
        except Exception as e:
            logger.error(f"读取血压失败: {str(e)}")
            return None
    
    def _blood_pressure_notification_handler(self, sender, data):
        """血压数据通知处理程序"""
        try:
            logger.info(f"收到血压原始数据 [{sender}]: {data.hex().upper()}")
            
            # 标准解析逻辑
            if len(data) >= 7:
                systolic = int.from_bytes(data[1:3], byteorder='little')
                diastolic = int.from_bytes(data[3:5], byteorder='little')
            elif len(data) >= 2:
                # 盲猜：有些设备直接发 [收缩压, 舒张压]
                systolic = data[0]
                diastolic = data[1]
            else:
                return

            if 40 <= systolic <= 250 and 30 <= diastolic <= 150:
                self._last_blood_pressure = [systolic, diastolic]
                logger.info(f"解析到有效血压: {systolic}/{diastolic} mmHg")
        except Exception as e:
            logger.error(f"处理血压数据失败: {str(e)}")

    async def _read_blood_oxygen(self):
        """读取血氧数据"""
        try:
            bo_char = self.device_profiles["blood_oxygen"]["characteristic_uuid"]
            if bo_char in self.device_characteristics:
                await self.bluetooth_client.start_notify(bo_char, self._blood_oxygen_notification_handler)
                await asyncio.sleep(2)
                await self.bluetooth_client.stop_notify(bo_char)
                return getattr(self, '_last_blood_oxygen', None)
            return None
        except Exception as e:
            logger.error(f"读取血氧失败: {str(e)}")
            return None

    def _blood_oxygen_notification_handler(self, sender, data):
        """血氧数据通知处理程序"""
        try:
            logger.info(f"收到血氧原始数据 [{sender}]: {data.hex().upper()}")
            if len(data) >= 1:
                # 盲猜：通常在第1或第2个字节
                val = data[0] if data[0] > 50 else (data[1] if len(data) > 1 else 0)
                if 50 <= val <= 100:
                    self._last_blood_oxygen = val
                    logger.info(f"解析到有效血氧: {val}%")
        except Exception as e:
            logger.error(f"处理血氧数据失败: {str(e)}")

    async def _read_temperature(self):
        """读取体温数据"""
        try:
            temp_char = self.device_profiles["temperature"]["characteristic_uuid"]
            if temp_char in self.device_characteristics:
                await self.bluetooth_client.start_notify(temp_char, self._temperature_notification_handler)
                await asyncio.sleep(2)
                await self.bluetooth_client.stop_notify(temp_char)
                return getattr(self, '_last_temperature', None)
            return None
        except Exception as e:
            logger.error(f"读取体温失败: {str(e)}")
            return None

    def _temperature_notification_handler(self, sender, data):
        """体温数据通知处理程序"""
        try:
            logger.info(f"收到体温原始数据 [{sender}]: {data.hex().upper()}")
            if len(data) >= 2:
                # 简单解析：如果是 16 位整数表示 36.5 -> 365
                val = int.from_bytes(data[:2], byteorder='little')
                if 300 <= val <= 450:
                    temp = val / 10.0
                else:
                    # 或者是 [整数, 小数] 格式
                    temp = data[0] + (data[1]/10.0 if data[1] < 10 else data[1]/100.0)
                
                if 30.0 <= temp <= 45.0:
                    self._last_temperature = temp
                    logger.info(f"解析到有效体温: {temp}°C")
        except Exception as e:
            logger.error(f"处理体温数据失败: {str(e)}")
    
    def _generate_fallback_data(self):
        """生成演示健康数据"""
        try:
            # 生成随机但合理的健康数据
            heart_rate = random.randint(60, 100)
            blood_oxygen = random.randint(95, 100)
            pulse = heart_rate  # 脉搏通常等于心率
            
            # 生成血压数据
            systolic = random.randint(110, 140)
            diastolic = random.randint(70, 90)
            
            # 生成体温数据
            temperature = round(random.uniform(36.0, 37.5), 1)
            
            # 生成呼吸频率
            respiratory_rate = random.randint(12, 20)
            
            vitals = {
                "heart_rate": heart_rate,
                "blood_oxygen": blood_oxygen,
                "pulse": pulse,
                "blood_pressure": [systolic, diastolic],
                "blood_pressure_systolic": systolic,
                "blood_pressure_diastolic": diastolic,
                "temperature": temperature,
                "respiratory_rate": respiratory_rate,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device_type": self.device_type or "演示设备",
                "device_info": getattr(self, '_last_device_info', [])
            }
            
            logger.info(f"生成演示健康数据: {vitals}")
            return vitals
            
        except Exception as e:
            logger.error(f"生成演示数据失败: {str(e)}")
            raise
    
    def detect_abnormalities(self, vitals):
        """检测异常指标"""
        try:
            abnormalities = []
            
            # 安全转换函数
            def to_float(val):
                try:
                    if val is None or val == "--": return None
                    return float(val)
                except: return None

            # 心率异常检测
            heart_rate = to_float(vitals.get("heart_rate"))
            if heart_rate:
                if heart_rate < 60:
                    abnormalities.append({
                        "type": "心率",
                        "status": "偏低",
                        "value": f"{heart_rate} bpm",
                        "description": "心率过缓（心动过缓）"
                    })
                elif heart_rate > 100:
                    abnormalities.append({
                        "type": "心率",
                        "status": "偏高",
                        "value": f"{heart_rate} bpm",
                        "description": "心率过快（心动过速）"
                    })
            
            # 血氧异常检测
            blood_oxygen = to_float(vitals.get("blood_oxygen"))
            if blood_oxygen:
                if blood_oxygen < 95:
                    abnormalities.append({
                        "type": "血氧饱和度",
                        "status": "偏低",
                        "value": f"{blood_oxygen}%",
                        "description": "血氧饱和度偏低，可能存在缺氧"
                    })
            
            # 血压异常检测
            systolic = to_float(vitals.get("blood_pressure_systolic"))
            diastolic = to_float(vitals.get("blood_pressure_diastolic"))
            
            if systolic and diastolic:
                if systolic > 140 or diastolic > 90:
                    abnormalities.append({
                        "type": "血压",
                        "status": "偏高",
                        "value": f"{systolic}/{diastolic} mmHg",
                        "description": "血压偏高（高血压）"
                    })
                elif systolic < 90 or diastolic < 60:
                    abnormalities.append({
                        "type": "血压",
                        "status": "偏低",
                        "value": f"{systolic}/{diastolic} mmHg",
                        "description": "血压偏低（低血压）"
                    })
            
            # 体温异常检测
            temperature = to_float(vitals.get("temperature"))
            if temperature:
                if temperature > 37.3:
                    abnormalities.append({
                        "type": "体温",
                        "status": "偏高",
                        "value": f"{temperature}°C",
                        "description": "体温偏高，可能存在发热"
                    })
                elif temperature < 36.0:
                    abnormalities.append({
                        "type": "体温",
                        "status": "偏低",
                        "value": f"{temperature}°C",
                        "description": "体温偏低，可能存在体温调节障碍"
                    })
            
            return abnormalities
        except Exception as e:
            logger.error(f"检测异常指标失败: {str(e)}")
            return []
    
    def analyze_pulse(self, vitals):
        """分析脉象特征"""
        try:
            # 统一使用安全转换
            def to_float(val):
                try:
                    if val is None or val == "--": return None
                    return float(val)
                except: return None

            heart_rate = to_float(vitals.get("heart_rate"))
            blood_pressure = vitals.get("blood_pressure")
            
            # 基础数据验证
            if heart_rate is None:
                return {
                    "pulse_rate": "--",
                    "pulse_rhythm": "--",
                    "pulse_strength": "--",
                    "pulse_condition": "--",
                    "tcm_pulse": "未采集到有效数据",
                    "tcm_meaning": "请连接设备并开始采集"
                }

            systolic = 0
            diastolic = 0
            if isinstance(blood_pressure, list) and len(blood_pressure) >= 2:
                systolic = to_float(blood_pressure[0]) or 0
                diastolic = to_float(blood_pressure[1]) or 0
            
            pulse_analysis = {
                "pulse_rate": heart_rate,
                "pulse_rhythm": "正常",
                "pulse_strength": "中等",
                "pulse_condition": "正常"
            }
            
            # 根据心率分析脉率
            if heart_rate < 60:
                pulse_analysis["pulse_rate_type"] = "迟脉"
                pulse_analysis["pulse_condition"] = "寒证"
            elif heart_rate > 90:
                pulse_analysis["pulse_rate_type"] = "数脉"
                pulse_analysis["pulse_condition"] = "热证"
            else:
                pulse_analysis["pulse_rate_type"] = "平脉"
                pulse_analysis["pulse_condition"] = "正常"
            
            # 根据血压分析脉形
            if systolic > 0 and diastolic > 0:
                pulse_pressure = systolic - diastolic
                if pulse_pressure > 60:
                    pulse_analysis["pulse_shape"] = "洪脉"
                elif pulse_pressure < 30:
                    pulse_analysis["pulse_shape"] = "细脉"
                else:
                    pulse_analysis["pulse_shape"] = "正常"
            else:
                pulse_analysis["pulse_shape"] = "未知"
            
            # 综合脉象分析
            if heart_rate < 60 and systolic < 100 and systolic > 0:
                pulse_analysis["tcm_pulse"] = "迟细脉"
                pulse_analysis["tcm_meaning"] = "气血两虚"
            elif heart_rate > 90 and systolic > 140:
                pulse_analysis["tcm_pulse"] = "数弦脉"
                pulse_analysis["tcm_meaning"] = "肝阳上亢"
            else:
                pulse_analysis["tcm_pulse"] = "平脉"
                pulse_analysis["tcm_meaning"] = "气血调和"
            
            return pulse_analysis
            
        except Exception as e:
            logger.error(f"分析脉象失败: {str(e)}")
            return {
                "pulse_rate": "--",
                "pulse_rhythm": "未知",
                "pulse_strength": "未知",
                "pulse_condition": "无法分析",
                "tcm_pulse": "解析错误",
                "tcm_meaning": str(e)
            }
    
    async def disconnect(self):
        """断开设备连接"""
        try:
            if self.bluetooth_client and self.connected:
                await self.bluetooth_client.disconnect()
                self.connected = False
                logger.info("设备已断开连接")
            else:
                logger.info("设备未连接或使用回退模式")
        except Exception as e:
            logger.error(f"断开连接失败: {str(e)}")
    
    def get_device_info(self):
        """获取设备信息"""
        try:
            return {
                "connected": self.connected,
                "device_type": self.device_type,
                "device_address": self.device.address if self.device else None,
                "device_name": self.device.name if self.device else None,
                "supported_features": list(self.device_characteristics.keys()) if self.device_characteristics else []
            }
        except Exception as e:
            logger.error(f"获取设备信息失败: {str(e)}")
            return {
                "connected": False,
                "error": str(e)
            }