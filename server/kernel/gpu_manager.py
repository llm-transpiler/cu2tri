# -*- coding: utf-8 -*-
"""
GPU资源管理器
负责监控GPU状态、分配GPU资源、管理任务队列
"""
import asyncio
import logging
import time
import subprocess
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
import threading
from datetime import datetime, timedelta

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available, GPU monitoring will be limited")

from .models import GPUInfo, GPUType, TestStage, GPUResourceConfig, TaskStatus

@dataclass
class TaskAllocation:
    """任务分配信息"""
    task_id: str
    conversation_id: str
    gpu_ids: List[int]
    allocated_at: float
    expected_duration: float
    task_type: str
    priority: int = 0

class GPUResourceManager:
    """GPU资源管理器"""
    
    def __init__(self, 
                 config: Optional[GPUResourceConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """初始化GPU资源管理器"""
        self.config = config or GPUResourceConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # GPU状态追踪
        self.gpu_info: Dict[int, GPUInfo] = {}
        self.task_allocations: Dict[str, TaskAllocation] = {}
        self.gpu_locks: Dict[int, asyncio.Lock] = {}
        
        # 监控状态
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.monitor_interval = 5.0  # 监控间隔（秒）
        
        # 统计信息
        self.stats = {
            'total_allocations': 0,
            'active_tasks': 0,
            'gpu_utilization_history': {},
            'average_task_duration': 0.0,
            'allocation_failures': 0
        }
        
        # 线程安全锁
        self._stats_lock = threading.Lock()
        self._allocation_lock = threading.Lock()
        
        # 初始化NVML
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.logger.info("NVML initialized successfully")
            except Exception as e:
                self.nvml_initialized = False
                self.logger.error(f"Failed to initialize NVML: {e}")
        else:
            self.nvml_initialized = False
    
    async def start(self):
        """启动GPU监控"""
        if self.monitoring:
            return
        
        self.logger.info("Starting GPU resource manager...")
        
        # 初始化GPU信息
        await self._discover_gpus()
        
        # 创建GPU锁
        for gpu_id in self.gpu_info.keys():
            self.gpu_locks[gpu_id] = asyncio.Lock()
        
        # 启动监控任务
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_gpus())
        
        self.logger.info(f"GPU resource manager started, managing {len(self.gpu_info)} GPUs")
    
    async def stop(self):
        """停止GPU监控"""
        if not self.monitoring:
            return
        
        self.logger.info("Stopping GPU resource manager...")
        self.monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # 清理分配
        with self._allocation_lock:
            self.task_allocations.clear()
        
        self.logger.info("GPU resource manager stopped")
    
    async def _discover_gpus(self):
        """发现可用的GPU"""
        discovered_gpus = {}
        
        if self.nvml_initialized:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    # 获取内存信息
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # 判断GPU类型
                    gpu_type = self._detect_gpu_type(name)
                    
                    discovered_gpus[i] = GPUInfo(
                        gpu_id=i,
                        gpu_type=gpu_type,
                        memory_total=memory_info.total // (1024 * 1024),  # MB
                        memory_used=memory_info.used // (1024 * 1024),   # MB
                        utilization=0.0,
                        temperature=0.0,
                        is_available=True
                    )
                    
                    self.logger.info(f"Discovered GPU {i}: {name} ({gpu_type.value})")
                    
            except Exception as e:
                self.logger.error(f"Error discovering GPUs with NVML: {e}")
        
        # 备用方法：使用nvidia-smi
        if not discovered_gpus:
            discovered_gpus = await self._discover_gpus_fallback()
        
        self.gpu_info = discovered_gpus
    
    def _detect_gpu_type(self, gpu_name: str) -> GPUType:
        """根据GPU名称检测GPU类型"""
        gpu_name_lower = gpu_name.lower()
        
        if 'h100' in gpu_name_lower:
            return GPUType.H100
        elif 'a100' in gpu_name_lower:
            return GPUType.A100
        elif 'l20' in gpu_name_lower or 'rtx' in gpu_name_lower:
            return GPUType.L20
        else:
            return GPUType.L20  # 默认
    
    async def _discover_gpus_fallback(self) -> Dict[int, GPUInfo]:
        """使用nvidia-smi作为备用GPU发现方法"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            discovered_gpus = {}
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpu_id = int(parts[0])
                    name = parts[1]
                    memory_total = int(parts[2])
                    memory_used = int(parts[3])
                    utilization = float(parts[4]) if parts[4] != 'N/A' else 0.0
                    temperature = float(parts[5]) if parts[5] != 'N/A' else 0.0
                    
                    gpu_type = self._detect_gpu_type(name)
                    
                    discovered_gpus[gpu_id] = GPUInfo(
                        gpu_id=gpu_id,
                        gpu_type=gpu_type,
                        memory_total=memory_total,
                        memory_used=memory_used,
                        utilization=utilization,
                        temperature=temperature,
                        is_available=True
                    )
                    
                    self.logger.info(f"Discovered GPU {gpu_id}: {name} ({gpu_type.value})")
            
            return discovered_gpus
            
        except Exception as e:
            self.logger.error(f"Failed to discover GPUs with fallback method: {e}")
            return {}
    
    async def _monitor_gpus(self):
        """监控GPU状态"""
        while self.monitoring:
            try:
                await self._update_gpu_status()
                await self._cleanup_expired_allocations()
                await asyncio.sleep(self.monitor_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in GPU monitoring: {e}")
                await asyncio.sleep(self.monitor_interval)
    
    async def _update_gpu_status(self):
        """更新GPU状态信息"""
        if self.nvml_initialized:
            try:
                for gpu_id, gpu_info in self.gpu_info.items():
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    
                    # 更新内存信息
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info.memory_used = memory_info.used // (1024 * 1024)
                    
                    # 更新利用率
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_info.utilization = float(utilization.gpu)
                    
                    # 更新温度
                    try:
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        gpu_info.temperature = float(temperature)
                    except:
                        pass  # 某些GPU可能不支持温度查询
                    
                    # 更新可用性
                    gpu_info.is_available = self._is_gpu_available(gpu_info)
                    
            except Exception as e:
                self.logger.error(f"Error updating GPU status with NVML: {e}")
        else:
            # 使用nvidia-smi备用方法
            await self._update_gpu_status_fallback()
    
    async def _update_gpu_status_fallback(self):
        """使用nvidia-smi更新GPU状态"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpu_id = int(parts[0])
                    if gpu_id in self.gpu_info:
                        gpu_info = self.gpu_info[gpu_id]
                        gpu_info.memory_used = int(parts[1])
                        gpu_info.utilization = float(parts[2]) if parts[2] != 'N/A' else 0.0
                        gpu_info.temperature = float(parts[3]) if parts[3] != 'N/A' else 0.0
                        gpu_info.is_available = self._is_gpu_available(gpu_info)
                        
        except Exception as e:
            self.logger.error(f"Error updating GPU status with fallback method: {e}")
    
    def _is_gpu_available(self, gpu_info: GPUInfo) -> bool:
        """判断GPU是否可用"""
        # 检查内存使用率
        memory_usage_ratio = gpu_info.memory_used / gpu_info.memory_total
        if memory_usage_ratio > self.config.memory_threshold:
            return False
        
        # 检查利用率
        if gpu_info.utilization > self.config.utilization_threshold * 100:
            return False
        
        # 检查温度
        if gpu_info.temperature > self.config.temperature_threshold:
            return False
        
        # 检查是否有活跃任务
        if gpu_info.current_task_id:
            return False
        
        return True
    
    async def _cleanup_expired_allocations(self):
        """清理过期的分配"""
        current_time = time.time()
        expired_tasks = []
        
        with self._allocation_lock:
            for task_id, allocation in self.task_allocations.items():
                if current_time - allocation.allocated_at > allocation.expected_duration + 60:  # 1分钟缓冲
                    expired_tasks.append(task_id)
        
        for task_id in expired_tasks:
            await self.release_gpu(task_id)
            self.logger.warning(f"Released expired allocation for task {task_id}")
    
    async def allocate_gpu(self, 
                          task_id: str,
                          conversation_id: str,
                          stage: TestStage,
                          task_type: str,
                          gpu_count: int = 1,
                          specific_gpus: Optional[List[int]] = None,
                          expected_duration: float = 300.0,
                          priority: int = 0,
                          wait_for_idle: bool = False) -> Optional[List[int]]:
        """分配GPU资源"""
        
        # 确定候选GPU
        if specific_gpus:
            candidate_gpus = specific_gpus
        elif stage == TestStage.DEVELOPMENT:
            candidate_gpus = self.config.development_gpus
        else:
            candidate_gpus = self.config.production_gpus
        
        # 如果需要等待空闲，先等待
        if wait_for_idle:
            await self._wait_for_gpu_idle(candidate_gpus)
        
        # 选择最佳GPU
        selected_gpus = await self._select_best_gpus(candidate_gpus, gpu_count)
        
        if not selected_gpus:
            with self._stats_lock:
                self.stats['allocation_failures'] += 1
            return None
        
        # 分配GPU
        allocation = TaskAllocation(
            task_id=task_id,
            conversation_id=conversation_id,
            gpu_ids=selected_gpus,
            allocated_at=time.time(),
            expected_duration=expected_duration,
            task_type=task_type,
            priority=priority
        )
        
        with self._allocation_lock:
            self.task_allocations[task_id] = allocation
            for gpu_id in selected_gpus:
                self.gpu_info[gpu_id].current_task_id = task_id
                self.gpu_info[gpu_id].is_available = False
        
        with self._stats_lock:
            self.stats['total_allocations'] += 1
            self.stats['active_tasks'] += 1
        
        self.logger.info(f"Allocated GPUs {selected_gpus} to task {task_id} ({task_type})")
        return selected_gpus
    
    async def _wait_for_gpu_idle(self, candidate_gpus: List[int], timeout: float = 300.0):
        """等待GPU空闲"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_idle = True
            for gpu_id in candidate_gpus:
                if gpu_id in self.gpu_info:
                    gpu_info = self.gpu_info[gpu_id]
                    if gpu_info.utilization > 10.0 or gpu_info.current_task_id:
                        all_idle = False
                        break
            
            if all_idle:
                return
            
            await asyncio.sleep(2.0)
        
        self.logger.warning(f"Timeout waiting for GPUs {candidate_gpus} to become idle")
    
    async def _select_best_gpus(self, candidate_gpus: List[int], gpu_count: int) -> List[int]:
        """选择最佳的GPU"""
        available_gpus = []
        
        for gpu_id in candidate_gpus:
            if gpu_id in self.gpu_info and self.gpu_info[gpu_id].is_available:
                available_gpus.append(gpu_id)
        
        if len(available_gpus) < gpu_count:
            return []
        
        # 按综合评分排序
        def gpu_score(gpu_id: int) -> float:
            gpu_info = self.gpu_info[gpu_id]
            # 评分标准：内存使用率越低越好，利用率越低越好，温度越低越好
            memory_score = 1.0 - (gpu_info.memory_used / gpu_info.memory_total)
            utilization_score = 1.0 - (gpu_info.utilization / 100.0)
            temperature_score = max(0, 1.0 - (gpu_info.temperature / 100.0))
            
            return memory_score * 0.4 + utilization_score * 0.4 + temperature_score * 0.2
        
        available_gpus.sort(key=gpu_score, reverse=True)
        return available_gpus[:gpu_count]
    
    async def release_gpu(self, task_id: str):
        """释放GPU资源"""
        with self._allocation_lock:
            if task_id in self.task_allocations:
                allocation = self.task_allocations[task_id]
                
                # 释放GPU
                for gpu_id in allocation.gpu_ids:
                    if gpu_id in self.gpu_info:
                        self.gpu_info[gpu_id].current_task_id = None
                        self.gpu_info[gpu_id].is_available = self._is_gpu_available(self.gpu_info[gpu_id])
                
                del self.task_allocations[task_id]
                
                with self._stats_lock:
                    self.stats['active_tasks'] = max(0, self.stats['active_tasks'] - 1)
                    
                    # 更新平均任务持续时间
                    task_duration = time.time() - allocation.allocated_at
                    current_avg = self.stats['average_task_duration']
                    total_tasks = self.stats['total_allocations']
                    self.stats['average_task_duration'] = (current_avg * (total_tasks - 1) + task_duration) / total_tasks
                
                self.logger.info(f"Released GPUs {allocation.gpu_ids} from task {task_id}")
    
    def get_gpu_info(self, gpu_ids: Optional[List[int]] = None) -> List[GPUInfo]:
        """获取GPU信息"""
        if gpu_ids is None:
            return list(self.gpu_info.values())
        else:
            return [self.gpu_info[gpu_id] for gpu_id in gpu_ids if gpu_id in self.gpu_info]
    
    def get_available_gpus(self, stage: TestStage) -> List[int]:
        """获取可用的GPU列表"""
        if stage == TestStage.DEVELOPMENT:
            candidate_gpus = self.config.development_gpus
        else:
            candidate_gpus = self.config.production_gpus
        
        return [gpu_id for gpu_id in candidate_gpus 
                if gpu_id in self.gpu_info and self.gpu_info[gpu_id].is_available]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._stats_lock:
            stats = self.stats.copy()
        
        # 添加实时信息
        stats['gpu_info'] = {gpu_id: {
            'type': info.gpu_type.value,
            'utilization': info.utilization,
            'memory_usage': f"{info.memory_used}/{info.memory_total}MB",
            'temperature': info.temperature,
            'is_available': info.is_available,
            'current_task': info.current_task_id
        } for gpu_id, info in self.gpu_info.items()}
        
        with self._allocation_lock:
            stats['current_allocations'] = len(self.task_allocations)
            stats['allocation_details'] = {
                task_id: {
                    'gpu_ids': alloc.gpu_ids,
                    'task_type': alloc.task_type,
                    'allocated_at': datetime.fromtimestamp(alloc.allocated_at).isoformat(),
                    'duration': time.time() - alloc.allocated_at
                } for task_id, alloc in self.task_allocations.items()
            }
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            'gpu_manager_running': self.monitoring,
            'nvml_available': NVML_AVAILABLE and self.nvml_initialized,
            'total_gpus': len(self.gpu_info),
            'available_gpus': len([g for g in self.gpu_info.values() if g.is_available]),
            'gpu_issues': []
        }
        
        # 检查GPU健康状态
        for gpu_id, gpu_info in self.gpu_info.items():
            if gpu_info.temperature > self.config.temperature_threshold:
                health_status['gpu_issues'].append(f"GPU {gpu_id} overheating: {gpu_info.temperature}°C")
            
            memory_usage = gpu_info.memory_used / gpu_info.memory_total
            if memory_usage > 0.95:
                health_status['gpu_issues'].append(f"GPU {gpu_id} low memory: {memory_usage:.1%} used")
        
        health_status['status'] = 'healthy' if not health_status['gpu_issues'] else 'warning'
        return health_status 