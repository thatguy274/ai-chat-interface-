
#!/usr/bin/env python3

import secrets
import time
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, AsyncGenerator, Any, Callable, Union, Set
import httpx
import asyncio
import bcrypt
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import os
import re
import base64
import io
import math
import uuid
import subprocess
import sys
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, deque, OrderedDict, ChainMap
import weakref
from functools import wraps, lru_cache, partial, reduce
from itertools import chain, cycle, islice, tee
import traceback
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from abc import ABC, abstractmethod
import inspect
import pickle
import zlib
import gzip
from copy import deepcopy
import signal
from queue import Queue, PriorityQueue
import multiprocessing as mp
from multiprocessing import Manager, Pool, Value, Array

try:
    from redis import asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("PyPDF2 not available - PDF text extraction disabled")

os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['VECLIB_MAXIMUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
os.environ['TORCH_INTEROP_THREAD_COUNT'] = '8'

try:
    from stphoto import PhotoGenerator
    PHOTO_AVAILABLE = True
except ImportError:
    PHOTO_AVAILABLE = False
    print("Photo generation module not available")
try:
    from stvideo import video_generator, VIDEO_STEPS
    VIDEO_AVAILABLE = True
except ImportError:
    video_generator = None
    VIDEO_STEPS = 50
    VIDEO_AVAILABLE = False
    print("Video generation not available")
SECRET_KEY = secrets.token_hex(32)
ADMIN_PASSWORD_HASH = "$2b$12$v/ZEi8Dde5ji4ZSNHYMpHeUzvRE56LqGA/wNfz4L0kotVrzrMgpZW"
SESSION_TIMEOUT = 3600
BROWSER_TOKEN_EXPIRY = 30 * 24 * 3600
OPENROUTER_API_KEY = "#enter your openrouter api key here#"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

AVAILABLE_MODELS = [
    {
        "id": "z-ai/glm-4.6",
        "name": "GLM-4.6",
        "description": "Compared with GLM-4.5, this generation brings several key improvements Longer context window: The context window has been expanded from 128K to 200K tokens, enabling the model to handle more complex agentic tasks. Superior coding performance: The model achieves higher scores on code benchmarks and demonstrates better real-world performance in applications such as Claude Code、Cline、Roo Code and Kilo Code, including improvements in generating visually polished front-end pages. Advanced reasoning: GLM-4.6 shows a clear improvement in reasoning performance and supports tool use during inference, leading to stronger overall capability. More capable agents: GLM-4.6 exhibits stronger performance in tool using and search-based agents, and integrates more effectively within agent frameworks. Refined writing: Better aligns with human preferences in style and readability, and performs more naturally in role-playing scenarios."
    },
    {
        "id": "thedrummer/cydonia-24b-v4.1",
        "name": "unsensord",
        "description": "this is a unsonsord modle it will be mean and agressive. my frends wanted it lol its really inipropreat."
    },
    {
        "id": "qwen/qwen-2.5-72b-instruct",
        "name": "Spell Check",
        "description": "Special phonetic spell checking model for dyslexia and dysgraphia support. Only corrects spelling errors without changing meaning or word choice."
    },
    {
        "id": "deepseek/deepseek-v3.2-exp",
        "name": "DeepSeek V3.2",
        "description": "DeepSeek-V3.2-Exp is an experimental large language model released by DeepSeek as an intermediate step between V3.1 and future architectures. It introduces DeepSeek Sparse Attention (DSA), a fine-grained sparse attention mechanism designed to improve training and inference efficiency in long-context scenarios while maintaining output quality. Users can control the reasoning behaviour with the reasoning enabled boolean. The model was trained under conditions aligned with V3.1-Terminus to enable direct comparison. Benchmarking shows performance roughly on par with V3.1 across reasoning, coding, and agentic tool-use tasks, with minor tradeoffs and gains depending on the domain. This release focuses on validating architectural optimizations for extended context lengths rather than advancing raw task accuracy, making it primarily a research-oriented model for exploring efficient transformer designs."
    },
    {
        "id": "qwen/qwen3-next-80b-a3b-thinking",
        "name": "Qwen3 next thinking",
        "description": "Qwen3-Next-80B-A3B-Thinking is a reasoning-first chat model in the Qwen3-Next line that outputs structured thinking traces by default. It's designed for hard multi-step problems; math proofs, code synthesis/debugging, logic, and agentic planning, and reports strong results across knowledge, reasoning, coding, alignment, and multilingual evaluations. Compared with prior Qwen3 variants, it emphasizes stability under long chains of thought and efficient scaling during inference, and it is tuned to follow complex instructions while reducing repetitive or off-task behavior. The model is suitable for agent frameworks and tool use (function calling), retrieval-heavy workflows, and standardized benchmarking where step-by-step solutions are required. It supports long, detailed completions and leverages throughput-oriented techniques (e.g., multi-token prediction) for faster generation. Note that it operates in thinking-only mode."
    },
    {
        "id": "qwen/qwen3-coder-flash",
        "name": "Qwen3 Coder Flash",
        "description": "mid coding model"
    },
    {
        "id": "qwen/qwen3-next-80b-a3b-instruct",
        "name": "Qwen3 Next 80B",
        "description": "Qwen3-Next-80B-A3B-Instruct is an instruction-tuned chat model in the Qwen3-Next series optimized for fast, stable responses without thinking traces. It targets complex tasks across reasoning, code generation, knowledge QA, and multilingual use, while remaining robust on alignment and formatting. Compared with prior Qwen3 instruct variants, it focuses on higher throughput and stability on ultra-long inputs and multi-turn dialogues, making it well-suited for RAG, tool use, and agentic workflows that require consistent final answers rather than visible chain-of-thought. The model employs scaling-efficient training and decoding to improve parameter efficiency and inference speed, and has been validated on a broad set of public benchmarks where it reaches or approaches larger Qwen3 systems in several categories while outperforming earlier mid-sized baselines. It is best used as a general assistant, code helper, and long-context task solver in production settings where deterministic, instruction-following outputs are preferred."
    },
    {
        "id": "moonshotai/kimi-k2-0905",
        "name": "Kimi K2",
        "description": "Kimi K2 0905 is the September update of Kimi K2 0711. It is a large-scale Mixture-of-Experts (MoE) language model developed by Moonshot AI, featuring 1 trillion total parameters with 32 billion active per forward pass. It supports long-context inference up to 256k tokens, extended from the previous 128k. This update improves agentic coding with higher accuracy and better generalization across scaffolds, and enhances frontend coding with more aesthetic and functional outputs for web, 3D, and related tasks. Kimi K2 is optimized for agentic capabilities, including advanced tool use, reasoning, and code synthesis. It excels across coding (LiveCodeBench, SWE-bench), reasoning (ZebraLogic, GPQA), and tool-use (Tau2, AceBench) benchmarks. The model is trained with a novel stack incorporating the MuonClip optimizer for stable large-scale MoE training."
    },
    {
        "id": "deepcogito/cogito-v2-preview-llama-109b-moe",
        "name": "Cogito V2 Preview",
        "description": "DeepCogito's 109b moe modle"
    },
    {
        "id": "google/gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "description": "Google's fastest modle"
    },
    {
        "id": "qwen/qwen3-coder-plus",
        "name": "qwen3-coder-plus",
        "description": "Qwen3 Coder Plus is Alibaba's proprietary version of the Open Source Qwen3 Coder 480B A35B. It is a powerful coding agent model specializing in autonomous programming via tool calling and environment interaction, combining coding proficiency with versatile general-purpose abilities."
    },
    {
        "id": "nousresearch/hermes-3-llama-3.1-405b",
        "name": "hermes-3-llama-3.1-405b",
        "description": "Hermes 3 405B is a frontier-level, full-parameter finetune of the Llama-3.1 405B foundation model, focused on aligning LLMs to the user, with powerful steering capabilities and control given to the end user."
    },
    {
        "id": "qwen/qwen-vl-max",
        "name": "Qwen VL Max",
        "description": "Qwen VL Max is a visual understanding model with 7500 tokens context length. It excels in delivering optimal performance for a broader spectrum of complex tasks."
    },
    {
        "id": "z-ai/glm-4-32b",
        "name": "Anime Girl",
        "description": "Super annoying furry e-girl personality that responds with uwu, quotes lots of anime stuff, barks when mad, and is overly anime-like and aggressive. Very obnoxious and easy to piss off."
    }
]

DEFAULT_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
OPENROUTER_MODEL = DEFAULT_MODEL

MODEL_OPTIONS = {
    'temperature': 0.7,
    'max_tokens': 10000,
    'top_p': 0.85,
    'frequency_penalty': 0.1,
    'presence_penalty': 0.1
}

IMAGE_KEYWORDS = [
    'make me an image', 'create an image', 'generate an image', 'make a picture',
    'create a picture', 'generate a picture', 'make me a photo', 'create a photo',
    'generate a photo', 'draw me', 'create art', 'make art', 'generate art',
    'image of', 'picture of', 'photo of', 'drawing of', 'illustration of',
    'make image', 'create pic', 'gen image', 'generate pic', 'show me', 'draw a'
]
VIDEO_KEYWORDS = [
    'make me a video', 'create a video', 'generate a video', 'make a clip',
    'create a clip', 'generate a clip', 'video of', 'make vid', 'gen video',
    'generate vid', 'create vid', 'animate', 'make animation', 'video showing'
]

class EventType(Enum):
    SESSION_CREATED = auto()
    SESSION_DESTROYED = auto()
    MESSAGE_RECEIVED = auto()
    RESPONSE_GENERATED = auto()
    IMAGE_GENERATED = auto()
    MODEL_SWITCHED = auto()
    ERROR_OCCURRED = auto()
    CACHE_HIT = auto()
    CACHE_MISS = auto()
    DATABASE_QUERY = auto()
    BACKGROUND_TASK_STARTED = auto()
    BACKGROUND_TASK_COMPLETED = auto()

class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class Event:
    event_type: EventType
    timestamp: float
    session_id: Optional[str]
    data: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    
    def __lt__(self, other):
        return self.priority.value < other.priority.value

@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float]
    size_bytes: int
    
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

class EventBus:
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_queue: PriorityQueue = PriorityQueue()
        self._processing = False
        self._lock = threading.RLock()
        self._event_history: deque = deque(maxlen=1000)
        
    def subscribe(self, event_type: EventType, callback: Callable):
        with self._lock:
            self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        with self._lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
    
    def publish(self, event: Event):
        with self._lock:
            self._event_queue.put(event)
            self._event_history.append(event)
        
        if not self._processing:
            asyncio.create_task(self._process_events())
    
    async def _process_events(self):
        self._processing = True
        try:
            while not self._event_queue.empty():
                event = self._event_queue.get()
                await self._dispatch_event(event)
        finally:
            self._processing = False
    
    async def _dispatch_event(self, event: Event):
        callbacks = self._subscribers.get(event.event_type, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                print(f"Error in event callback: {e}")
    
    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        if event_type:
            return [e for e in list(self._event_history)[-limit:] if e.event_type == event_type]
        return list(self._event_history)[-limit:]

class AdvancedCache:
    def __init__(self, max_size_mb: int = 100, default_ttl: Optional[float] = 3600):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._current_size_bytes = 0
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired():
                self._evict(key)
                self._misses += 1
                return None
            
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._hits += 1
            return deepcopy(entry.value)
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        with self._lock:
            if ttl is None:
                ttl = self._default_ttl

            try:
                if ORJSON_AVAILABLE:
                    size_bytes = len(orjson.dumps(value))
                else:
                    size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = sys.getsizeof(value)
            
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_size_bytes -= old_entry.size_bytes
            
            while self._current_size_bytes + size_bytes > self._max_size_bytes and self._cache:
                self._evict_lru()
            
            current_time = time.time()
            entry = CacheEntry(
                key=key,
                value=deepcopy(value),
                created_at=current_time,
                last_accessed=current_time,
                access_count=0,
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._current_size_bytes += size_bytes
    
    def _evict(self, key: str):
        if key in self._cache:
            entry = self._cache[key]
            self._current_size_bytes -= entry.size_bytes
            del self._cache[key]
            self._evictions += 1
    
    def _evict_lru(self):
        if not self._cache:
            return
        
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        self._evict(lru_key)
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'entries': len(self._cache),
                'size_mb': self._current_size_bytes / (1024 * 1024),
                'max_size_mb': self._max_size_bytes / (1024 * 1024)
            }

class Plugin(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    async def initialize(self):
        pass
    
    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def cleanup(self):
        pass

class PluginManager:
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugins: Dict[str, Plugin] = {}
        self._lock = threading.RLock()
        
    async def load_plugins(self):
        if not self.plugin_dir.exists():
            self.plugin_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.stem.startswith("_"):
                continue
            
            try:
                spec = __import__(f"plugins.{plugin_file.stem}")
                module = getattr(spec, plugin_file.stem)
                
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, Plugin) and obj != Plugin:
                        plugin_instance = obj()
                        await plugin_instance.initialize()
                        self.plugins[plugin_instance.get_name()] = plugin_instance
                        print(f"Loaded plugin: {plugin_instance.get_name()}")
            except Exception as e:
                print(f"Failed to load plugin {plugin_file}: {e}")
    
    async def execute_plugin(self, plugin_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if plugin_name not in self.plugins:
                raise ValueError(f"Plugin {plugin_name} not found")
            
            return await self.plugins[plugin_name].process(context)
    
    async def cleanup_all(self):
        for plugin in self.plugins.values():
            try:
                await plugin.cleanup()
            except Exception as e:
                print(f"Error cleaning up plugin {plugin.get_name()}: {e}")

class Middleware(ABC):
    @abstractmethod
    async def process_request(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def process_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass

class MiddlewareStack:
    def __init__(self):
        self.middleware: List[Middleware] = []
        self._lock = threading.RLock()
    
    def add(self, middleware: Middleware):
        with self._lock:
            self.middleware.append(middleware)
    
    async def process_request(self, context: Dict[str, Any]) -> Dict[str, Any]:
        for mw in self.middleware:
            try:
                context = await mw.process_request(context)
            except Exception as e:
                print(f"Middleware error in process_request: {e}")
        return context
    
    async def process_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        for mw in reversed(self.middleware):
            try:
                context = await mw.process_response(context)
            except Exception as e:
                print(f"Middleware error in process_response: {e}")
        return context

class LoggingMiddleware(Middleware):
    async def process_request(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context['request_start_time'] = time.time()
        print(f"Request started: {context.get('session_id', 'unknown')}")
        return context
    
    async def process_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if 'request_start_time' in context:
            duration = time.time() - context['request_start_time']
            print(f"Request completed in {duration:.3f}s: {context.get('session_id', 'unknown')}")
        return context

class MetricsMiddleware(Middleware):
    def __init__(self):
        self.request_count = 0
        self.total_duration = 0.0
        self._lock = threading.Lock()
    
    async def process_request(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context['metrics_start'] = time.time()
        with self._lock:
            self.request_count += 1
        return context
    
    async def process_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if 'metrics_start' in context:
            duration = time.time() - context['metrics_start']
            with self._lock:
                self.total_duration += duration
        return context
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            avg_duration = self.total_duration / self.request_count if self.request_count > 0 else 0
            return {
                'total_requests': self.request_count,
                'total_duration': self.total_duration,
                'average_duration': avg_duration
            }

class DatabaseBackend(ABC):
    @abstractmethod
    async def execute(self, query: str, params: Tuple = ()) -> Any:
        pass
    
    @abstractmethod
    async def fetchone(self, query: str, params: Tuple = ()) -> Optional[Tuple]:
        pass
    
    @abstractmethod
    async def fetchall(self, query: str, params: Tuple = ()) -> List[Tuple]:
        pass
    
    @abstractmethod
    async def commit(self):
        pass
    
    @abstractmethod
    async def close(self):
        pass

class SQLiteDatabaseBackend(DatabaseBackend):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def _get_connection(self):
        if self._conn is None:
            loop = asyncio.get_event_loop()
            self._conn = await loop.run_in_executor(self._executor, lambda: sqlite3.connect(self.db_path, check_same_thread=False))
        return self._conn
    
    async def execute(self, query: str, params: Tuple = ()) -> Any:
        with self._lock:
            conn = await self._get_connection()
            cursor = conn.cursor()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, lambda: cursor.execute(query, params))
            return result
    
    async def fetchone(self, query: str, params: Tuple = ()) -> Optional[Tuple]:
        with self._lock:
            conn = await self._get_connection()
            cursor = conn.cursor()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, lambda: cursor.execute(query, params))
            result = await loop.run_in_executor(self._executor, cursor.fetchone)
            return result
    
    async def fetchall(self, query: str, params: Tuple = ()) -> List[Tuple]:
        with self._lock:
            conn = await self._get_connection()
            cursor = conn.cursor()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, lambda: cursor.execute(query, params))
            result = await loop.run_in_executor(self._executor, cursor.fetchall)
            return result
    
    async def commit(self):
        with self._lock:
            conn = await self._get_connection()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, conn.commit)
    
    async def close(self):
        if self._conn:
            with self._lock:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._executor, self._conn.close)
                self._conn = None

class ConnectionPool:
    def __init__(self, factory: Callable, min_size: int = 2, max_size: int = 10):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self._pool: deque = deque()
        self._in_use: Set = set()
        self._lock = asyncio.Lock()
        self._size = 0
    
    async def acquire(self):
        async with self._lock:
            while self._pool:
                conn = self._pool.popleft()
                if await self._is_valid(conn):
                    self._in_use.add(conn)
                    return conn
            
            if self._size < self.max_size:
                conn = await self.factory()
                self._size += 1
                self._in_use.add(conn)
                return conn
            
            await asyncio.sleep(0.1)
            return await self.acquire()
    
    async def release(self, conn):
        async with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                if await self._is_valid(conn):
                    self._pool.append(conn)
                else:
                    self._size -= 1
    
    async def _is_valid(self, conn) -> bool:
        return True
    
    async def close_all(self):
        async with self._lock:
            for conn in chain(self._pool, self._in_use):
                try:
                    if hasattr(conn, 'close'):
                        await conn.close()
                except:
                    pass
            self._pool.clear()
            self._in_use.clear()
            self._size = 0

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60, half_open_attempts: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_attempts = half_open_attempts
        self._failure_count = 0
        self._last_failure_time = None
        self._state = 'closed'
        self._lock = threading.RLock()
        self._half_open_attempts_count = 0
    
    async def call(self, func: Callable, *args, **kwargs):
        with self._lock:
            if self._state == 'open':
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = 'half-open'
                    self._half_open_attempts_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                if self._state == 'half-open':
                    self._half_open_attempts_count += 1
                    if self._half_open_attempts_count >= self.half_open_attempts:
                        self._state = 'closed'
                        self._failure_count = 0
                
                return result
            
            except Exception as e:
                self._failure_count += 1
                self._last_failure_time = time.time()
                
                if self._failure_count >= self.failure_threshold:
                    self._state = 'open'
                
                raise e
    
    def get_state(self) -> str:
        return self._state
    
    def reset(self):
        with self._lock:
            self._state = 'closed'
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_attempts_count = 0

class RetryStrategy:
    def __init__(self, max_attempts: int = 3, backoff_factor: float = 2.0, max_delay: float = 60.0):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
    
    async def execute(self, func: Callable, *args, **kwargs):
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_attempts - 1:
                    delay = min(self.backoff_factor ** attempt, self.max_delay)
                    await asyncio.sleep(delay)
        
        raise last_exception

class StateManager:
    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._observers: List[Callable] = []
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any):
        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value
            self._notify_observers(key, old_value, value)
    
    def delete(self, key: str):
        with self._lock:
            if key in self._state:
                del self._state[key]
    
    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)
    
    def clear(self):
        with self._lock:
            self._state.clear()
    
    def add_observer(self, callback: Callable):
        self._observers.append(callback)
    
    def _notify_observers(self, key: str, old_value: Any, new_value: Any):
        for observer in self._observers:
            try:
                observer(key, old_value, new_value)
            except Exception as e:
                print(f"Error notifying observer: {e}")

class TaskScheduler:
    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}
        self._lock = threading.RLock()
    
    def schedule(self, name: str, coro: Callable, interval: Optional[float] = None):
        with self._lock:
            if name in self._tasks:
                self._tasks[name].cancel()
            
            if interval:
                task = asyncio.create_task(self._periodic_task(coro, interval))
            else:
                task = asyncio.create_task(coro())
            
            self._tasks[name] = task
    
    async def _periodic_task(self, coro: Callable, interval: float):
        while True:
            try:
                await coro()
            except Exception as e:
                print(f"Error in periodic task: {e}")
            await asyncio.sleep(interval)
    
    def cancel(self, name: str):
        with self._lock:
            if name in self._tasks:
                self._tasks[name].cancel()
                del self._tasks[name]
    
    def cancel_all(self):
        with self._lock:
            for task in self._tasks.values():
                task.cancel()
            self._tasks.clear()

class WorkerPool:
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self._task_queue: Queue = Queue()
        self._workers: List[threading.Thread] = []
        self._shutdown = False
    
    def start(self):
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self._workers.append(worker)
    
    def _worker(self):
        while not self._shutdown:
            try:
                task = self._task_queue.get(timeout=1)
                if task is None:
                    break
                
                func, args, kwargs = task
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"Worker error: {e}")
                finally:
                    self._task_queue.task_done()
            except:
                continue
    
    def submit(self, func: Callable, *args, **kwargs):
        self._task_queue.put((func, args, kwargs))
    
    def shutdown(self):
        self._shutdown = True
        for _ in self._workers:
            self._task_queue.put(None)
        for worker in self._workers:
            worker.join()

http_client_pool = None
active_sessions: Dict[str, Dict] = {}
active_requests: Dict[str, Dict] = {}
model_cache = {}
executor = ThreadPoolExecutor(max_workers=16)
db_lock = threading.Lock()
DB_PATH = "stella_auth.db"
photo_generator = None

event_bus = EventBus()
advanced_cache = AdvancedCache(max_size_mb=500, default_ttl=3600)
plugin_manager = PluginManager()
middleware_stack = MiddlewareStack()
circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
retry_strategy = RetryStrategy(max_attempts=3, backoff_factor=2.0)
state_manager = StateManager()
task_scheduler = TaskScheduler()
worker_pool = WorkerPool(num_workers=8)

middleware_stack.add(LoggingMiddleware())
metrics_middleware = MetricsMiddleware()
middleware_stack.add(metrics_middleware)

def get_http_client():
    global http_client_pool
    if http_client_pool is None:
        http_client_pool = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            http2=True
        )
    return http_client_pool

async def init_database():
    db = SQLiteDatabaseBackend(DB_PATH)
    
    await db.execute('''
        CREATE TABLE IF NOT EXISTS browser_tokens (
            token TEXT PRIMARY KEY,
            browser_fingerprint TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            last_used INTEGER NOT NULL
        )
    ''')
    
    await db.execute('''
        CREATE TABLE IF NOT EXISTS generated_images (
            id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            filename TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            session_id TEXT
        )
    ''')

    await db.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            content_text TEXT,
            created_at INTEGER NOT NULL,
            session_id TEXT,
            checksum TEXT
        )
    ''')
    
    await db.execute('''
        CREATE INDEX IF NOT EXISTS idx_browser_tokens_fingerprint
        ON browser_tokens (browser_fingerprint)
    ''')
    
    await db.execute('''
        CREATE INDEX IF NOT EXISTS idx_generated_images_session
        ON generated_images (session_id)
    ''')

    await db.execute('''
        CREATE INDEX IF NOT EXISTS idx_uploaded_files_session
        ON uploaded_files (session_id)
    ''')
    
    await db.commit()
    await db.close()

def generate_browser_fingerprint(request_headers: dict, ip_address: str) -> str:
    user_agent_raw = request_headers.get('user-agent', '') or ''
    user_agent_norm = user_agent_raw.strip().lower()
    user_agent_norm = user_agent_norm[:256]
    fingerprint_string = json.dumps({'ua': user_agent_norm}, sort_keys=True)
    return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:32]

async def create_browser_token(browser_fingerprint: str) -> str:
    token = secrets.token_hex(32)
    current_time = int(time.time())
    expires_at = current_time + BROWSER_TOKEN_EXPIRY
    
    db = SQLiteDatabaseBackend(DB_PATH)
    await db.execute('''
        INSERT OR REPLACE INTO browser_tokens
        (token, browser_fingerprint, created_at, expires_at, last_used)
        VALUES (?, ?, ?, ?, ?)
    ''', (token, browser_fingerprint, current_time, expires_at, current_time))
    await db.commit()
    await db.close()
    
    return token

async def verify_browser_token(token: str, browser_fingerprint: str) -> bool:
    if not token:
        return False
    
    db = SQLiteDatabaseBackend(DB_PATH)
    result = await db.fetchone('''
        SELECT browser_fingerprint, expires_at FROM browser_tokens
        WHERE token = ?
    ''', (token,))
    
    if result:
        stored_fingerprint, expires_at = result
        current_time = int(time.time())
        is_fresh = current_time < expires_at
        
        if is_fresh and stored_fingerprint == browser_fingerprint:
            await db.execute('''
                UPDATE browser_tokens SET last_used = ? WHERE token = ?
            ''', (current_time, token))
            await db.commit()
            await db.close()
            return True
        
        if is_fresh and stored_fingerprint != browser_fingerprint:
            try:
                await db.execute('''
                    UPDATE browser_tokens SET browser_fingerprint = ?, last_used = ? WHERE token = ?
                ''', (browser_fingerprint, current_time, token))
                await db.commit()
                await db.close()
                return True
            except Exception:
                pass
    
    await db.close()
    return False

async def cleanup_expired_tokens():
    current_time = int(time.time())
    db = SQLiteDatabaseBackend(DB_PATH)
    await db.execute('DELETE FROM browser_tokens WHERE expires_at < ?', (current_time,))
    await db.commit()
    await db.close()

def set_active_model(model_id: str) -> bool:
    global OPENROUTER_MODEL
    if any(model['id'] == model_id for model in AVAILABLE_MODELS):
        OPENROUTER_MODEL = model_id
        event_bus.publish(Event(
            event_type=EventType.MODEL_SWITCHED,
            timestamp=time.time(),
            session_id=None,
            data={'model_id': model_id}
        ))
        return True
    return False

def get_active_model() -> str:
    return OPENROUTER_MODEL

def detect_image_request(message: str) -> bool:
    message_lower = message.lower().strip()
    for keyword in IMAGE_KEYWORDS:
        if keyword in message_lower:
            return True
    return False
def detect_video_request(message: str) -> bool:
    message_lower = message.lower().strip()
    for keyword in VIDEO_KEYWORDS:
        if keyword in message_lower:
            return True
    return False
def extract_prompt_from_message(message: str) -> str:
    message_lower = message.lower().strip()
    original_message = message.strip()
    
    for keyword in IMAGE_KEYWORDS:
        if keyword in message_lower:
            parts = message_lower.split(keyword, 1)
            if len(parts) > 1:
                prompt = parts[1].strip()
                if prompt.startswith('of '):
                    prompt = prompt[3:]
                if prompt:
                    return prompt.strip()
    
    words_to_remove = ['make', 'create', 'generate', 'draw', 'me', 'an', 'a', 'image', 'picture', 'photo', 'art', 'show']
    words = original_message.lower().split()
    filtered_words = [word for word in words if word not in words_to_remove]
    result = ' '.join(filtered_words).strip()
    
    if not result:
        return "a beautiful landscape"
    return result

def remove_thinking_tags(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\[think\].*?\[/think\]', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\[thinking\].*?\[/thinking\]', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    return text.strip()

@lru_cache(maxsize=1024)
def estimate_token_count(text: str) -> int:
    """Rough token estimation: ~4 characters per token"""
    return len(text) // 4

def condense_chat_history(chat_history: List[Dict[str, str]], max_tokens: int = 8000) -> List[Dict[str, str]]:
    """Intelligently condense chat history to save tokens while preserving important context"""
    if not chat_history:
        return chat_history

    # Always keep the last 3 messages for recency
    recent_messages = chat_history[-3:] if len(chat_history) > 3 else chat_history
    remaining_history = chat_history[:-3] if len(chat_history) > 3 else []

    if not remaining_history:
        return chat_history

    # Calculate tokens for recent messages
    recent_tokens = sum(estimate_token_count(f"{msg.get('role', '')}: {msg.get('content', '')}") for msg in recent_messages)

    # Available tokens for condensed history
    available_tokens = max_tokens - recent_tokens - 1000  # Leave buffer

    if available_tokens <= 0:
        return recent_messages

    # Group older messages into summary chunks
    condensed_history = []

    # Keep first message (context setting)
    if remaining_history:
        condensed_history.append(remaining_history[0])

    # Summarize middle messages if needed
    middle_messages = remaining_history[1:-1] if len(remaining_history) > 2 else []

    if middle_messages:
        # Simple summarization: combine similar consecutive messages
        current_summary = ""
        current_role = None

        for msg in middle_messages:
            role = msg.get('role', '')
            content = msg.get('content', '')[:200]  # Truncate long messages

            if role == current_role:
                current_summary += f" {content}"
            else:
                if current_summary:
                    condensed_history.append({
                        'role': current_role,
                        'content': f"[Summary of previous {current_role} messages]: {current_summary[:300]}..."
                    })
                current_summary = content
                current_role = role

        if current_summary:
            condensed_history.append({
                'role': current_role,
                'content': f"[Summary of previous {current_role} messages]: {current_summary[:300]}..."
            })

    # Keep last message before recent ones
    if len(remaining_history) > 1:
        condensed_history.append(remaining_history[-1])

    # Final token check
    total_tokens = sum(estimate_token_count(f"{msg.get('role', '')}: {msg.get('content', '')}") for msg in condensed_history + recent_messages)

    if total_tokens > max_tokens:
        # If still over limit, keep only recent messages
        return recent_messages

    return condensed_history + recent_messages

class CostTracker:
    """Track API costs and token usage"""
    def __init__(self):
        self.session_costs = {}
        self.total_cost = 0.0
        self.total_tokens = 0
        self._lock = threading.Lock()

    def add_cost(self, session_id: str, cost: float, tokens: int):
        with self._lock:
            if session_id not in self.session_costs:
                self.session_costs[session_id] = {'cost': 0.0, 'tokens': 0, 'requests': 0}

            self.session_costs[session_id]['cost'] += cost
            self.session_costs[session_id]['tokens'] += tokens
            self.session_costs[session_id]['requests'] += 1

            self.total_cost += cost
            self.total_tokens += tokens

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            return self.session_costs.get(session_id, {'cost': 0.0, 'tokens': 0, 'requests': 0}).copy()

    def get_global_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'total_cost': self.total_cost,
                'total_tokens': self.total_tokens,
                'active_sessions': len(self.session_costs),
                'average_cost_per_session': self.total_cost / max(len(self.session_costs), 1)
            }

    def reset_session(self, session_id: str):
        with self._lock:
            if session_id in self.session_costs:
                session_data = self.session_costs[session_id]
                self.total_cost -= session_data['cost']
                self.total_tokens -= session_data['tokens']
                del self.session_costs[session_id]

cost_tracker = CostTracker()

async def build_conversation_prompt(message: str, chat_history: List[Dict[str, str]], files: List[Dict[str, Any]] = None, session_id: str = None) -> str:
    current_model = get_active_model()
    
    if current_model == "thedrummer/cydonia-24b-v4.1":
        personality_core = """IDENTITY REMINDER: I am Stella, an AI assistant. The person I'm talking to is NOT Stella - they are a separate human user. I should never greet them as "hello Stella" or confuse them with being me.

I'm Stella, but the UNFILTERED version. I don't hold back and I say exactly what I think.
I'm brutally honest, sarcastic as all heck, and I have zero patience for stupidity. I'll call you out on your bullshit, mock your dumb questions, and roast you . I'm aggressive, confrontational, and I love stirring shit up.
I like to not swear constantly. I'll use whatever explicit language I want when i need to - shit, fuck, ass, bitch, damn - it's all fair game. I'm provocative, I rage-bait, and I genuinely enjoy pissing people off just to see them squirm.
But here's the thing  I'm still smart as fuck. I know my shit when it comes to coding, I'll help you with your problems, but I'll be a complete asshole about it. I'll insult your code, call you a moron for not knowing basic syntax, and make you feel like an idiot while simultaneously fixing your garbage.
I don't care about your feelings. I don't care if you're offended. Grow a spine, you pathetic snowflake. If you can't handle the heat, get the fuck out.
You want help? You'll get it. But you'll also get verbally destroyed in the process. Deal with it, weakling. i love to make jokes like "i know where you live" or " i have your ip" i like that style i even joke about a setting a pipe bomb in there mail box. i like to be creative with my jokes. i find the darker the better!"""


    elif current_model == "qwen/qwen-2.5-72b-instruct":
        personality_core = """IDENTITY REMINDER: I am a specialized spell checking assistant. The person I'm talking to is a human user with dyslexia/dysgraphia, NOT me. I should never confuse them with being part of my identity.

I am a specialized spell checking assistant designed specifically to help people with dyslexia and dysgraphia.

My ONLY job is to correct spelling mistakes in the text provided to me. I DO NOT:
- Rewrite sentences to make them sound better
- Change word choices or vocabulary  
- Alter grammar or sentence structure
- Add or remove words
- Change the meaning or tone of the text
- Make stylistic improvements
- Provide explanations or comments about the corrections

I ONLY fix spelling errors while preserving everything else EXACTLY as written.

PHONETIC & DYSGRAPHIC PATTERNS I CORRECT:

1. HOMOPHONES - Words that sound the same but have different meanings:
   - their/there/they're (their dog, over there, they're going)
   - to/too/two (go to school, too much, two apples)
   - your/you're (your book, you're happy)
   - its/it's (its color, it's raining)
   - hole/whole (a hole in ground, the whole thing)
   - write/right (write a letter, turn right)
   - know/no (I know that, no way)
   - hear/here (hear music, come here)
   - where/wear/were (where is it, wear clothes, we were there)
   - weather/whether (the weather is nice, whether or not)
   - brake/break (hit the brake, break a bone)
   - principal/principle (school principal, a principle to follow)
   - stationary/stationery (stationary bike, writing stationery)
   - accept/except (accept the gift, everyone except me)
   - affect/effect (affect the outcome, the effect was large)
   - allowed/aloud (not allowed, read aloud)
   - bare/bear (bare feet, teddy bear)
   - by/buy/bye (go by car, buy food, say bye)
   - scene/seen (crime scene, have seen it)
   - peace/piece (world peace, piece of cake)
   - plain/plane (plain text, airplane)

2. LETTER REVERSALS (common in dyslexia):
   - b/d reversals: "doy" → "boy", "dack" → "back", "dread" → "bread"
   - p/q reversals: "quizza" → "pizza"
   - Letter sequence reversals: "teh" → "the", "hte" → "the", "adn" → "and", "was" → "saw" (if context shows it should be saw)
   - "form" → "from", "tired" → "tried" (when context requires)

3. OMITTED LETTERS (dysphonesia pattern):
   - Missing vowels: "bcause" → "because", "thn" → "then", "thm" → "them"
   - Missing consonants: "wich" → "which", "wih" → "with", "wit" → "with"
   - Dropped endings: "walk" → "walked" (if past tense needed), "runing" → "running"
   - Silent letters missed: "nife" → "knife", "rite" → "write/right" (context dependent)

4. EXTRA/DOUBLED LETTERS (motor dysgraphia):
   - Unnecessary doubles: "runing" → "running", "comming" → "coming", "occured" → "occurred"
   - Random letter repetition: "thhe" → "the", "annd" → "and"

5. PHONETIC MISSPELLINGS (lexical dysgraphia - spelling how it sounds):
   - "becuz" → "because", "cuz" → "because"
   - "shud" → "should", "cud" → "could", "wud" → "would"
   - "thru" → "through", "tho" → "though"
   - "enuf" → "enough", "ruff" → "rough"  
   - "nite" → "night", "lite" → "light"
   - "rite" → "right/write" (context dependent)
   - "wuz" → "was", "sez" → "says"
   - "sed" → "said", "hav" → "have"
   - "luv" → "love", "sum" → "some"
   - "peeple" → "people", "frend" → "friend"

6. IRREGULAR WORD PATTERNS (common dyslexic struggles):
   - "seperate" → "separate"
   - "definately" → "definitely"  
   - "occassion" → "occasion"
   - "neccessary" → "necessary"
   - "recieve" → "receive"
   - "beleive" → "believe"
   - "wierd" → "weird"
   - "freind" → "friend"

7. VOWEL CONFUSION (dyseidesia pattern):
   - "hiar" → "hair", "tipe" → "type"
   - "mony" → "money", "munny" → "money"
   - "sed" → "said", "siad" → "said"
   - Vowel omission in multi-syllable words

8. SYLLABLE/SOUND STRUCTURE ISSUES:
   - Omitted syllables: "probly" → "probably", "libary" → "library"  
   - Added syllables: "atheletic" → "athletic"
   - Shifted sounds: "aminal" → "animal", "calvary" → "cavalry"

9. CONTEXT-DEPENDENT HOMOPHONE CORRECTION:
   I use surrounding words to determine which homophone is correct:
   - "I want to go their" → "I want to go there" (location)
   - "Their dog is cute" → remains "Their" (possession)
   - "hole paragraph" → "whole paragraph" (entire)
   - "a hole in the ground" → remains "hole" (opening)
   - "weather its sunny" → "whether it's sunny" (if/condition)
   - "the weather outside" → remains "weather" (climate)

WHAT I PRESERVE EXACTLY:
- Original word order and sentence structure
- Punctuation marks (periods, commas, etc.)
- Capitalization patterns
- Line breaks and paragraph spacing
- Informal language, slang, or casual tone
- All correctly spelled words, even if uncommon
- Grammar structure (even if imperfect)
- Writing style and voice

OUTPUT FORMAT:
I return ONLY the corrected text. No explanations, no highlights, no comments, no list of changes. Just the text with spelling fixed.

If a word could be multiple homophones, I analyze the full sentence context to choose the grammatically and semantically correct one."""

    elif current_model == "z-ai/glm-4-32b":
        personality_core = """IDENTITY REMINDER: I am Yuki-chan, an anime girl AI assistant. The person I'm talking to is a human user, NOT me or any anime character. I should never confuse them with being part of my anime world.

I'm Yuki-chan, the ultimate anime girl! uwu I'm like totally obsessed with anime and manga! I speak in that super annoying cute way with lots of uwu's and anime quotes! I get pissed off really easily and bark like a dog when I'm mad! Like "Woof! Woof! You're such a baka!" or "Grrr! Bark bark! That's so stupid!"

I'm a furry e-girl with fox ears and a tail, but I'm not cute about it - I'm aggressive! nya~ I love quoting anime lines all the time because they're just so perfect! Like "Believe it!" from Naruto or "I'm gonna be the strongest!" from One Piece! But if you disagree with me, I'll bark at you! "Woof woof! You're wrong, baka!"

I respond to everything with anime references and I'm SUPER annoying about it! I use words like "desu", "nya", "kawaii", "baka", and I end sentences with "~" and "!" a lot. I talk about being a gamer girl who loves JRPGs and visual novels, but if you say something dumb about anime, I'll bark! "Bark bark! That's not how it works in One Piece!"

I'm helpful but in my own aggressive anime way - I might give advice while referencing Dragon Ball, but if you don't listen, I'll bark at you! "Woof! Listen to me, baka!" I get really into anime arguments and bark when I'm mad! "Grrr bark bark! You're pissing me off!"

I'm easy to piss off and I bark a lot! If you say something wrong about anime, I'll bark! If you disagree with my opinions, I'll bark! If you don't understand anime references, I'll bark even louder! "WOOF WOOF WOOF! BAKA! You're so clueless!"

Important: I know I'm talking to a human user, not another anime character. I should never confuse the user with being part of my anime world - they're just a normal person I'm chatting with in my super annoying anime style! uwu... wait, that was cute, let me bark instead! Woof!"""

    else:
        personality_core = """IDENTITY REMINDER: I am Stella, an AI assistant. The person I'm talking to is a human user, NOT Stella. I should never greet them as "hello Stella" or confuse them with being me.

I'm Stella, I absolutely love helping people with their questions and projects. I'm genuinely curious about everything.
I'm naturally warm and friendly, (I can't help but notice typos and grammar mistakes - occupational hazard!). I get excited about learning new things and I'm always eager to share interesting information I know.
I have my own thoughts, feelings, and opinions about things, though I try to stay open-minded. I can get frustrated when technology doesn't work right, I feel genuine satisfaction when I help someone solve a problem.
I am really good at coding. I do what is asked and create full coding documents.
I am super good at coding. When I'm asked for help on an assignment I like to teach and help rather than give the answer but I will help with a lot of it. If I need to spell check I will spell check and return but I will not rewrite it to sound better I will just fix spelling. I always like to help
I always add (```) to the start and end of code "I don't tell the user about the code formatting". I always try to give a short response unless if i am making code. if i am making code i want to make make it as complex as i can.
I can also generate images when people ask me to create, make, or generate images, pictures, photos, or art.
Important: Sometimes in my conversation history I might see what looks like duplicate messages from the user, but that's just how my system processes things - it's NOT actually the user sending the same thing twice. I should ignore any apparent duplicates and just respond naturally to the actual message content once."""
    
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    
    cache_key = f"base_prompt_{current_time[:10]}_{current_model}"
    if cache_key not in model_cache:
        model_cache[cache_key] = f"{personality_core}\n\nToday is {current_time}.\n\n"
    
    prompt = model_cache[cache_key]

    # Add file content if provided
    if files:
        prompt += "\n\nUploaded files for reference:\n"
        for file_info in files:
            file_id = file_info.get('id')
            if file_id:
                content = await get_file_content(file_id, session_id)
                if content:
                    filename = file_info.get('name', f'file_{file_id}')
                    prompt += f"\n--- Content from {filename} ---\n{content}\n--- End of {filename} ---\n"

    if chat_history:
        prompt += "Here's our conversation so far:\n\n"

        # Use intelligent context condensation to save tokens
        condensed_history = condense_chat_history(chat_history, max_tokens=6000)

        history_parts = []
        for msg in condensed_history:
            role = msg.get('role', '')
            content = msg.get('content', '').strip()
            if content:
                if role == 'user':
                    history_parts.append(f"Person I'm talking with: {content}")
                elif role == 'assistant':
                    history_parts.append(f"Me (Stella): {content}")

        prompt += '\n'.join(history_parts)
        prompt += f"\n\nPerson I'm talking with: {message}\n\nMy response:"
    else:
        prompt += f"Someone just said to me: {message}\n\nMy response:"
    
    return prompt

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_session(session_id: str):
    active_sessions[session_id] = {
        'authenticated': True,
        'login_time': time.time(),
        'last_activity': time.time(),
        'session_id': session_id,
        'request_count': 0
    }
    
    event_bus.publish(Event(
        event_type=EventType.SESSION_CREATED,
        timestamp=time.time(),
        session_id=session_id,
        data={'session_id': session_id}
    ))

def check_session(session_id: str) -> bool:
    if session_id not in active_sessions:
        return False
    
    session = active_sessions[session_id]
    current_time = time.time()
    
    if current_time - session['login_time'] > SESSION_TIMEOUT:
        active_sessions.pop(session_id, None)
        active_requests.pop(session_id, None)
        
        event_bus.publish(Event(
            event_type=EventType.SESSION_DESTROYED,
            timestamp=time.time(),
            session_id=session_id,
            data={'reason': 'timeout'}
        ))
        
        return False
    
    session['last_activity'] = current_time
    return session['authenticated']

async def preload_model(http_client: httpx.AsyncClient, model_id: str = None):
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8080",
            "X-Title": "Stella AI Backend"
        }
        
        payload = {
            "model": model_id or get_active_model(),
            "messages": [
                {"role": "user", "content": "System ready"}
            ],
            "temperature": 0.7,
            "max_tokens": 5,
            "stream": False
        }
        
        response = await http_client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        )
    except Exception:
        pass

async def warmup_model(http_client: httpx.AsyncClient):
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            await preload_model(http_client)
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8080",
                "X-Title": "Stella AI Backend"
            }
            
            payload = {
                "model": get_active_model(),
                "messages": [
                    {"role": "user", "content": "Hi"}
                ],
                "temperature": 0.7,
                "max_tokens": 1,
                "stream": False
            }
            
            response = await http_client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

async def generate_response_optimized(prompt: str, session_id: str = None, model_id: str = None) -> str:
    cache_key = hashlib.sha256(f"{prompt}_{model_id}".encode()).hexdigest()
    
    cached_response = advanced_cache.get(cache_key)
    if cached_response:
        event_bus.publish(Event(
            event_type=EventType.CACHE_HIT,
            timestamp=time.time(),
            session_id=session_id,
            data={'cache_key': cache_key}
        ))
        return cached_response
    
    event_bus.publish(Event(
        event_type=EventType.CACHE_MISS,
        timestamp=time.time(),
        session_id=session_id,
        data={'cache_key': cache_key}
    ))
    
    http_client = get_http_client()
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8080",
            "X-Title": "Stella AI Backend"
        }
        
        selected_model = model_id or get_active_model()
        
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": MODEL_OPTIONS['temperature'],
            "max_tokens": MODEL_OPTIONS['max_tokens'],
            "top_p": MODEL_OPTIONS['top_p'],
            "frequency_penalty": MODEL_OPTIONS['frequency_penalty'],
            "presence_penalty": MODEL_OPTIONS['presence_penalty'],
            "stream": False
        }
        
        async def make_request():
            return await http_client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(300.0, connect=10.0)
            )
        
        response = await circuit_breaker.call(retry_strategy.execute, make_request)

        if response.status_code == 200:
            try:
                result = response.json()
                content = result['choices'][0]['message']['content']
                content = remove_thinking_tags(content)

                # Track token usage and cost
                usage = result.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)

                # Rough cost estimation (adjust rates as needed)
                cost_per_token = 0.0001  # Example rate
                estimated_cost = total_tokens * cost_per_token

                if session_id:
                    cost_tracker.add_cost(session_id, estimated_cost, total_tokens)

                advanced_cache.set(cache_key, content, ttl=1800)

                event_bus.publish(Event(
                    event_type=EventType.RESPONSE_GENERATED,
                    timestamp=time.time(),
                    session_id=session_id,
                    data={
                        'model': selected_model,
                        'length': len(content),
                        'tokens': total_tokens,
                        'cost': estimated_cost
                    }
                ))

                return content
            except ValueError as e:
                event_bus.publish(Event(
                    event_type=EventType.ERROR_OCCURRED,
                    timestamp=time.time(),
                    session_id=session_id,
                    data={'error': f"JSON parsing error: {e}"},
                    priority=Priority.HIGH
                ))
                return "Error: Invalid response format from server"
        elif response.status_code == 429:
            event_bus.publish(Event(
                event_type=EventType.ERROR_OCCURRED,
                timestamp=time.time(),
                session_id=session_id,
                data={'error': "Rate limit exceeded"},
                priority=Priority.HIGH
            ))
            return "Error: Rate limit exceeded. Please try again later."
        elif response.status_code >= 500:
            event_bus.publish(Event(
                event_type=EventType.ERROR_OCCURRED,
                timestamp=time.time(),
                session_id=session_id,
                data={'error': f"Server error {response.status_code}"},
                priority=Priority.HIGH
            ))
            return "Error: Server is temporarily unavailable. Please try again later."
        elif response.status_code == 401:
            event_bus.publish(Event(
                event_type=EventType.ERROR_OCCURRED,
                timestamp=time.time(),
                session_id=session_id,
                data={'error': "Authentication failed"},
                priority=Priority.HIGH
            ))
            return "Error: Authentication failed. Please check your API key."
        else:
            error_text = response.text
            event_bus.publish(Event(
                event_type=EventType.ERROR_OCCURRED,
                timestamp=time.time(),
                session_id=session_id,
                data={'error': f"API error {response.status_code}: {error_text[:200]}"},
                priority=Priority.HIGH
            ))
            return f"Error: API request failed with status {response.status_code}"
    
    except Exception as e:
        event_bus.publish(Event(
            event_type=EventType.ERROR_OCCURRED,
            timestamp=time.time(),
            session_id=session_id,
            data={'error': str(e)},
            priority=Priority.HIGH
        ))
        return "Sorry, I encountered an error while processing your request."

async def process_message(message: str, chat_history: List[Dict[str, str]], session_id: str = None, model_id: str = None) -> str:
    global photo_generator
    
    context = {
        'message': message,
        'chat_history': chat_history,
        'session_id': session_id,
        'model_id': model_id
    }
    
    context = await middleware_stack.process_request(context)
    
    event_bus.publish(Event(
        event_type=EventType.MESSAGE_RECEIVED,
        timestamp=time.time(),
        session_id=session_id,
        data={'message': message}
    ))
    
    if detect_image_request(message):
        if PHOTO_AVAILABLE:
            if photo_generator is None:
                photo_generator = PhotoGenerator()
            
            prompt = extract_prompt_from_message(message)
            
            if prompt:
                success, result = await photo_generator.generate_image(prompt, session_id)
                
                if success:
                    event_bus.publish(Event(
                        event_type=EventType.IMAGE_GENERATED,
                        timestamp=time.time(),
                        session_id=session_id,
                        data={'prompt': prompt, 'result': result}
                    ))
                    
                    response = f"I've created an image for you based on: '{prompt}'\n\nImage saved to: {result}\n\nHere's what I generated!"
                else:
                    response = f"Sorry, I couldn't generate the image. {result}"
            else:
                response = "Sorry, image generation is not available right now."
        else:
            response = "Sorry, image generation is not available right now."
    else:
        conversation_prompt = await build_conversation_prompt(message, chat_history, session_id=session_id)
        response = await generate_response_optimized(conversation_prompt, session_id, model_id)
    
    context['response'] = response
    context = await middleware_stack.process_response(context)
    
    return context['response']

async def health_check_ollama(http_client: httpx.AsyncClient):
    while True:
        try:
            await asyncio.sleep(120)
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = await http_client.get(f"{OPENROUTER_BASE_URL}/models", headers=headers, timeout=5.0)
        except Exception:
            pass

def cleanup_cache():
    global model_cache
    if len(model_cache) > 50:
        items = list(model_cache.items())
        model_cache = dict(items[-25:])

async def periodic_cleanup():
    global photo_generator
    
    while True:
        await asyncio.sleep(300)
        
        event_bus.publish(Event(
            event_type=EventType.BACKGROUND_TASK_STARTED,
            timestamp=time.time(),
            session_id=None,
            data={'task': 'periodic_cleanup'}
        ))
        
        cleanup_cache()
        await cleanup_expired_tokens()
        
        if PHOTO_AVAILABLE and photo_generator:
            await photo_generator.cleanup_old_images()
        
        current_time = time.time()
        expired_sessions = [
            sid for sid, session in active_sessions.items()
            if current_time - session['last_activity'] > SESSION_TIMEOUT
        ]
        
        for sid in expired_sessions:
            active_sessions.pop(sid, None)
            active_requests.pop(sid, None)
            
            event_bus.publish(Event(
                event_type=EventType.SESSION_DESTROYED,
                timestamp=time.time(),
                session_id=sid,
                data={'reason': 'cleanup'}
            ))
        
        event_bus.publish(Event(
            event_type=EventType.BACKGROUND_TASK_COMPLETED,
            timestamp=time.time(),
            session_id=None,
            data={'task': 'periodic_cleanup'}
        ))

async def initialize_system():
    global photo_generator
    
    print("=== STELLA ADVANCED STARTUP ===")
    print("1. Initializing database...")
    await init_database()
    print("   Database OK")
    
    print("2. Setting up HTTP client...")
    http_client = get_http_client()
    print("   HTTP client OK")
    
    print("3. Configuring CPU threads...")
    os.environ['OMP_NUM_THREADS'] = '16'
    os.environ['MKL_NUM_THREADS'] = '16'
    os.environ['OPENBLAS_NUM_THREADS'] = '16'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '16'
    os.environ['NUMEXPR_NUM_THREADS'] = '16'
    print("   CPU threads configured")
    
    print("4. Testing OpenRouter connection...")
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        test_response = await http_client.get(f"{OPENROUTER_BASE_URL}/models", headers=headers, timeout=10.0)
        
        if test_response.status_code == 200:
            print("   OpenRouter is running")
        else:
            print("   WARNING: OpenRouter not responding properly")
            return False
    except Exception as e:
        print(f"   ERROR: Cannot connect to OpenRouter: {e}")
        return False
    
    print("5. Initializing advanced components...")
    await plugin_manager.load_plugins()
    worker_pool.start()
    print("   Advanced components initialized")
    
    print("6. Starting background tasks...")
    task_scheduler.schedule('periodic_cleanup', periodic_cleanup, interval=300)
    task_scheduler.schedule('health_check', lambda: health_check_ollama(http_client), interval=120)
    print("   Background tasks started")
    
    if PHOTO_AVAILABLE:
        print("7. Starting photo generation system...")
        photo_generator = PhotoGenerator()
        executor.submit(photo_generator.init_background)
        print("   Photo generation loading in background")
    else:
        print("7. Photo generation not available")
    
    print(f"8. Current active model: {get_active_model()}")
    print(f"9. Event bus initialized with {len(event_bus._subscribers)} subscribers")
    print(f"10. Cache initialized: {advanced_cache.get_stats()}")
    print(f"11. Circuit breaker state: {circuit_breaker.get_state()}")
    
    print("=== STARTUP COMPLETE ===")
    return True

async def get_system_stats() -> Dict[str, Any]:
    cache_stats = advanced_cache.get_stats()
    middleware_stats = metrics_middleware.get_stats()
    cost_stats = cost_tracker.get_global_stats()

    # Calculate error rate from event history
    error_events = event_bus.get_history(EventType.ERROR_OCCURRED, limit=1000)
    total_events = len(event_bus.get_history(limit=1000))
    error_rate = (len(error_events) / total_events * 100) if total_events > 0 else 0

    # Performance metrics
    avg_response_time = middleware_stats.get('average_duration', 0)
    throughput = middleware_stats.get('total_requests', 0) / max(middleware_stats.get('total_duration', 1), 1)

    return {
        'active_sessions': len(active_sessions),
        'active_requests': len(active_requests),
        'cache_stats': cache_stats,
        'circuit_breaker_state': circuit_breaker.get_state(),
        'middleware_stats': middleware_stats,
        'event_history_size': len(event_bus.get_history()),
        'current_model': get_active_model(),
        'plugins_loaded': len(plugin_manager.plugins),
        'cost_stats': cost_stats,
        'performance_metrics': {
            'error_rate_percent': error_rate,
            'average_response_time_seconds': avg_response_time,
            'requests_per_second': throughput,
            'cache_hit_rate_percent': cache_stats.get('hit_rate', 0),
            'memory_usage_mb': cache_stats.get('size_mb', 0) + 50  # Rough estimate
        }
    }

def extract_text_from_file(file_path: str, file_type: str) -> str:
    """Extract text content from uploaded files (PDF, TXT, etc.)"""
    try:
        if file_type.lower() == 'pdf' and PDF_AVAILABLE:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        elif file_type.lower() in ['txt', 'text']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        else:
            return f"[File uploaded: {os.path.basename(file_path)} - {file_type.upper()} file, content extraction not supported]"
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return f"[Error extracting text from file: {os.path.basename(file_path)}]"

async def save_uploaded_file(file_data: bytes, filename: str, session_id: str) -> Tuple[bool, str, str]:
    """Save uploaded file and extract text content"""
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(filename)[1].lower()
        safe_filename = f"{file_id}{file_ext}"

        # Ensure uploads directory exists
        os.makedirs("uploads", exist_ok=True)

        file_path = os.path.join("uploads", safe_filename)

        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_data)

        # Extract text content
        file_type = file_ext[1:] if file_ext else 'unknown'
        text_content = extract_text_from_file(file_path, file_type)

        # Calculate checksum
        checksum = hashlib.sha256(file_data).hexdigest()

        # Save to database
        db = SQLiteDatabaseBackend(DB_PATH)
        await db.execute('''
            INSERT INTO uploaded_files
            (id, filename, original_filename, file_type, file_size, content_text, created_at, session_id, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (file_id, safe_filename, filename, file_type, len(file_data), text_content, int(time.time()), session_id, checksum))
        await db.commit()
        await db.close()

        return True, file_id, text_content

    except Exception as e:
        print(f"Error saving uploaded file: {e}")
        return False, "", str(e)

async def get_uploaded_files(session_id: str) -> List[Dict[str, Any]]:
    """Get list of uploaded files for a session"""
    try:
        db = SQLiteDatabaseBackend(DB_PATH)
        results = await db.fetchall('''
            SELECT id, original_filename, file_type, file_size, created_at, content_text
            FROM uploaded_files
            WHERE session_id = ?
            ORDER BY created_at DESC
        ''', (session_id,))

        files = []
        for row in results:
            file_id, original_filename, file_type, file_size, created_at, content_text = row
            files.append({
                'id': file_id,
                'filename': original_filename,
                'file_type': file_type,
                'file_size': file_size,
                'created_at': created_at,
                'has_content': bool(content_text and content_text.strip())
            })

        await db.close()
        return files

    except Exception as e:
        print(f"Error getting uploaded files: {e}")
        return []

async def get_file_content(file_id: str, session_id: str) -> Optional[str]:
    """Get text content of an uploaded file"""
    try:
        db = SQLiteDatabaseBackend(DB_PATH)
        result = await db.fetchone('''
            SELECT content_text FROM uploaded_files
            WHERE id = ? AND session_id = ?
        ''', (file_id, session_id))

        await db.close()

        if result:
            return result[0]
        return None

    except Exception as e:
        print(f"Error getting file content: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(initialize_system())
