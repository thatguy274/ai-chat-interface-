#!/usr/bin/env python3

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
import httpx
import json
import time
import queue
import hashlib
import hmac
from datetime import datetime, timedelta
import secrets
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator, Set, Tuple
import uvicorn
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import structlog
import orjson
import lz4.frame
from cryptography.fernet import Fernet
from enum import Enum
import uuid
from collections import defaultdict, deque
import psutil
import base64
import os  # <-- Add this import

from stbrain import (
    OPENROUTER_API_KEY, ADMIN_PASSWORD_HASH, SESSION_TIMEOUT, OPENROUTER_BASE_URL, 
    OPENROUTER_MODEL, AVAILABLE_MODELS, DEFAULT_MODEL, MODEL_OPTIONS, active_sessions, active_requests, 
    build_conversation_prompt, verify_password, create_session, check_session, 
    warmup_model, generate_browser_fingerprint,
    create_browser_token, verify_browser_token, init_database, cleanup_expired_tokens,
    process_message, detect_image_request, extract_prompt_from_message, set_active_model, get_active_model
)

try:
    from stbrain import PHOTO_AVAILABLE
    if PHOTO_AVAILABLE:
        print("Image generation integration enabled")
    else:
        print("Image generation not available")
except:
    PHOTO_AVAILABLE = False
    print("Image generation import check failed")

try:
    from stphoto import photo_generator, IMAGE_STEPS
except ImportError:
    photo_generator = None
    IMAGE_STEPS = 18
    print("Photo generator not available")

from stcoding import CodeBlockTracker

logger = structlog.get_logger()

class MessageType(str, Enum):
    CHAT = "chat"
    CODE = "code"
    SYSTEM = "system"
    ERROR = "error"

class Priority(int, Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class EnhancedChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    history: List[Dict[str, str]] = Field(default_factory=list, max_length=100)
    message_type: MessageType = MessageType.CHAT
    priority: Priority = Priority.NORMAL
    context: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    streaming: bool = True
    files: List[Dict[str, Any]] = Field(default_factory=list, max_length=10)
    
    @field_validator('history')
    @classmethod
    def validate_history(cls, v):
        for item in v:
            if not isinstance(item, dict) or 'role' not in item or 'content' not in item:
                raise ValueError("History items must have 'role' and 'content' fields")
            if item['role'] not in ['user', 'assistant', 'system']:
                raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return v

class LoginRequest(BaseModel):
    password: str = Field(..., min_length=1)
    remember_me: bool = False
    device_info: Optional[Dict[str, Any]] = None

class BrowserAuthRequest(BaseModel):
    browser_token: Optional[str] = None
    fingerprint_data: Optional[Dict[str, Any]] = None

class WebSocketMessage(BaseModel):
    type: str
    data: Any
    timestamp: Optional[str] = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.connection_metadata[session_id] = {
            'connected_at': datetime.now(),
            'last_activity': datetime.now(),
            'message_count': 0
        }
        await self.send_personal_message({"type": "connection", "status": "connected"}, session_id)
    
    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        self.connection_metadata.pop(session_id, None)
    
    async def send_personal_message(self, message: dict, session_id: str):
        websocket = self.active_connections.get(session_id)
        if websocket:
            try:
                await websocket.send_text(orjson.dumps(message).decode())
                if session_id in self.connection_metadata:
                    self.connection_metadata[session_id]['last_activity'] = datetime.now()
                    self.connection_metadata[session_id]['message_count'] += 1
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
                self.disconnect(session_id)
    
    async def broadcast(self, message: dict):
        disconnected = []
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(orjson.dumps(message).decode())
            except Exception:
                disconnected.append(session_id)
        
        for session_id in disconnected:
            self.disconnect(session_id)

class InMemoryCache:
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.ttls: Dict[str, float] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        
        if key in self.ttls and time.time() > self.ttls[key]:
            self.delete(key)
            return None
            
        return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            self.delete(oldest_key)
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
        self.ttls[key] = time.time() + ttl
    
    def delete(self, key: str):
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
        self.ttls.pop(key, None)
    
    def clear(self):
        self.cache.clear()
        self.timestamps.clear()
        self.ttls.clear()

class SecurityManager:
    def __init__(self):
        key = Fernet.generate_key()
        self.fernet = Fernet(key)
        self.blocked_ips: Set[str] = set()
    
    def encrypt_data(self, data: str) -> str:
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def block_ip(self, ip: str):
        self.blocked_ips.add(ip)
    
    def is_blocked(self, ip: str) -> bool:
        return ip in self.blocked_ips

class RequestQueue:
    def __init__(self, max_size: int = 50):
        self.queues: Dict[Priority, deque] = {
            Priority.CRITICAL: deque(),
            Priority.HIGH: deque(),
            Priority.NORMAL: deque(),
            Priority.LOW: deque()
        }
        self.max_size = max_size
        self.processing: Set[str] = set()
    
    def enqueue(self, request_id: str, priority: Priority) -> bool:
        total_size = sum(len(q) for q in self.queues.values())
        if total_size >= self.max_size:
            return False
        
        self.queues[priority].append(request_id)
        return True
    
    def dequeue(self) -> Optional[str]:
        for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
            if self.queues[priority]:
                request_id = self.queues[priority].popleft()
                self.processing.add(request_id)
                return request_id
        return None
    
    def complete(self, request_id: str):
        self.processing.discard(request_id)
    
    def get_stats(self) -> Dict[str, int]:
        return {
            'queued': sum(len(q) for q in self.queues.values()),
            'processing': len(self.processing),
            'by_priority': {p.name: len(self.queues[p]) for p in Priority}
        }

http_client: Optional[httpx.AsyncClient] = None
executor: Optional[ThreadPoolExecutor] = None
cache_manager: InMemoryCache = None
security_manager: SecurityManager = None
connection_manager: ConnectionManager = None
request_queue: RequestQueue = None
model_keepalive_task: Optional[asyncio.Task] = None

async def keep_model_in_memory():
    while True:
        try:
            await asyncio.sleep(300)
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8080",
                "X-Title": "Stella AI Backend"
            }
            
            payload = {
                "model": get_active_model(),
                "messages": [
                    {"role": "user", "content": "ping"}
                ],
                "temperature": 0.1,
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
                logger.debug("Model keepalive ping successful")
            else:
                logger.warning(f"Model keepalive failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Model keepalive error: {e}")
            await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, executor, cache_manager, security_manager, connection_manager, request_queue, model_keepalive_task
    
    await init_database()

    
    cache_manager = InMemoryCache(max_size=500)
    security_manager = SecurityManager()
    connection_manager = ConnectionManager()
    request_queue = RequestQueue()
    
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(180.0, connect=30.0, read=120.0),
        limits=httpx.Limits(
            max_keepalive_connections=50,
            max_connections=200,
            keepalive_expiry=60
        ),
        http2=True,
        follow_redirects=True,
        headers={"User-Agent": "Stella-AI-Backend/3.0"}
    )
    
    executor = ThreadPoolExecutor(
        max_workers=min(8, (psutil.cpu_count() or 1) + 4),
        thread_name_prefix="stella-worker"
    )
    
    await warmup_model(http_client)
    
    model_keepalive_task = asyncio.create_task(keep_model_in_memory())
    asyncio.create_task(cleanup_inactive_sessions())
    asyncio.create_task(cache_cleanup_task())
    
    logger.info("Stella Backend v3.0 fully initialized")
    
    yield
    
    if model_keepalive_task:
        model_keepalive_task.cancel()
        
    if http_client:
        await http_client.aclose()
    if executor:
        executor.shutdown(wait=True)

app = FastAPI(
    title="Stella AI Backend",
    description="Advanced FastAPI backend for Stella AI with OpenRouter integration",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "crankyvase.site", "*.crankyvase.site"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "https://crankyvase.site", "http://crankyvase.site",
        "https://crankyvase.site:8080", "http://crankyvase.site:8080",
        "https://crankyvase.site:3000", "http://crankyvase.site:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"]
)

security = HTTPBearer(auto_error=False)

def get_client_info(request: Request) -> Dict[str, Any]:
    forwarded_for = request.headers.get("x-forwarded-for")
    client_ip = forwarded_for.split(',')[0].strip() if forwarded_for else getattr(request.client, 'host', 'unknown')
    
    return {
        'ip': client_ip,
        'user_agent': request.headers.get('user-agent', 'unknown'),
        'accept_language': request.headers.get('accept-language', 'unknown'),
        'forwarded_for': forwarded_for,
        'real_ip': request.headers.get('x-real-ip'),
        'host': request.headers.get('host'),
        'referer': request.headers.get('referer'),
        'timestamp': datetime.now().isoformat()
    }

async def verify_browser_auth(request: Request) -> str:
    client_info = get_client_info(request)
    
    if security_manager.is_blocked(client_info['ip']):
        raise HTTPException(status_code=403, detail="IP address is blocked")
    
    auth_header = request.headers.get("Authorization", "")
    bearer_token = None
    if auth_header.startswith("Bearer "):
        bearer_token = auth_header[7:] 
    
    browser_token = (
        request.cookies.get("browser_token") or 
        request.headers.get("X-Browser-Token") or
        bearer_token
    )
    
    if not browser_token:
        raise HTTPException(status_code=401, detail="Browser authentication required")
    
    browser_fingerprint = generate_browser_fingerprint(dict(request.headers), client_info['ip'])
    
    if not await verify_browser_token(browser_token, browser_fingerprint):

        raise HTTPException(status_code=401, detail="Invalid browser token")
    
    return browser_token

async def get_session_info(request: Request) -> Dict[str, Any]:
    browser_token = await verify_browser_auth(request)  
    session_id = request.cookies.get("session_id") or request.headers.get("X-Session-ID")  
    
    session_valid = False
    if session_id:
        try:
            session_valid = check_session(session_id)
        except (KeyError, AttributeError, TypeError):
            session_valid = session_id in active_sessions
    
    if not session_id or not session_valid:
        raise HTTPException(status_code=401, detail="Valid session required")
    
    return {
        'session_id': session_id,
        'browser_token': browser_token,
        'client_info': get_client_info(request)
    }

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Server-Version"] = "3.0.0"
    return response
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://static.cloudflareinsights.com; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self' https://api.openrouter.ai; "
        "font-src 'self' data:;"
    )
    return response


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = Path("stella.html")
    if not frontend_path.exists():
        return HTMLResponse(
            content="""
            <html>
                <head>
                    <title>Stella AI v3.0</title>
                    <style>
                        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                               color: white; text-align: center; padding: 50px; min-height: 100vh; margin: 0; }
                        .container { max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.1); 
                                   padding: 40px; border-radius: 20px; backdrop-filter: blur(10px); }
                        h1 { font-size: 3em; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
                        .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }
                        .feature { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; }
                        a { color: #fff; text-decoration: none; background: rgba(255,255,255,0.2); 
                            padding: 10px 20px; border-radius: 5px; display: inline-block; margin: 10px; transition: all 0.3s; }
                        a:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Stella AI v3.0</h1>
                        <p>Enhanced AI Backend - OpenRouter Integration</p>
                        <div class="features">
                            <div class="feature">
                                <h3>Multiple AI Models</h3>
                                <p>Choose from multiple advanced AI models</p>
                            </div>
                            <div class="feature">
                                <h3>Image Generation</h3>
                                <p>AI can generate images using local models</p>
                            </div>
                            <div class="feature">
                                <h3>High Performance</h3>
                                <p>Memory caching, connection pooling, streaming</p>
                            </div>
                        </div>
                        <p style="color: #ffcccb; margin-top: 30px;">Frontend file (stella.html) not found</p>
                        <div>
                            <a href="/api/health">System Health</a>
                            <a href="/api/docs">API Documentation</a>
                        </div>
                    </div>
                </body>
            </html>
            """,
            status_code=200
        )
    
    try:
        with open(frontend_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f"Error loading frontend: {e}")
        raise HTTPException(status_code=500, detail="Frontend loading error")

static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/api/health")
async def comprehensive_health_check():
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "storage_mode": "in-memory",
        "model_provider": "OpenRouter",
        "current_model": get_active_model()
    }
    
    health_data["image_generation"] = PHOTO_AVAILABLE
  
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        response = await http_client.get(f"{OPENROUTER_BASE_URL}/models", headers=headers, timeout=10.0)
        health_data["openrouter_status"] = "healthy" if response.status_code == 200 else "unhealthy"
        if response.status_code == 200:
            models_data = response.json()
            health_data["available_models"] = len(models_data.get("data", []))
    except Exception as e:
        health_data["openrouter_status"] = "unhealthy"
        health_data["openrouter_error"] = str(e)
    
    health_data.update({
        "active_sessions": len(active_sessions),
        "active_requests": len(active_requests),
        "websocket_connections": len(connection_manager.active_connections),
        "queue_stats": request_queue.get_stats(),
        "cache_size": len(cache_manager.cache),
        "model": get_active_model(),
        "frontend_available": Path("stella.html").exists(),
        "system_resources": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent
        },
        "features": [
            "openrouter_integration", "websocket_support", 
            "request_queuing", "compression", "connection_pooling", "in_memory_caching",
            "image_generation", "multi_model_support", "session_management", "advanced_code_tracking"
        ]
    })
    
    return health_data

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    if not check_session(session_id):
        await websocket.close(code=4001, reason="Invalid session")
        return
    
    await connection_manager.connect(websocket, session_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = WebSocketMessage(**orjson.loads(data))
                await handle_websocket_message(message, session_id)
            except Exception as e:
                await connection_manager.send_personal_message(
                    {"type": "error", "message": f"Invalid message format: {str(e)}"},
                    session_id
                )
    except WebSocketDisconnect:
        connection_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(session_id)

async def handle_websocket_message(message: WebSocketMessage, session_id: str):
    if message.type == "ping":
        await connection_manager.send_personal_message({"type": "pong"}, session_id)
    elif message.type == "status_request":
        stats = {
            "type": "status_response",
            "queue_stats": request_queue.get_stats(),
            "session_info": active_sessions.get(session_id, {}),
            "connection_info": connection_manager.connection_metadata.get(session_id, {}),
            "current_model": get_active_model()
        }
        await connection_manager.send_personal_message(stats, session_id)

async def generate_image(prompt: str, session_id: str = None) -> Tuple[bool, str]:
    try:
        if not PHOTO_AVAILABLE:
            return False, "Image generation not available"
        
        from stphoto import PhotoGenerator
        photo_gen = PhotoGenerator()
        
        if not photo_gen.init_image_pipeline():
            return False, "Failed to initialize image generation"
        
        return await photo_gen.generate_image(prompt, session_id)
        
    except Exception as e:
        print(f"Image generation error: {e}")
        return False, f"Image generation failed: {str(e)}"

@app.post("/api/browser-auth")
async def browser_auth(request: Request, auth_data: BrowserAuthRequest):
    client_info = get_client_info(request)
    
    if security_manager.is_blocked(client_info['ip']):
        raise HTTPException(status_code=403, detail="IP address is blocked")
    
    fingerprint_data = auth_data.fingerprint_data or {}
    enhanced_fingerprint = generate_browser_fingerprint(
        dict(request.headers), 
        client_info['ip']
    )
    
    existing_token = auth_data.browser_token
    if existing_token and await verify_browser_token(existing_token, enhanced_fingerprint):
        return {
            "success": True,
            "browser_token": existing_token,
            "new_token": False,
            "expires_in_days": 30,
            "security_level": "verified",
            "message": "Existing browser authentication verified"
        }
    
    new_token = await create_browser_token(enhanced_fingerprint)

    
    response_data = {
        "success": True,
        "browser_token": new_token,
        "new_token": True,
        "expires_in_days": 30,
        "security_level": "new",
        "client_fingerprint": hashlib.sha256(enhanced_fingerprint.encode()).hexdigest()[:16],
        "message": "New browser authentication created"
    }
    
    response = JSONResponse(response_data)
    response.set_cookie(
        key="browser_token",
        value=new_token,
        max_age=30*24*3600,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict"
    )
    
    return response

@app.post("/api/login")
async def login(request: Request, login_data: LoginRequest):
    client_info = get_client_info(request)
    
    if security_manager.is_blocked(client_info['ip']):
        raise HTTPException(status_code=403, detail="IP address is blocked")
    
    browser_token = await verify_browser_auth(request)
    
    if not verify_password(login_data.password, ADMIN_PASSWORD_HASH):
        await asyncio.sleep(2)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    session_id = str(uuid.uuid4())
    create_session(session_id)
    
    if session_id not in active_sessions:
        active_sessions[session_id] = {
            'session_id': session_id,
            'login_time': time.time(),
            'last_activity': time.time(),
            'request_count': 0,
            'authenticated': True
        }
    
    active_sessions[session_id].update({
        'client_info': client_info,
        'device_info': login_data.device_info,
        'remember_me': login_data.remember_me,
        'login_timestamp': datetime.now().isoformat()
    })
    
    response_data = {
        "success": True,
        "session_id": session_id,
        "browser_token": browser_token,
        "expires_in": SESSION_TIMEOUT*2 if login_data.remember_me else SESSION_TIMEOUT,
        "remember_me": login_data.remember_me,
        "timestamp": datetime.now().isoformat()
    }
    
    response = JSONResponse(response_data)
    response.set_cookie(
        key="session_id",
        value=session_id,
        max_age=SESSION_TIMEOUT*2 if login_data.remember_me else SESSION_TIMEOUT,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict"
    )
    
    return response

@app.get("/api/models")
async def get_available_models(session_info: Dict = Depends(get_session_info)):
    return {
        "models": AVAILABLE_MODELS,
        "current_model": get_active_model(),
        "default_model": DEFAULT_MODEL
    }

@app.post("/api/models/select")
async def select_model(request: Request, session_info: Dict = Depends(get_session_info)):
    try:
        data = await request.json()
        model_id = data.get('model_id')
        
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id required")
        
        success = set_active_model(model_id)
        
        if success:
            return {
                "success": True,
                "model_id": model_id,
                "message": f"Model switched to {model_id}"
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid model ID")
            
    except Exception as e:
        logger.error(f"Model selection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_image_with_progress(request_id: str, session_id: str, prompt: str) -> AsyncGenerator[str, None]:
    try:
        active_requests[session_id] = {
            'request_id': request_id,
            'started_at': time.time(),
            'stopped': False,
            'tokens_generated': 0
        }
        
        yield f"data: {orjson.dumps({'status': 'initializing', 'request_id': request_id}).decode()}\n\n"
        
        import threading
        import queue
        
        # Use a queue for thread-safe communication
        progress_queue = queue.Queue()
        # Send initial status
        yield f"data: {orjson.dumps({'status': 'Initializing image generation...'}).decode()}\n\n"

        progress_state = {
            "current": 0, 
            "total": IMAGE_STEPS,
            "percentage": 0, 
            "eta": "calculating...", 
            "last_update": time.time(),
        }
        generation_complete = threading.Event()
        generation_result = {"success": False, "result": "", "error": None}
        
        def sync_progress_callback(current, total, percentage, eta):
    # Put progress updates into the queue
            progress_data = {
            "current": current,
            "total": total,
            "percentage": percentage,
            "eta": eta,
            "timestamp": time.time()
    }
            print(f"Backend sending progress: {percentage}% ({current}/{total}) ETA: {eta}")
            try:
                progress_queue.put(progress_data, block=False)
            except queue.Full:
                print("Progress queue full, skipping update")


        
        def run_generation():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                print(f"[backend.py] Setting progress callback in thread")  # <-- ADD THIS LINE
                photo_generator.set_progress_callback(sync_progress_callback)
                print(f"[backend.py] Callback set, starting generation")  # <-- ADD THIS LINE
                success, result = loop.run_until_complete(
                    photo_generator.generate_image(prompt, session_id)
                )
                generation_result["success"] = success
                generation_result["result"] = result
            except Exception as e:
                print(f"Generation thread error: {e}")
                import traceback
                traceback.print_exc()
                generation_result["error"] = str(e)
            finally:
                generation_complete.set()
                loop.close()
        
        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()
        
        last_sent_percentage = -1
        last_update_time = time.time()
        
        while not generation_complete.is_set():
            request_info = active_requests.get(session_id, {})
            if request_info.get('stopped', False):
                photo_generator.stop()
                yield f"data: {orjson.dumps({'status': 'stopped', 'request_id': request_id}).decode()}\n\n"
                break
            
            try:
                while True:
                    update = progress_queue.get_nowait()
                    progress_state.update(update)
            except queue.Empty:
                pass
            
            current_percentage = progress_state['percentage']
            current_time = time.time()
            
            # Send update if percentage changed OR every 2 seconds (for long steps)
            if (current_percentage != last_sent_percentage) or (current_time - last_update_time >= 25.0):

                yield f"data: {orjson.dumps({
                    'image_progress': {
                        'current': progress_state['current'],
                        'total': progress_state['total'],
                        'percentage': current_percentage,
                        'eta': progress_state['eta']
                    }
                }).decode()}\n\n"
                last_sent_percentage = current_percentage
                last_update_time = current_time
            
            await asyncio.sleep(0.1)

        
        gen_thread.join(timeout=10)
        
        # Send final 100% update
        yield f"data: {orjson.dumps({
            'image_progress': {
                'current': progress_state["total"],
                'total': progress_state["total"],
                'percentage': 100,
                'eta': '0s'
            }
        }).decode()}\n\n"
        
        await asyncio.sleep(0.1)
        
        if generation_result.get("error"):
            error_response = f"Sorry, there was an error: {generation_result['error']}"
            yield f"data: {orjson.dumps({
                'response': error_response,
                'request_id': request_id,
                'image_generated': False,
                'error': generation_result['error'],
                'done': True
            }).decode()}\n\n"
        elif generation_result["success"]:
            image_path = generation_result['result']
            filename = os.path.basename(image_path) if os.path.isabs(image_path) else image_path
            
            image_response = f"I've created an image for you based on: '{prompt}'"
            yield f"data: {orjson.dumps({
                'response': image_response,
                'request_id': request_id,
                'image_generated': True,
                'image_path': filename,
                'prompt': prompt,
                'done': True
            }).decode()}\n\n"
        else:
            error_response = f"Sorry, I couldn't generate the image. {generation_result['result']}"
            yield f"data: {orjson.dumps({
                'response': error_response,
                'request_id': request_id,
                'image_generated': False,
                'error': generation_result['result'],
                'done': True
            }).decode()}\n\n"
    
    except Exception as e:
        print(f"Image generation streaming error: {e}")
        import traceback
        traceback.print_exc()
        yield f"data: {orjson.dumps({
            'error': str(e),
            'request_id': request_id,
            'done': True
        }).decode()}\n\n"
    
    finally:
        request_queue.complete(request_id)
        if session_id in active_requests:
            active_requests.pop(session_id, None)


@app.post("/api/chat")
async def enhanced_chat(request: Request, chat_data: EnhancedChatMessage, session_info: Dict = Depends(get_session_info)):
    session_id = session_info['session_id']
    client_info = session_info['client_info']
    
    if session_id in active_requests and not active_requests[session_id].get('stopped', True):
        raise HTTPException(status_code=429, detail="Another request is already in progress")
    
    request_id = str(uuid.uuid4())
    
    if not request_queue.enqueue(request_id, chat_data.priority):
        raise HTTPException(status_code=503, detail="Server is at capacity. Please try again later.")
    
    if session_id in active_sessions:
        active_sessions[session_id]['request_count'] += 1
        active_sessions[session_id]['last_activity'] = time.time()
    
    print(f"Processing chat request: '{chat_data.message}' (session: {session_id})")
    
    if detect_image_request(chat_data.message):
        print("Image request detected - using streaming!")
        prompt = extract_prompt_from_message(chat_data.message)
        print(f"Extracted prompt: '{prompt}'")
        
        return StreamingResponse(
            generate_image_with_progress(request_id, session_id, prompt),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
                "Access-Control-Expose-Headers": "X-Request-ID"
            }
        )
    
    cache_key = f"conversation:{session_id}:{hashlib.md5(chat_data.message.encode()).hexdigest()[:8]}"
    cached_response = cache_manager.get(cache_key)
    
    if cached_response and not chat_data.streaming:
        request_queue.complete(request_id)
        return JSONResponse(cached_response)
    
    model_options = MODEL_OPTIONS.copy()
    if chat_data.temperature is not None:
        model_options['temperature'] = chat_data.temperature
    if chat_data.max_tokens is not None:
        model_options['max_tokens'] = chat_data.max_tokens
    
    enhanced_prompt = await build_conversation_prompt(
        chat_data.message,
        chat_data.history,
        chat_data.files,
        session_id
    )
    
    if chat_data.streaming:
        return StreamingResponse(
            generate_streaming_response(request_id, session_id, chat_data, enhanced_prompt, model_options),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
                "Access-Control-Expose-Headers": "X-Request-ID"
            }
        )
    else:
        return await generate_non_streaming_response(request_id, session_id, chat_data, enhanced_prompt, model_options, cache_key)

async def generate_streaming_response(
    request_id: str,
    session_id: str,
    chat_data: EnhancedChatMessage, 
    prompt: str, 
    model_options: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    
    accumulated_response = ""
    code_tracker = CodeBlockTracker()
    
    try:
        active_requests[session_id] = {
            'request_id': request_id,
            'started_at': time.time(),
            'stopped': False,
            'priority': chat_data.priority,
            'tokens_generated': 0
        }
        
        logger.info(f"Starting streaming request {request_id}")
        
        if session_id in connection_manager.active_connections:
            await connection_manager.send_personal_message({
                "type": "chat_start",
                "request_id": request_id,
                "priority": chat_data.priority.name
            }, session_id)
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8080",
            "X-Title": "Stella AI Backend"
        }
        
        messages = []
        if chat_data.history:
            for msg in chat_data.history[-10:]:
                if msg.get('role') in ['user', 'assistant']:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
        
        messages.append({
            "role": "user", 
            "content": prompt
        })
        
        payload = {
            "model": get_active_model(),
            "messages": messages,
            "temperature": model_options.get('temperature', 0.7),
            "max_tokens": model_options.get('max_tokens', 4000),
            "top_p": model_options.get('top_p', 0.85),
            "frequency_penalty": model_options.get('frequency_penalty', 0.1),
            "presence_penalty": model_options.get('presence_penalty', 0.1),
            "stream": True
        }
        
        async with http_client.stream(
            'POST',
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=300
        ) as response:
            
            if response.status_code != 200:
                error_msg = f"OpenRouter API error (HTTP {response.status_code})"
                logger.error(f"Error: {error_msg}")
                yield f"data: {orjson.dumps({'error': error_msg, 'request_id': request_id}).decode()}\n\n"
                return
            
            buffer = ""
            total_tokens = 0
            start_time = time.time()
            last_heartbeat = start_time
            token_times = deque(maxlen=50)
            
            yield f"data: {orjson.dumps({'status': 'connected', 'request_id': request_id, 'model': get_active_model()}).decode()}\n\n"
            
            async for chunk in response.aiter_bytes(1024):
                current_time = time.time()
                
                if active_requests.get(session_id, {}).get('stopped', True):
                    logger.info(f"Request {request_id} stopped by user")
                    yield f"data: {orjson.dumps({'status': 'stopped', 'message': 'Request stopped by user', 'request_id': request_id}).decode()}\n\n"
                    break
                
                if current_time - last_heartbeat > 10:
                    yield f"data: {orjson.dumps({'type': 'heartbeat', 'timestamp': current_time}).decode()}\n\n"
                    last_heartbeat = current_time
                
                try:
                    buffer += chunk.decode('utf-8', errors='ignore')
                    lines = buffer.split('\n')
                    buffer = lines[-1]
                    
                    for line in lines[:-1]:
                        line = line.strip()
                        if not line or not line.startswith('data: '):
                            continue
                        
                        if line == 'data: [DONE]':
                            final_result = code_tracker.finalize()
                            elapsed = current_time - start_time
                            avg_tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                            
                            logger.info(f"Request {request_id} completed: {total_tokens} tokens in {elapsed:.2f}s ({avg_tokens_per_sec:.1f} tok/s)")
                            
                            completion_data = {
                                'done': True,
                                'request_id': request_id,
                                'stats': {
                                    'tokens': total_tokens,
                                    'elapsed': elapsed,
                                    'tokens_per_second': avg_tokens_per_sec,
                                    'priority': chat_data.priority.name
                                },
                                'final_response': accumulated_response,
                                'code_blocks': final_result.get('code_blocks', []),
                                'has_code': final_result.get('has_code', False)
                            }
                            
                            cache_manager.set(f"response:{request_id}", {
                                'response': accumulated_response,
                                'stats': completion_data['stats'],
                                'timestamp': datetime.now().isoformat()
                            }, ttl=3600)
                            
                            if session_id in connection_manager.active_connections:
                                await connection_manager.send_personal_message({
                                    "type": "chat_complete",
                                    "request_id": request_id,
                                    "stats": completion_data['stats']
                                }, session_id)
                            
                            yield f"data: {orjson.dumps(completion_data).decode()}\n\n"
                            return
                        
                        try:
                            json_str = line[6:]
                            data = orjson.loads(json_str)
                            
                            if 'choices' in data and len(data['choices']) > 0:
                                choice = data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    token = choice['delta']['content']
                                    if token:
                                        total_tokens += 1
                                        accumulated_response += token
                                        token_times.append(current_time)
                                        
                                        active_requests[session_id]['tokens_generated'] = total_tokens
                                        
                                        process_result = code_tracker.process_token(token)
                                        
                                        response_data = {
                                            'token': token,
                                            'text_token': process_result.get('text', ''),
                                            'done': False,
                                            'request_id': request_id,
                                            'tokens_generated': total_tokens,
                                            'elapsed': current_time - start_time,
                                            'in_code_block': process_result.get('in_code', False),
                                            'accumulated_response': accumulated_response
                                        }
                                        
                                        if process_result.get('code_block_started'):
                                            response_data['code_block_started'] = True
                                            response_data['code_block_id'] = process_result.get('code_block_id')
                                            response_data['code_language'] = process_result.get('code_language')
                                            logger.info(f"CODE BLOCK STARTED: {response_data['code_block_id']} ({response_data['code_language']})")
                                        
                                        if process_result.get('streaming_code'):
                                            response_data['streaming_code'] = process_result['streaming_code']
                                        
                                        if process_result.get('code_block'):
                                            response_data['code_block_complete'] = process_result['code_block']
                                            logger.info(f"CODE BLOCK COMPLETE: {process_result['code_block'].get('id')}")
                                        
                                        if len(token_times) >= 10:
                                            recent_tokens_per_sec = len(token_times) / (token_times[-1] - token_times[0]) if len(token_times) > 1 else 0
                                            response_data['tokens_per_second'] = round(recent_tokens_per_sec, 2)
                                        
                                        yield f"data: {orjson.dumps(response_data).decode()}\n\n"
                        
                        except orjson.JSONDecodeError:
                            continue
                
                except UnicodeDecodeError:
                    continue
    
    except httpx.ConnectError:
        error_msg = "Cannot connect to OpenRouter API. Service may be down."
        logger.error(f"Connection error: {error_msg}")
        yield f"data: {orjson.dumps({'error': error_msg, 'request_id': request_id}).decode()}\n\n"
    
    except httpx.TimeoutException:
        error_msg = "Request timed out. Please try again with a shorter message."
        logger.error(f"Timeout error for request {request_id}")
        yield f"data: {orjson.dumps({'error': error_msg, 'request_id': request_id}).decode()}\n\n"
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error in request {request_id}: {e}")
        yield f"data: {orjson.dumps({'error': error_msg, 'request_id': request_id}).decode()}\n\n"
    
    finally:
        request_queue.complete(request_id)
        if session_id in active_requests:
            active_requests.pop(session_id, None)
        logger.info(f"Cleaned up request {request_id}")

async def generate_non_streaming_response(
    request_id: str,
    session_id: str,
    chat_data: EnhancedChatMessage,
    prompt: str,
    model_options: Dict[str, Any],
    cache_key: str
) -> JSONResponse:
    
    try:
        start_time = time.time()
        
        if detect_image_request(chat_data.message):
            print("Image request in non-streaming mode")
            image_prompt = extract_prompt_from_message(chat_data.message)
            success, result = await generate_image(image_prompt, session_id)
            
            if success:
                response_text = f"I've created an image for you based on: '{image_prompt}'\n\nImage saved to: {result}"
                
                return JSONResponse({
                    'response': response_text,
                    'request_id': request_id,
                    'image_generated': True,
                    'image_path': result,
                    'prompt': image_prompt,
                    'done': True
                })
            else:
                error_text = f"Sorry, I couldn't generate the image. {result}"
                
                return JSONResponse({
                    'response': error_text,
                    'request_id': request_id,
                    'image_generated': False,
                    'error': result,
                    'done': True
                })
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8080",
            "X-Title": "Stella AI Backend"
        }
        
        messages = []
        if chat_data.history:
            for msg in chat_data.history[-10:]:
                if msg.get('role') in ['user', 'assistant']:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
        
        messages.append({
            "role": "user", 
            "content": prompt
        })
        
        payload = {
            "model": get_active_model(),
            "messages": messages,
            "temperature": model_options.get('temperature', 0.7),
            "max_tokens": model_options.get('max_tokens', 4000),
            "top_p": model_options.get('top_p', 0.85),
            "frequency_penalty": model_options.get('frequency_penalty', 0.1),
            "presence_penalty": model_options.get('presence_penalty', 0.1),
            "stream": False
        }
        
        response = await http_client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=120
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="OpenRouter API error")
        
        data = response.json()
        elapsed = time.time() - start_time
        response_content = data['choices'][0]['message']['content']
        
        result = {
            'response': response_content,
            'request_id': request_id,
            'stats': {
                'elapsed': elapsed,
                'total_tokens': data.get('usage', {}).get('total_tokens', 0),
                'prompt_tokens': data.get('usage', {}).get('prompt_tokens', 0),
                'completion_tokens': data.get('usage', {}).get('completion_tokens', 0)
            },
            'model': get_active_model(),
            'done': True
        }
        
        cache_manager.set(cache_key, result, ttl=1800)
        request_queue.complete(request_id)
        
        return JSONResponse(result)
        
    except Exception as e:
        request_queue.complete(request_id)
        logger.error(f"Non-streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/logout")
async def logout(request: Request, session_info: Dict = Depends(get_session_info)):
    session_id = session_info['session_id']
    
    if session_id in active_sessions:
        session_data = active_sessions[session_id]
        logger.info(f"User logout: {session_data.get('login_time', 'unknown time')}")
    
    active_sessions.pop(session_id, None)
    
    if session_id in active_requests:
        active_requests[session_id]['stopped'] = True
        active_requests.pop(session_id, None)
    
    connection_manager.disconnect(session_id)
    
    response = JSONResponse({
        "success": True,
        "message": "Logged out successfully",
        "timestamp": datetime.now().isoformat()
    })
    
    response.delete_cookie("session_id")
    
    return response

@app.post("/api/upload")
async def upload_file(request: Request, session_info: Dict = Depends(get_session_info)):
    """Upload a file and extract text content for AI processing"""
    try:
        # Check if upload was made
        form = await request.form()
        if 'file' not in form:
            raise HTTPException(status_code=400, detail="No file provided")

        file = form['file']
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")

        # Validate file type
        allowed_types = ['pdf', 'txt', 'text', 'md', 'markdown']
        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''

        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type '{file_ext}' not allowed. Allowed types: {', '.join(allowed_types)}"
            )

        # Validate file size (10MB limit)
        file_size = len(await file.read())
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size: 10MB")

        # Reset file pointer and read content
        await file.seek(0)
        file_data = await file.read()

        # Save file and extract text
        from stbrain import save_uploaded_file
        success, file_id, text_content = await save_uploaded_file(file_data, file.filename, session_info['session_id'])

        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {text_content}")

        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "file_type": file_ext,
            "file_size": file_size,
            "has_content": bool(text_content and text_content.strip()),
            "message": f"File '{file.filename}' uploaded successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/files")
async def get_uploaded_files(session_info: Dict = Depends(get_session_info)):
    """Get list of uploaded files for the current session"""
    try:
        from stbrain import get_uploaded_files
        files = await get_uploaded_files(session_info['session_id'])

        return {
            "success": True,
            "files": files
        }

    except Exception as e:
        logger.error(f"Error getting files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get files: {str(e)}")

@app.get("/api/files/{file_id}/content")
async def get_file_content(file_id: str, session_info: Dict = Depends(get_session_info)):
    """Get text content of an uploaded file"""
    try:
        from stbrain import get_file_content
        content = await get_file_content(file_id, session_info['session_id'])

        if content is None:
            raise HTTPException(status_code=404, detail="File not found")

        return {
            "success": True,
            "file_id": file_id,
            "content": content
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file content: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get file content: {str(e)}")

@app.get("/api/session-check")
async def session_check(request: Request, browser_token: str = Depends(verify_browser_auth)):
    session_id = request.cookies.get("session_id") or request.headers.get("X-Session-ID")
    is_session_valid = session_id and check_session(session_id)
    
    session_info = None
    if is_session_valid and session_id in active_sessions:
        session = active_sessions[session_id]
        session_info = {
            'login_time': session.get('login_time'),
            'last_activity': session.get('last_activity'),
            'request_count': session.get('request_count', 0),
            'device_info': session.get('device_info'),
            'remember_me': session.get('remember_me', False)
        }
    
    return {
        "authenticated": is_session_valid,
        "browser_authenticated": True,
        "browser_token": browser_token,
        "session_id": session_id if is_session_valid else None,
        "session_info": session_info,
        "requires_login": not is_session_valid,
        "security_status": {
            "ip_blocked": security_manager.is_blocked(get_client_info(request)['ip'])
        },
        "server_info": {
            "version": "3.0.0",
            "storage_mode": "in-memory",
            "model_provider": "OpenRouter",
            "current_model": get_active_model(),
            "features_enabled": ["websockets", "compression", "openrouter_integration", "image_generation", "multi_model_support", "session_management", "advanced_code_tracking"]
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stats")
async def get_system_stats(session_info: Dict = Depends(get_session_info)):
    return {
        "system": {
            "active_sessions": len(active_sessions),
            "active_requests": len(active_requests),
            "websocket_connections": len(connection_manager.active_connections),
            "queue_stats": request_queue.get_stats(),
            "cache_size": len(cache_manager.cache),
            "storage_mode": "in-memory",
            "model_provider": "OpenRouter",
            "current_model": get_active_model()
        },
        "performance": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent
        },
        "security": {
            "blocked_ips": len(security_manager.blocked_ips),
            "auth_mode": "browser_token"
        }
    }

@app.get("/generated_images/{filename}")
async def serve_image(filename: str, session_info: Dict = Depends(get_session_info)):
    image_path = Path("generated_images") / filename
    if image_path.exists() and image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
        return FileResponse(
            image_path, 
            media_type=f"image/{image_path.suffix[1:]}", 
            headers={"Cache-Control": "public, max-age=3600"}
        )
    raise HTTPException(status_code=404, detail="Image not found")

@app.get("/favicon.ico")
async def get_favicon():
    favicon_path = Path("favicon.png")
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/x-icon")
    raise HTTPException(status_code=404, detail="Favicon not found")

async def cache_cleanup_task():
    while True:
        try:
            await asyncio.sleep(1800)
            current_time = time.time()
            
            expired_keys = []
            for key, ttl in cache_manager.ttls.items():
                if current_time > ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                cache_manager.delete(key)
            
            if len(cache_manager.cache) > 300:
                oldest_keys = sorted(cache_manager.timestamps.keys(), 
                                   key=lambda k: cache_manager.timestamps[k])[:50]
                for key in oldest_keys:
                    cache_manager.delete(key)
            
            logger.info(f"Cache cleanup: removed {len(expired_keys)} expired items")
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            await asyncio.sleep(300)

async def cleanup_inactive_sessions():
    while True:
        try:
            await asyncio.sleep(300)
            current_time = time.time()
            
            expired_sessions = []
            for session_id, session in list(active_sessions.items()):
                if current_time - session.get('last_activity', session['login_time']) > SESSION_TIMEOUT:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                active_sessions.pop(session_id, None)
                active_requests.pop(session_id, None)
                connection_manager.disconnect(session_id)
            
            stale_requests = []
            for session_id, request in list(active_requests.items()):
                if current_time - request.get('started_at', current_time) > 900:
                    stale_requests.append(session_id)
            
            for session_id in stale_requests:
                active_requests.pop(session_id, None)
            
            cleanup_expired_tokens()
            
            fifteen_minutes_ago = int(current_time) - 900
            try:
                from stbrain import db_lock, DB_PATH
                import sqlite3
                import os
                
                with db_lock:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT filename FROM generated_images 
                        WHERE created_at < ?
                    ''', (fifteen_minutes_ago,))
                    
                    old_images = cursor.fetchall()
                    
                    for (filename,) in old_images:
                        image_path = Path("generated_images") / filename
                        try:
                            if image_path.exists():
                                image_path.unlink()
                                logger.info(f"Auto-deleted old image: {filename}")
                        except Exception as e:
                            logger.error(f"Error deleting image {filename}: {e}")
                    
                    cursor.execute('DELETE FROM generated_images WHERE created_at < ?', (fifteen_minutes_ago,))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    conn.close()
                    
                    if deleted_count > 0:
                        logger.info(f"Auto-cleanup: removed {deleted_count} old images")
                        
            except Exception as e:
                logger.error(f"Image auto-cleanup error: {e}")
            
            if expired_sessions or stale_requests:
                logger.info(f"Cleanup: {len(expired_sessions)} expired sessions, {len(stale_requests)} stale requests")
                
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")

if __name__ == '__main__':
    try:
        frontend_exists = Path("stella.html").exists()
        logger.info(f"Frontend file: {'found' if frontend_exists else 'missing'}")
        
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8080,
            workers=1,
            reload=False,
            access_log=False,
            server_header=False,
            date_header=False,
            log_level="info",
            loop="asyncio",
            lifespan="on",
            ws_ping_interval=20,
            ws_ping_timeout=20,
            timeout_keep_alive=30
        )
        
        server = uvicorn.Server(config)
        logger.info("Starting Stella AI Backend v3.0 - OpenRouter Integration with Advanced Code Tracking")
        server.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server startup error: {e}")
    finally:
        logger.info("Server shutdown complete")
 