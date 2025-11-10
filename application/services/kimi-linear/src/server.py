"""
Kimi Linear 48B API Server - SPARSE ACTIVATION VERSION
- 48B Total Parameters, 3B Active per Forward Pass
- Requires fla-core for A3B Attention Architecture
- Trust_remote_code=True is mandatory for fla-core!
"""

import os
import logging
from typing import List, Optional, Dict, Any
import time
import psutil
import json
import asyncpg
import asyncio
import signal
import gc  # NEU: Für garbage collection

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# NEU: Für Concurrency Control und GPU Serialization
from asyncio import Semaphore, Lock

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kimi-linear-48b-sparse")

app = FastAPI(
    title="Kimi Linear 48B API (A3B Sparse)",
    description="48B Parameters, 3B Active per Forward Pass",
    version="1.0.0"
)

# NEU - Concurrency Control
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
request_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

# NEU - GPU Serialization Lock
_generation_lock = Lock()

# NEU - Signal Handler (nach app Definition)
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Exit
    import sys
    sys.exit(0)

# Register handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# NEU - Cleanup Funktion
def cleanup_model():
    """Cleanup model and free GPU memory"""
    global model, tokenizer
    
    if model is not None:
        try:
            # Clear model from memory
            del model
            model = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            logger.info("Model cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
    
    if tokenizer is not None:
        del tokenizer
        tokenizer = None

# Pydantic Models
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    store_metadata: bool = True  # Store extended metadata

class GenerateResponse(BaseModel):
    text: str
    tokens_used: int
    model: str
    generation_time_ms: float
    memory_id: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_info: Dict[str, Any]

# Global state
model: Optional[torch.nn.Module] = None
tokenizer: Optional[AutoTokenizer] = None
model_config: Dict[str, Any] = {}

# NEU - Memory Monitoring
def check_memory_available(required_gb: float = 2.0) -> bool:
    """Check if enough memory available"""
    if torch.cuda.is_available():
        free_memory = torch.cuda.mem_get_info()[0] / (1024**3)
        return free_memory >= required_gb
    return True

def load_model():
    """Load Kimi Linear 48B with Sparse Activation (A3B)"""
    global model, tokenizer, model_config
    
    MODEL_NAME = os.getenv("MODEL_NAME", "moonshotai/Kimi-Linear-48B-A3B-Instruct")
    MODEL_PATH = os.getenv("MODEL_PATH", "/models/kimi-linear-48b")
    LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
    TORCH_COMPILE = os.getenv("TORCH_COMPILE", "true").lower() == "true"
    TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
    
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_NAME
    
    logger.info(f"════════════════════════════════════════════════════════════")
    logger.info(f"  Loading Kimi Linear 48B (A3B - Sparse Activation)")
    logger.info(f"  Total Params: 48B | Active Params: 3B per Forward Pass")
    logger.info(f"  Model Path: {model_path}")
    logger.info(f"════════════════════════════════════════════════════════════")
    
    # Load Tokenizer (trust_remote_code is mandatory!)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading configuration
    load_kwargs = {
        "trust_remote_code": True,  # CRITICAL for fla-core A3B!
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # 4-bit quantization: reduces 3B active → ~1.5GB VRAM!
    if LOAD_IN_4BIT:
        logger.info("Enabling 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    
    # Load model (this can take 5-10 minutes)
    logger.info("Loading model (this may take 5-10 minutes)...")
    start_load = time.time()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        
        # torch.compile optimization for sparse activation
        if TORCH_COMPILE:
            logger.info("Compiling model with torch.compile (sparse)...")
            model = torch.compile(model, mode="reduce-overhead")
        
        load_time = time.time() - start_load
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        active_params = 3_000_000_000  # As per documentation
        
        model_config = {
            "model_name": MODEL_NAME,
            "model_path": model_path,
            "load_in_4bit": LOAD_IN_4BIT,
            "torch_compile": TORCH_COMPILE,
            "device": str(model.device),
            "load_time_seconds": round(load_time, 2),
            "total_params_b": round(total_params / 1e9, 1),
            "active_params_b": round(active_params / 1e9, 1),
            "architecture": "A3B - Sparse Activation"
        }
        
        logger.info(f"✅ Model loaded in {load_time:.2f}s")
        logger.info(f"📊 Total Params: {model_config['total_params_b']}B")
        logger.info(f"⚡ Active Params: {model_config['active_params_b']}B")
        logger.info(f"🎯 VRAM Usage: ~{6 if LOAD_IN_4BIT else 24}GB")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# NEU - Nach load_model() Funktion
def validate_model_loaded():
    """Validate model is properly loaded"""
    if model is None:
        raise RuntimeError("Model not loaded")
    
    if tokenizer is None:
        raise RuntimeError("Tokenizer not loaded")
    
    # Test inference
    try:
        test_input = tokenizer("Test", return_tensors="pt")
        test_input = {k: v.to(model.device) for k, v in test_input.items()}
        
        with torch.no_grad():
            _ = model.generate(**test_input, max_new_tokens=1)
        
        logger.info("Model validation successful")
        return True
    
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise RuntimeError(f"Model validation failed: {e}")

def build_prompt(request: GenerateRequest) -> str:
    """Build final prompt from request"""
    if request.messages:
        # Use built-in chat template
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    elif request.prompt:
        prompt = request.prompt
    else:
        raise ValueError("Either 'prompt' or 'messages' must be provided")
    
    return prompt

def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    gpu_info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "devices": []
    }
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_info = torch.cuda.mem_get_info(i)
        
        gpu_info["devices"].append({
            "index": i,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / (1024**3), 2),
            "free_memory_gb": round(memory_info[0] / (1024**3), 2),
            "used_memory_gb": round((props.total_memory - memory_info[0]) / (1024**3), 2)
        })
    
    return gpu_info

async def store_generation_metadata(memory_id: int, metadata: Dict[str, Any]):
    """Store extended generation metadata in Cognee"""
    try:
        conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
        
        await conn.execute("""
            INSERT INTO generation_metadata (
                memory_id, prompt_text, prompt_tokens, generated_tokens,
                total_tokens, generation_time_ms, temperature, top_p,
                model_name, model_version, gpu_device
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """,
            memory_id,
            metadata.get("prompt_text"),
            metadata.get("prompt_tokens"),
            metadata.get("generated_tokens"),
            metadata.get("total_tokens"),
            metadata.get("generation_time_ms"),
            metadata.get("temperature"),
            metadata.get("top_p"),
            metadata.get("model_name"),
            metadata.get("model_version"),
            metadata.get("gpu_device")
        )
        
        await conn.close()
        logger.info(f"Stored generation metadata for memory {memory_id}")
        
    except Exception as e:
        logger.warning(f"Failed to store generation metadata: {e}")

# NEU - Startup Event mit Validierung
@app.on_event("startup")
async def startup_event():
    """Startup: Load and validate model"""
    try:
        load_model()
        validate_model_loaded()  # NEU: Model validieren
        logger.info("✅ Model loaded and validated successfully")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Model loading/validation failed: {e}", exc_info=True)
        # ✅ Exit mit Error Code
        import sys
        sys.exit(1)

# NEU - Shutdown Event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Kimi Linear API...")
    
    # Cleanup model
    cleanup_model()
    
    # Final garbage collection
    gc.collect()
    
    logger.info("Shutdown complete")

# NEU - Generate Endpoint mit Validierung und Concurrency Control
@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate with validation and concurrency limit"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # ✅ NEU: Comprehensive Validation
    if request.prompt and request.messages:
        raise HTTPException(
            status_code=400,
            detail="Cannot specify both 'prompt' and 'messages'"
        )
    
    if not request.prompt and not request.messages:
        raise HTTPException(
            status_code=400,
            detail="Must specify either 'prompt' or 'messages'"
        )
    
    # ✅ NEU: Token Budget Check
    estimated_tokens = len(request.prompt or '') // 4  # Rough estimate
    if request.messages:
        estimated_tokens = sum(len(m.content) for m in request.messages) // 4
    
    if estimated_tokens + request.max_tokens > 32000:  # Model context limit
        raise HTTPException(
            status_code=400,
            detail=f"Request too large: {estimated_tokens} input + {request.max_tokens} output > 32K limit"
        )
    
    # ✅ NEU - Input validation
    if request.prompt:
        if len(request.prompt) > 50000:  # ~50K chars = ~12K tokens
            raise HTTPException(
                status_code=400,
                detail="Prompt too long (max 50K characters)"
            )
    
    if request.messages:
        total_length = sum(len(m.content) for m in request.messages)
        if total_length > 50000:
            raise HTTPException(
                status_code=400,
                detail="Combined messages too long (max 50K characters)"
            )
        
        if len(request.messages) > 100:
            raise HTTPException(
                status_code=400,
                detail="Too many messages (max 100)"
            )
    
    # Validate temperature range
    if not 0.0 <= request.temperature <= 2.0:
        raise HTTPException(
            status_code=400,
            detail="Temperature must be between 0.0 and 2.0"
        )
    
    # ✅ NEU - Memory Check
    if not check_memory_available(required_gb=1.5):
        raise HTTPException(
            status_code=503,  # ✅ FIX: _code -> status_code
            detail="Insufficient GPU memory. Please try again later."
        )
    
    # ✅ NEU - Concurrency Control
    if request_semaphore.locked():
        raise HTTPException(
            status_code=503,
            detail=f"Server busy. Max {MAX_CONCURRENT_REQUESTS} concurrent requests allowed."
        )
    
    async with request_semaphore:
        try:
            # ✅ NEU - Mit Timeout
            result = await asyncio.wait_for(
                generate_internal(request),
                timeout=300.0  # 5 Minuten max
            )
            return result
        
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Generation timeout (5 minutes)"
            )
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

# ✅ NEU & KORRIGIERT: Vollständige Funktion mit OOM-Recovery und GPU-Serialisierung
async def generate_internal(
    request: GenerateRequest,
    _retry_count: int = 0  # ✅ Track retries
) -> GenerateResponse:
    """Interne Generierungslogik mit OOM-Wiederherstellung und GPU-Serialisierung"""
    async with _generation_lock:  # ✅ GPU-Zugriff serialisieren
        try:
            start_time = time.time()
            
            # Input vorbereiten
            prompt = build_prompt(request)
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            input_length = inputs["input_ids"].shape[1]
            
            # Generierung (non-blocking via ThreadPoolExecutor)
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,  # Default ThreadPoolExecutor
                lambda: model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            )
            
            # Nur neue Tokens dekodieren
            generated_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Antwort vorbereiten
            response = GenerateResponse(
                text=generated_text,
                tokens_used=len(generated_tokens),
                model=model_config.get("model_name", "kimi-linear-48b"),
                generation_time_ms=generation_time_ms,
                memory_id=None
            )
            
            return response
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA OOM - Full cleanup...")
                
                # ✅ Aggressives Cleanup
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                # ✅ Clear KV-Cache wenn vorhanden
                if hasattr(model, 'past_key_values'):
                    model.past_key_values = None
                
                # ✅ Force garbage collection
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                if _retry_count == 0:
                    # Retry mit stark reduzierten Parametern
                    request.max_tokens = min(request.max_tokens // 2, 256)
                    request.temperature = 0.1  # Deterministic
                    return await generate_internal(request, _retry_count=1)
            else:
                raise
        except Exception as e:
            logger.error(f"Generierungsfehler: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with sparse activation info"""
    gpu_info = get_gpu_info()
    
    return HealthResponse(
        status="healthy" if model is not None else "unloaded",
        model_loaded=model is not None,
        gpu_info={
            "available": gpu_info["available"],
            "device_count": gpu_info["device_count"],
            "sparse_activation": "A3B",
            "active_params_b": 3.0 if model else 0,
            "devices": gpu_info.get("devices", [])
        }
    )

@app.get("/model/architecture")
async def model_architecture():
    """Show architecture details"""
    if not model_config:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "architecture": "Kimi Linear 48B (A3B - Sparse Activation)",
        "description": "48B Total Parameters, 3B Active per Forward Pass",
        "model_config": model_config,
        "gpu_info": get_gpu_info(),
        "performance_notes": [
            "16x faster than dense 48B models",
            f"~{6 if model_config['load_in_4bit'] else 24}GB VRAM in 4-bit",
            "fla-core for efficient sparse attention"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Kimi Linear 48B API (A3B)",
        "version": "1.0.0",
        "model": model_config,
        "status": "running" if model is not None else "loading",
        "features": ["code-generation", "chat-completion", "sparse-activation"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8003")))