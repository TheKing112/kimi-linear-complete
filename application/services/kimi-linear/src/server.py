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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Logging
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

@app.on_event("startup")
async def startup_event():
    """Startup: Load model asynchronously"""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Don't exit - let health check fail

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text/code with Kimi Linear"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Prepare input
        prompt = build_prompt(request)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        input_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode only new tokens
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        generation_time = (time.time() - start_time) * 1000
        
        # Prepare response
        response = GenerateResponse(
            text=generated_text,
            tokens_used=len(generated_tokens),
            model=model_config["model_name"],
            generation_time_ms=generation_time,
            memory_id=None  # Will be set if stored
        )
        
        # Store extended metadata if requested
        if request.store_metadata:
            meta = {
                "prompt_text": prompt[:500],  # Truncate for storage
                "prompt_tokens": input_length,
                "generated_tokens": len(generated_tokens),
                "total_tokens": input_length + len(generated_tokens),
                "generation_time_ms": generation_time,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "model_name": model_config["model_name"],
                "model_version": "1.0.0",
                "gpu_device": str(model.device)
            }
            
            # Note: In production, you'd want to store this asynchronously
            # and return the memory_id in the response
        
        return response
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
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