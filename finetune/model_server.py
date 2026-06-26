"""
模块7：model_server.py（微调模型推理服务）

文件位置：finetune/model_server.py
负责人：李广旭

功能说明：
    加载 LoRA adapter，启动一个本地 HTTP 服务，提供和 OpenAI 一样格式的 API。
    这样 completionModule.js 不用改代码就能调用微调模型

接口规范来源：接口参数规范文档 V2.0
"""

import os

# ============================================================================
# 强制离线模式：必须在 import transformers / huggingface_hub 之前设置！
# ----------------------------------------------------------------------------
# 新版 transformers(5.x) 在加载 tokenizer / 模型时会主动联网拉取
# chat template、custom_generate 模块等文件。当网络走代理/镜像且不稳定时，
# 这些请求会抛 httpx.RemoteProtocolError / "client has been closed" 导致
# 加载直接崩溃。权重已完整缓存到本地，根本不需要联网。
#
# 这里把所有联网开关一次性关死，让 HF 全程只读本地缓存、绝不发网络请求。
# 注意：os.environ 必须在任何 HF 库被 import 之前赋值，import 之后再设无效。
# ============================================================================
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # huggingface_hub 离线
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # transformers 离线
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import json
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# 可选依赖检查
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: fastapi and uvicorn not installed. Run: pip install fastapi uvicorn pydantic")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers or peft not installed. Run: pip install transformers peft accelerate")


@dataclass
class ServerConfig:
    """服务器配置"""
    base_model: str
    adapter_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    model_name: str = ""
    load_in_4bit: bool = False  # CPU 上强制 4bit 量化
    # no_adapter=True 时强制只跑基座模型（baseline），忽略 adapter_path。
    # 用于干净地对比「无 LoRA vs 有 LoRA」，不必删除 adapter 目录。
    no_adapter: bool = False


class ModelLoader:
    """模型加载器"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.adapter_loaded = False
        # adapter_active：如实记录「本次服务是否真的带 LoRA」。
        # 区别于 adapter_loaded —— 后者在 merge_and_unload 后会被重置为 False
        # （因为合并后已不是 PeftModel），导致 /health 永远显示 false，
        # 根本分不清当前测的是 baseline 还是 LoRA。对比微调效果时这是致命的，
        # 所以单独用 adapter_active 标记，merge 后依然保持 True。
        self.adapter_active = False
    
    def load(self) -> Dict[str, Any]:
        """
        加载基座模型和 LoRA adapter
        
        Returns:
            BUG-007 修复：返回结构与 start_server() 保持一致，均包含 status 和 error 字段
            - status: "loaded" 表示加载成功
            - error: 空字符串表示正常
        """
        try:
            if not TRANSFORMERS_AVAILABLE:
                return {
                    "status": "error",
                    "error": "transformers or peft not installed. Please run: pip install transformers peft accelerate"
                }
            
            print(f"Loading base model: {self.config.base_model}")

            # 基座模型加载路径：必须用 base_model（如 "Qwen/Qwen2.5-Coder-1.5B-Instruct"）。
            # 注意 config.model_name 是 start_server 拼出来的"显示标签"
            # （形如 "Qwen2.5-Coder-1.5B-Instruct-lora_adapter"），不是可加载的模型路径，
            # 误用它会被当成 HuggingFace repo id 去查找而报 OSError。
            model_path = self.config.base_model

            # 设备自适应：有 CUDA 才用 GPU + 4bit 量化 + float16；
            # CPU 环境下 4bit(bitsandbytes) 不可用、float16 极慢，必须改用 float32 全精度。
            has_cuda = torch.cuda.is_available()
            print(f"CUDA available: {has_cuda} -> {'GPU(4bit/fp16)' if has_cuda else 'CPU(fp32)'} 模式")

            # 加载 tokenizer
            # local_files_only=True：只读本地缓存，绝不发起网络请求。
            # 新版 transformers 会在加载时尝试联网拉 chat template / custom_generate，
            # 代理或镜像不稳定时直接崩溃；权重已完整缓存，强制离线即可。
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left",
                local_files_only=True,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 按设备组装加载参数
            # local_files_only=True：同 tokenizer，强制离线，避免 load_custom_generate 等联网点崩溃
            load_kwargs = {"trust_remote_code": True, "local_files_only": True}
            if has_cuda:
                # GPU：4bit 量化省显存
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["device_map"] = "auto"
                load_kwargs["torch_dtype"] = torch.float16
            elif self.config.load_in_4bit:
                # CPU：强制 4bit 量化加载大模型
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float32,
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["torch_dtype"] = torch.float32
                load_kwargs["device_map"] = "cpu"
            else:
                # CPU：不量化、全精度 float32，显式放到 cpu
                load_kwargs["torch_dtype"] = torch.float32
                load_kwargs["device_map"] = "cpu"

            # 加载基座模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs,
            )
            
            # 尝试加载 LoRA adapter
            # no_adapter=True 时显式跳过，用于跑 baseline（无微调）对照组。
            if self.config.no_adapter:
                print("--no-adapter 指定：跳过 LoRA，使用纯基座模型（baseline）")
            elif os.path.exists(self.config.adapter_path):
                print(f"Loading LoRA adapter from: {self.config.adapter_path}")
                try:
                    peft_config = PeftConfig.from_pretrained(self.config.adapter_path)
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        self.config.adapter_path
                    )
                    self.adapter_loaded = True
                    print("LoRA adapter loaded successfully")
                except Exception as e:
                    print(f"Warning: Failed to load LoRA adapter: {e}")
                    print("Using base model without adapter")
                    self.adapter_loaded = False
            else:
                print(f"Adapter path not found: {self.config.adapter_path}")
                print("Using base model without adapter")

            # 合并 adapter 到 base model（可选，提升推理速度）
            if self.adapter_loaded:
                print("Merging LoRA adapter with base model...")
                self.model = self.model.merge_and_unload()
                # 关键：merge 只是把权重融进基座、卸掉 PeftModel 包装，
                # 微调效果仍在。adapter_loaded 反映「是否还挂着 Peft 包装」，
                # 这里置 False；但 adapter_active 记录「本服务是否含微调权重」，
                # 必须保持 True，否则 /health 无法区分 baseline 与 LoRA，
                # 整个对比实验失去意义。
                self.adapter_loaded = False
                self.adapter_active = True
            
            self.model.eval()
            
            return {
                "status": "loaded",
                "error": ""
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to load model: {str(e)}\n{traceback.format_exc()}"
            }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成代码补全
        
        Args:
            prompt: 输入提示词
            **kwargs: generation 参数
            
        Returns:
            生成的文本
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        # 构建输入
        messages = [
            {"role": "system", "content": "You are a helpful code assistant. Complete the code based on the given context."},
            {"role": "user", "content": prompt}
        ]
        
        # 使用 chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 生成参数
        generation_config = {
            "max_new_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.95),
            "do_sample": kwargs.get("do_sample", True),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        # 解码（去掉输入部分）
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text


class ChatCompletionRequest(BaseModel):
    """Chat completions 请求模型"""
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    n: Optional[int] = 1


class CompletionRequest(BaseModel):
    """Completions 请求模型"""
    model: str
    prompt: str
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    n: Optional[int] = 1


def create_app(config: ServerConfig, model_loader: ModelLoader) -> Any:
    """创建 FastAPI 应用"""
    
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Run: pip install fastapi uvicorn pydantic")
    
    app = FastAPI(title="LoRA Model Server", version="1.0.0")
    
    @app.get("/health")
    async def health_check():
        """健康检查接口

        adapter_active 如实反映本服务是否含 LoRA 微调权重（含 merge 后的情况），
        用于区分 baseline 与 LoRA 两次运行，是对比实验可信的前提。
        """
        return {
            "status": "healthy",
            "model": config.model_name,
            "adapter_active": model_loader.adapter_active,
            "mode": "lora" if model_loader.adapter_active else "baseline"
        }
    
    @app.get("/v1/models")
    async def list_models():
        """列出可用模型"""
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model_name,
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "local",
                }
            ]
        }
    
    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        """Chat completions 接口（OpenAI 兼容）

        注意：必须用普通 def，不能用 async def。
        model_loader.generate() 是阻塞的 CPU 密集调用，CPU 上跑 1.5B 难题
        要数分钟。若放在 async def 里会冻结 uvicorn 的事件循环，期间无法处理
        TCP 保活/收发，客户端或 OS 会重置连接（WinError 10054）。
        改为普通 def 后 FastAPI 会自动把它丢到线程池执行，事件循环不再被冻结。
        """
        try:
            # 提取用户消息
            user_message = ""
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            if not user_message:
                return {
                    "error": {
                        "message": "No user message found",
                        "type": "invalid_request_error"
                    }
                }
            
            # 生成 n 个候选：必须逐个独立生成，不能生成一次再复制 n 份。
            # 复制会让 Pass@k 的多个候选完全相同，采样多样性失效，指标失真。
            n = request.n or 1
            generated_texts = [
                model_loader.generate(
                    user_message,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    do_sample=request.temperature > 0
                )
                for _ in range(n)
            ]

            # 构建响应
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": 1234567890,
                "model": request.model,
                "choices": [
                    {
                        "index": i,
                        "message": {
                            "role": "assistant",
                            "content": text
                        },
                        "finish_reason": "stop"
                    }
                    for i, text in enumerate(generated_texts)
                ],
                "usage": {
                    "prompt_tokens": 0,  # 简化版不计算
                    "completion_tokens": sum(len(t.split()) for t in generated_texts),
                    "total_tokens": sum(len(t.split()) for t in generated_texts)
                }
            }
            
        except Exception as e:
            return {
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
    
    @app.post("/v1/completions")
    def completions(request: CompletionRequest):
        """Completions 接口（OpenAI 兼容）

        同 chat_completions：用普通 def 让阻塞生成跑在线程池，避免冻结事件循环。
        """
        try:
            # 生成
            generated_text = model_loader.generate(
                request.prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                do_sample=request.temperature > 0
            )
            
            # 构建响应
            return {
                "id": f"cmppl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": 1234567890,
                "model": request.model,
                "choices": [
                    {
                        "text": generated_text,
                        "index": i,
                        "finish_reason": "stop"
                    }
                    for i in range(request.n)
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(generated_text.split())
                }
            }
            
        except Exception as e:
            return {
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
    
    return app


def start_server(base_model: str, adapter_path: str, host: str = "0.0.0.0",
                 port: int = 8000, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.9,
                 no_adapter: bool = False,
                 load_in_4bit: bool = False) -> Dict[str, Any]:
    """
    启动 LoRA 模型推理服务
    
    参数说明（来自接口规范文档）：
    - base_model: 基座模型的名字，如 "Qwen2.5-Coder-7B-Instruct"
    - adapter_path: LoRA adapter 权重文件路径
    - host: 服务监听 IP，默认 "0.0.0.0"
    - port: 服务监听端口，默认 8000
    - max_model_len: 模型一次最多处理多少 token，默认 4096
    - gpu_memory_utilization: 最多用多少比例的 GPU 显存，默认 0.9
    
    返回值（BUG-007 修复：严格对齐接口规范）：
    - endpoint: 服务的访问地址，如 "http://localhost:8000/v1"
    - model_name: 模型的名字，如 "qwen2.5-coder-7b-lora"
    - status: "running" 表示启动成功，"error" 表示失败
    - error: 正常是空字符串，出错时写错误原因
    
    使用示例：
        result = start_server(
            base_model="Qwen2.5-Coder-7B-Instruct",
            adapter_path="./output/lora_adapter",
            port=8000
        )
        # result = {"endpoint": "...", "model_name": "...", "status": "...", "error": ""}
    """
    try:
        # 构建模型名称：baseline（无 LoRA）与 LoRA 用不同后缀，
        # 这样评测报告 / 输出目录能一眼区分两组实验，避免张冠李戴。
        base_short = base_model.split('/')[-1]
        if no_adapter:
            model_name = f"{base_short}-baseline"
        else:
            adapter_name = os.path.basename(adapter_path.rstrip("/\\")) if adapter_path else "base"
            model_name = f"{base_short}-{adapter_name}"

        # 创建配置
        config = ServerConfig(
            base_model=base_model,
            adapter_path=adapter_path,
            host=host,
            port=port,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            model_name=model_name,
            no_adapter=no_adapter,
            load_in_4bit=load_in_4bit,
        )

        # 加载模型
        print("Loading model...")
        model_loader = ModelLoader(config)
        load_result = model_loader.load()
        
        if load_result.get("status") == "error":
            return {
                "endpoint": "",
                "model_name": model_name,
                "status": "error",
                "error": load_result.get("error", "Failed to load model")
            }
        
        # 创建 FastAPI 应用
        app = create_app(config, model_loader)
        
        # 启动服务器（在后台）
        endpoint = f"http://{host}:{port}/v1"
        
        print(f"Starting server at {endpoint}")
        print(f"API endpoints:")
        print(f"  - POST /v1/chat/completions (main)")
        print(f"  - POST /v1/completions")
        print(f"  - GET  /v1/models")
        print(f"  - GET  /health")
        
        # 使用 threading 在后台启动服务器
        import threading
        def run_server():
            uvicorn.run(app, host=host, port=port, log_level="info")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        return {
            "endpoint": endpoint,
            "model_name": model_name,
            "status": "running",
            "error": ""
        }
        
    except Exception as e:
        return {
            "endpoint": "",
            "model_name": "",
            "status": "error",
            "error": f"Failed to start server: {str(e)}\n{traceback.format_exc()}"
        }


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="LoRA Model Server - 启动微调模型推理服务")
    
    parser.add_argument("--base-model", type=str, required=True,
                        help="基座模型名字，如 Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="",
                        help="LoRA adapter 权重文件路径（跑 baseline 时可不填或配合 --no-adapter）")
    parser.add_argument("--no-adapter", action="store_true",
                        help="不加载 LoRA，使用纯基座模型跑 baseline 对照组")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="服务监听 IP，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=8000,
                        help="服务监听端口，默认 8000")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="强制 4bit 量化加载（CPU 上加载大模型时使用）")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="最大 token 长度，默认 4096")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU 显存使用比例，默认 0.9")

    args = parser.parse_args()

    result = start_server(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        no_adapter=args.no_adapter,
        host=args.host,
        port=args.port,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        load_in_4bit=args.load_in_4bit
    )
    
    print("\n=== Server Start Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if result["status"] == "running":
        print(f"\nServer is running at {result['endpoint']}")
        print("Press Ctrl+C to stop")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
    else:
        print(f"\nFailed to start server: {result['error']}")
        exit(1)


if __name__ == "__main__":
    main()
