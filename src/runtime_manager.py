import threading
from threading import Lock
from typing import Any
from dataclasses import dataclass
from langgraph.graph.state import CompiledStateGraph

from .classes import SocketConfig, RabbitMQConfig, RuntimeConfig
from .jarvis_runtime import JarvisKitRuntime

@dataclass
class InitRuntimeParams:
    namespace: str
    namespace_api_key: str
    runtime_endpoint: str
    socket_config: SocketConfig
    timeout: int = 10
    max_concurrent_workers: int = 2
    rabbitmq_config: RabbitMQConfig | None = None

class JarvisKitRuntimeManager:
    """Singleton pattern cho quản lý runtime"""
    _instance: Any | None = None
    _lock: Lock = threading.Lock()  
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._runtime: JarvisKitRuntime | None = None
            self._initialized: bool = True
    
    async def initialize(self, config: RuntimeConfig, agents: dict[str, CompiledStateGraph[Any, Any, Any]] = {}) -> JarvisKitRuntime:
        """Khởi tạo runtime với proper error handling"""
        if self._runtime is not None:
            await self.shutdown()
        
        self._runtime = JarvisKitRuntime(config, agents=agents)
        await self._runtime.initialize()
        return self._runtime
    
    async def shutdown(self):
        """Proper cleanup"""
        if self._runtime:
            await self._runtime.shutdown()
            self._runtime = None
            
    def get_runtime(self) -> JarvisKitRuntime:
        if self._runtime is None:
            raise RuntimeError("Runtime not initialized")
        return self._runtime
    
    @classmethod
    async def init_runtime(cls, params: InitRuntimeParams, agents: dict[str, CompiledStateGraph[Any, Any, Any]] = {}) -> JarvisKitRuntime:
        config = RuntimeConfig(
            namespace=params.namespace,
            namespace_api_key=params.namespace_api_key,
            runtime_endpoint=params.runtime_endpoint,
            socket_config=params.socket_config,
            rabbitmq_config=params.rabbitmq_config or RabbitMQConfig(url='amqp://localhost'),
            max_concurrent_workers=params.max_concurrent_workers,
            timeout=params.timeout,
        )
        manager = cls()
        runtime = await manager.initialize(config, agents=agents)
        if runtime.wait_for_connection(params.timeout):
            print("Agent runtime initialized successfully")
            print(f"Space name: {runtime.namespace}")
            print(f"Agents: {', '.join(runtime.agents.keys())}")
            print("-"*100)
            return runtime
        else:
            print(f"Failed to initialize agent runtime '{params.namespace}' within {params.timeout} seconds")
            exit(1)
