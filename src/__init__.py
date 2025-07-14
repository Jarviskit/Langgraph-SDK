from typing import Any, cast
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from .tool_node_wrapper import JarvisKitToolNode
from .callback_handler import JarvisKitCallbackHandler
from .classes import SocketConfig, MessageEvent, RabbitMQConfig
from .jarvis_runtime import JarvisKitRuntime
from .runtime_manager import JarvisKitRuntimeManager, InitRuntimeParams
from .rabbit import AsyncRabbitMQSubscriber


async def default_message_handler(agent: CompiledStateGraph[Any, Any, Any], event: MessageEvent) -> bool:
    try:
        config: RunnableConfig = RunnableConfig(
            configurable={
                "thread_id": event.message["thread"],
                "checkpoint_ns": agent.name,
                **event.config
            },
            callbacks=[JarvisKitCallbackHandler(JarvisKitRuntimeManager().get_runtime(), event.message["thread"])]
        )
        
        await agent.ainvoke(cast(Any, {}), config=config)
        
        return True
    except Exception as e:
        print(f"Task execution failed: {e}")
        return False
    
__all__ = [
    "JarvisKitToolNode", 
    "JarvisKitCallbackHandler", 
    "SocketConfig",
    "MessageEvent",
    "RabbitMQConfig",
    "JarvisKitRuntime",
    "JarvisKitRuntimeManager",
    "InitRuntimeParams",
    "default_message_handler",
    "AsyncRabbitMQSubscriber"
]