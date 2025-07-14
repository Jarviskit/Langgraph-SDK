from collections.abc import Sequence
from typing import Any, Callable, final, override
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig

from .runtime_manager import JarvisKitRuntimeManager
from .jarvis_runtime import JarvisKitRuntime

@final
class JarvisKitToolNode(ToolNode):
    def __init__(
        self,
        tools: Sequence[BaseTool | Callable[..., Any]] = [],
        *args: Any,
        name: str = "tools",
        tags: list[str] | None = None,
        handle_tool_errors: bool | str | Callable[..., str] | tuple[type[Exception], ...] = True,
        messages_key: str = "messages",
        **kwargs: Any
    ):
        super().__init__(
            *args,
            tools=tools,
            name=name,
            tags=tags,
            handle_tool_errors=handle_tool_errors,
            messages_key=messages_key,
            **kwargs
        )
        self.jarvis_runtime: JarvisKitRuntime | None = None  # Lazy load runtime
        
    @override
    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        if not config:
            raise ValueError("WrapperToolNode requires a config")
        
        # Lazy load runtime
        if self.jarvis_runtime is None:
            self.jarvis_runtime = JarvisKitRuntimeManager().get_runtime()
        
        assert self.jarvis_runtime is not None  # Type guard
        thread_id = config.get("configurable", {}).get("thread_id", "")
        input_state = self.jarvis_runtime.prepare_tool_input(thread_id, input)
        
        if len(input_state.get("messages", [])) < 1  or input_state.get("messages", [])[-1].tool_calls[0].get("id") is None:
            raise ValueError("There is no tool call id in the message history")
        
        response = super().invoke(
            input=input_state,
            config=config,
            **kwargs
        )
        
        self.jarvis_runtime.put_store_message(thread_id, response.get("messages").pop())
        return response
    
    @override
    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        if not config:
            raise ValueError("WrapperToolNode requires a config")
        
        # Lazy load runtime
        if self.jarvis_runtime is None:
            self.jarvis_runtime = JarvisKitRuntimeManager().get_runtime()
        
        assert self.jarvis_runtime is not None  # Type guard
        
        thread_id = config.get("configurable", {}).get("thread_id", "")
        input_state = self.jarvis_runtime.prepare_tool_input(thread_id, input)
        
        if len(input_state.get("messages", [])) < 1  or input_state.get("messages", [])[-1].tool_calls[0].get("id") is None:
            raise ValueError("There is no tool call id in the message history")
        
        response = await super().ainvoke(
            input=input_state,
            config=config,
            **kwargs
        )
        
        self.jarvis_runtime.put_store_message(thread_id, response.get("messages").pop())
        return response