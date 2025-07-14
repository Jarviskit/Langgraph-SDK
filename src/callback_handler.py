import os
from ag_ui.core import (
    BaseMessage,
    EventType,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageStartEvent,
    TextMessageEndEvent,
    TextMessageContentEvent,
    ToolCallArgsEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    CustomEvent,
    RunErrorEvent
)
from uuid import UUID
from typing import Any, override, final, cast

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult

from .jarvis_runtime import JarvisKitRuntime



@final
class JarvisKitCallbackHandler(AsyncCallbackHandler):
    run_id_to_disable_stream: UUID | None = None
    
    # This variable will be set when the llm decided to call a tool and will be cleared when the tool call has its result
    current_tool_call_id: str | None = None
    current_message_id: str | None = None
    
    def __init__(self, agent_runtime: JarvisKitRuntime, thread_id: str):
        self.jarvis_runtime = agent_runtime
        self.thread_id = thread_id
        self.order = 0
        self.root_run_id = None
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    # Lifecycle events
    @override
    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if metadata and metadata.get("no_stream", False):
            self.run_id_to_disable_stream = run_id
            return
        
        # Send the RunStartedEvent only once for the first chain start
        if self.order == 0:
            if self.debug:
                print(f'[{EventType.RUN_STARTED} - {self.thread_id}] Chain started')
                print(f'Run ID: {run_id}')
                print(f'Parent Run ID: {parent_run_id}')
                print(f'Tags: {tags}')
                print(f'Metadata: {metadata}')
                print(f'Inputs: {inputs}')
                print(f'Serialized: {serialized}')
                print(f'Kwargs: {kwargs}')
                print(f'[{self.order}] {'-'*30}')
                
            
            self.root_run_id = run_id
            self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=self.thread_id,
                run_id=str(run_id),
                raw_event={
                    "metadata": metadata,
                    "tags": tags,
                    "parent_run_id": str(parent_run_id),
                    "run_id": str(run_id)
                }
            ), self.order)
        
            
            self.order += 1
        
    @override
    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Set run_id_to_disable_stream to None to enable stream again if the run_id is the same as the run_id_to_disable_stream
        if run_id == self.run_id_to_disable_stream:
            self.run_id_to_disable_stream = None
            return
        
        # If this is the last event of the run, clear the store of the thread in runtime and unlock the thread
        if run_id == self.root_run_id:
            if self.debug:
                print(f'[{EventType.RUN_FINISHED} - {self.thread_id}] Chain ended with outputs: {outputs}')
                print(f"Run ID: {run_id}")
                print(f"Parent Run ID: {parent_run_id}")
                print(f"Tags: {tags}")
                print(f"Metadata: {metadata}")
                print(f"Kwargs: {kwargs}")
                print(f'[{self.order}] {'-'*30}')
            
            # Send the RunFinishedEvent only once for the last chain end
            self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=self.thread_id,
                run_id=str(run_id),
                raw_event={
                    "metadata": metadata,
                    "tags": tags,
                    "parent_run_id": str(parent_run_id),
                    "run_id": str(run_id)
                }
            ), self.order)

            self.order += 1
            
            self.jarvis_runtime.clear_store_messages(self.thread_id)

        
    # Message events
    @override
    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[BaseMessage],
        *args: Any,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.run_id_to_disable_stream or (tags and 'no_stream' in tags):
            return
        
        if self.debug:
            print(f'[{EventType.TEXT_MESSAGE_START} - on_chat_model_start - {self.thread_id}] LLM started with prompts: {messages}')
            print(f'Metadata: {metadata}')
            print(f'[{self.order}] {'-'*30}')
        
        self.current_message_id = str(run_id)
        self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id=str(run_id),
            role="assistant"
        ), self.order)
        
        self.order += 1
        
    @override
    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.run_id_to_disable_stream or (tags and 'no_stream' in tags):
            return
        
        if self.debug:
            print(f'[{EventType.TEXT_MESSAGE_START} - on_llm_start - {self.thread_id}] LLM started with prompts: {prompts}')
            print(f'[{self.order}] {'-'*30}')
        
        self.current_message_id = str(run_id)
        self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id=str(run_id),
            role="assistant"
        ), self.order)
        
        self.order += 1
        
    @override
    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.run_id_to_disable_stream or (tags and 'no_stream' in tags):
            return
        
        if self.debug:
            print(f'[{EventType.TEXT_MESSAGE_END} - {self.thread_id}] LLM ended with response: {response}')
            print(f'Metadata: {metadata}')
            print(f'[{self.order}] {'-'*30}')
        
        chat_generation = response.generations[0][0]
        self.current_message_id = None
        self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=str(run_id),
            raw_event=chat_generation.message
        ), self.order)
        
        self.order += 1
        
    @override
    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.run_id_to_disable_stream or (tags and 'no_stream' in tags):
            return
        
        if self.debug:
            print(f'[{EventType.TEXT_MESSAGE_END} - {self.thread_id}] LLM error: {error}')
            print(f'[{self.order}] {'-'*30}')
        
        self.current_message_id = None
        self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), TextMessageEndEvent(
            type=EventType.TEXT_MESSAGE_END,
            message_id=str(run_id)
        ), self.order)
                
        self.order += 1
        
    @override
    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.run_id_to_disable_stream or (tags and 'no_stream' in tags):
            return
        
        chunk = cast(ChatGenerationChunk, chunk)
        if chunk.message.additional_kwargs.get("tool_calls", None):
            tool_call = chunk.message.additional_kwargs.get("tool_calls", [])[0]
            
            if tool_call.get("id", None) is not None:
                # On llm decided to call a tool
                if self.debug:
                    print(f'[{EventType.TOOL_CALL_START} - {self.thread_id} - {tool_call.get("id")}] {tool_call.get("function", {}).get("name", "")}')
                    print(tool_call)
                    print(f'[{self.order}] {'-'*30}')
                
                self.current_tool_call_id = tool_call.get("id")
                self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=tool_call.get("id"),
                    tool_call_name=tool_call.get("function", {}).get("name", ""),
                    parent_message_id=self.current_message_id,
                    raw_event={
                        "id": tool_call.get("id"),
                        "name": tool_call.get("name"),
                        "args": {},
                        "message_id": self.current_message_id,
                    }
                ), self.order)
            else:
                # On streaming tool args
                if self.debug:
                    print(f'[{EventType.TOOL_CALL_ARGS} - {self.thread_id} - {self.current_tool_call_id}] {tool_call.get("function", {}).get("arguments", "")}')
                    print(f'[{self.order}] {'-'*30}')
                
                self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), ToolCallArgsEvent(
                    type=EventType.TOOL_CALL_ARGS,
                    tool_call_id=self.current_tool_call_id or "",
                    delta=tool_call.get("function", {}).get("arguments", ""),
                    raw_event={
                        "message_id": self.current_message_id,
                    }
                ), self.order)
                
            self.order += 1
        else:
            # On llm decided to end the message streaming because of the tool calls(finish_reason is tool_calls)
            if chunk.generation_info and chunk.generation_info.get("finish_reason", None) == "tool_calls":
                if (self.debug):
                    print(f'[{EventType.TOOL_CALL_END} - {self.thread_id} - {self.current_tool_call_id}]')
                    print(f'[{self.order}] {'-'*30}')
                
                self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=self.current_tool_call_id or "",
                ), self.order)
                
                self.order += 1
            
            # On llm streaming text, we need to filter out empty token to avoid sending empty delta
            # Normally, langchain will send empty token once in the beginning of the stream and once in the end of the stream
            if len(token) > 0:
                if (self.debug):
                    print(f'[{EventType.TEXT_MESSAGE_CONTENT} - {self.thread_id} - {run_id}] -> {token}')
                    print(chunk)
                    print(f'Metadata: {metadata}')
                    print(f'[{self.order}] {'-'*30}')
                
                self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=str(run_id),
                    delta=token
                ), self.order)
    
                self.order += 1
        
    @override
    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.run_id_to_disable_stream or (tags and 'no_stream' in tags):
            return
         
        if self.debug:
            print(f'[{EventType.TOOL_CALL_RESULT} - {self.thread_id} - {self.current_tool_call_id}]')
            print(f'output: {output}')
            print(f'[{self.order}] {'-'*30}')
        
        self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), ToolCallResultEvent(
            type=EventType.TOOL_CALL_RESULT,
            tool_call_id=self.current_tool_call_id or "",
            message_id=self.current_message_id or "",
            content=output.content,
            raw_event={
                "message_id": self.current_message_id,
            }
        ), self.order)
        
        self.current_tool_call_id = None
        self.order += 1
        
    @override
    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.run_id_to_disable_stream or (tags and 'no_stream' in tags):
            return
        
        if self.debug:
            print(f'[{EventType.TOOL_CALL_END} - {self.thread_id} - {self.current_tool_call_id}] Tool Error: {error}')
            print(f'[{self.order}] {'-'*30}')
        
        self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), ToolCallEndEvent(
            type=EventType.TOOL_CALL_END,
            tool_call_id=self.current_tool_call_id or "",
        ), self.order)

        self.current_tool_call_id = None
        
        self.order += 1

    # Custom events
    @override
    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.run_id_to_disable_stream or (tags and 'no_stream' in tags):
            return
        
        if self.debug:
            print(
                f"Received event {name} with data: {data}, with tags: {tags}, with metadata: {metadata} and run_id: {run_id}"
            )
            print(f'[{self.order}] {'-'*30}')
        
        self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), CustomEvent(
            type=EventType.CUSTOM,
            name=name,
            value=data
        ), self.order)
        
        self.order += 1
    @override
    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.run_id_to_disable_stream or (tags and 'no_stream' in tags):
            return
            
        # If this is the last event of the run, clear the store of the thread in runtime and unlock the thread
        if run_id == self.root_run_id:
            if self.debug:
                print(f'[{EventType.RUN_ERROR} - {self.thread_id}] Chain error: {error}')
                print(f"Run ID: {run_id}")
                print(f"Parent Run ID: {parent_run_id}")
                print(f"Tags: {tags}")
                print(f'[{self.order}] {'-'*30}')
            
            # Send the RunFinishedEvent only once for the last chain end
            self.jarvis_runtime.send_agui_event(self.thread_id, str(self.root_run_id), RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=str(error),
                code="TASK_FAILED",
                raw_event={
                    "message": str(error),
                    "code": "TASK_FAILED",
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id),
                    "tags": tags,
                    "kwargs": kwargs
                }
            ), self.order)

            self.order += 1
            
            self.jarvis_runtime.clear_store_messages(self.thread_id)
