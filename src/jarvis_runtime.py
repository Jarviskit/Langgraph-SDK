import json
import asyncio
from langgraph.graph.state import CompiledStateGraph

from typing import Any, Callable, Awaitable, final, cast
from ag_ui.core.events import Event
from langgraph.store.memory import InMemoryStore
import requests
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

import socketio
import threading
from .classes import RuntimeMessage, ClientResponseData, MessageEvent, RabbitMQConfig, RuntimeConfig
from .agui_util import encode_event
from .rabbit import AsyncRabbitMQSubscriber


@final
class JarvisKitRuntime:
    namespace: str | None = None # The name of agent space
    namespace_api_key: str | None = None # The api key of agent space
    
    agents: dict[str, CompiledStateGraph[Any, Any, Any]] = {}
    _loop: asyncio.AbstractEventLoop | None = None
    rabbitmq_config: RabbitMQConfig
    
    sio: socketio.Client | None = None

    def __init__(
        self,
        config: RuntimeConfig,
        agents: dict[str, CompiledStateGraph[Any, Any, Any]] = {},
    ):
        self.config = config
        
        self.namespace = config.namespace
        self.namespace_api_key = config.namespace_api_key
        self.runtime_endpoint = config.runtime_endpoint
        self.agents = agents
        
        self.max_concurrent_workers = config.max_concurrent_workers
        self.rabbitmq_config = config.rabbitmq_config
        
        self.store: InMemoryStore = InMemoryStore()
        
        self.pending_responses: dict[str, asyncio.Event] = {}
        self.response_data: dict[str, Any] = {}
        self._connection_event: threading.Event = threading.Event()
        
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    async def initialize(self):
        self.config.validate()
        socket_config = self.config.socket_config
        self.sio = socketio.Client(
            reconnection=socket_config.reconnection,
            reconnection_attempts=socket_config.reconnection_attempts,
            reconnection_delay=socket_config.reconnection_delay,
            reconnection_delay_max=socket_config.reconnection_delay_max,
        )
        self.sio.on("connect", self.on_connect)
        self.sio.on("disconnect", self.on_disconnect)
        self.sio.on("client_response", self.handle_client_response)
        self.sio.connect(
            url=self.runtime_endpoint,
            auth={"namespace_api_key": self.namespace_api_key},
            wait_timeout=self.config.timeout,
            retry=True,
            transports=['polling']
        )

    async def shutdown(self):
        if hasattr(self, 'sio') and self.sio:
            self.sio.disconnect()
        if self._loop:
            self._loop.close()

    def set_max_concurrent_workers(self, max_concurrent_workers: int):
        self.max_concurrent_workers = max_concurrent_workers

    # Agent management
    def add_agent(self, agent_name: str, agent: CompiledStateGraph[Any, Any, Any]):
        self.agents[agent_name] = agent
        print(f"[{agent_name}] has been registered")
    
    def get_agent(self, agent_name: str) -> CompiledStateGraph[Any, Any, Any]:
        agent = self.agents.get(agent_name)
        
        if agent is None:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        return agent
    
    def get_agents(self) -> dict[str, CompiledStateGraph[Any, Any, Any]]:
        return self.agents
        
    async def serve(self, handler: Callable[[CompiledStateGraph[Any, Any, Any], MessageEvent], Awaitable[bool]]):
        ssl_context = self.rabbitmq_config.ssl_context
        rabbitmq_subscriber = AsyncRabbitMQSubscriber(
            url=self.rabbitmq_config.url,
            ssl_context=ssl_context,
            max_concurrent_workers=self.max_concurrent_workers
        )
        await rabbitmq_subscriber.connect()
        await rabbitmq_subscriber.declare_queue(f'tasks_queue:{self.namespace}', durable=True)
            
        await rabbitmq_subscriber.subscribe(
            queue_name=f'tasks_queue:{self.namespace}',
            callback=lambda event: handler(self.get_agent(event.agent_name), event)
        )
        
     # Socket.io events
    def on_connect(self):
        assert self.sio is not None
        self.sio.emit("join_agent_space", { "name": self.namespace, "api_key": self.namespace_api_key })
        self._connection_event.set()  # Signal that connection is established
        print("Connected to agent runtime")
        
    def on_disconnect(self):
        print("Disconnected from agent runtime")
        self._connection_event.clear()  # Clear the connection event
        
    def wait_for_connection(self, timeout: int = 30) -> bool:
        """Wait for the runtime to be connected"""
        return self._connection_event.wait(timeout)
    
    def is_connected(self) -> bool:
        """Check if the runtime is connected"""
        return self._connection_event.is_set()
    
    # Message management
    def convert_message_to_langgraph_message(self, messages: list[RuntimeMessage]) -> list[Any]:
        """Convert the messages to the format expected by langgraph"""
        langgraph_messages = []
        
        for message in messages:
            if message["role"] == "user":
                # User message
                langgraph_messages.append(HumanMessage(id=message["id"], content=message["content"]))
            elif message["role"] == "agent":
                if message.get("toolCallId"):
                    # Tool call message
                    langgraph_messages.append(
                        AIMessage(
                            id=message["id"],
                            content=message["content"],
                            tool_calls=[
                                {
                                    "id": message["toolCallId"],
                                    "name": message["toolName"],
                                    "args": message["toolInput"],
                                    "type": "tool_call"
                                }
                            ]
                        )
                    )
                    
                    if message.get("toolResults"):
                        # Tool result message
                        langgraph_messages.append(
                            ToolMessage(
                                tool_call_id=message["toolCallId"],
                                content=json.dumps(message["toolResults"]),
                                status="success" if message.get("toolStatus") else "error"
                            )
                        )
                else:
                    # Normal text message
                    langgraph_messages.append(AIMessage(id=message["id"], content=message["content"]))
            else:
                print(f"Unknown role: {message['role']}")
        
        return langgraph_messages
        
    def get_messages(self, thread_id: str) -> list[Any]:
        store_messages = self.get_store_messages(thread_id)
        if store_messages:
            return store_messages
        else:
            params = { "threadId": thread_id, "limit": 30 }
            headers = { 
                "x-agent-namespace": self.namespace,
                "x-agent-namespace-secret": self.namespace_api_key
            }
            response = requests.get(f"{self.runtime_endpoint}/agents/get-thread-messages", params=params, headers=headers)
            
            response_data = response.json()
            
            if not response_data.get("success"):
                raise Exception(response_data.get("message", "Unknown error"))
            
            langgraph_messages = self.convert_message_to_langgraph_message(response.json().get("data", []))
            self.set_store_messages(thread_id, langgraph_messages)
            return langgraph_messages
    
    def get_store_messages(self, thread_id: str) -> list[Any]:
        store_data = self.store.get(
            namespace=("thread", thread_id),
            key="memory"
        )
        return store_data.value.get("messages", []) if store_data else []
    
    def set_store_messages(self, thread_id: str, messages: list[Any]):
        self.store.put(
            namespace=("thread", thread_id),
            key="memory",
            value={"messages": messages}
        )
    
    def put_store_message(self, thread_id: str, message: BaseMessage):
        old_messages = self.get_store_messages(thread_id)
        self.set_store_messages(
            thread_id,
            messages=[*old_messages, message]
        )
        
    def clear_store_messages(self, thread_id: str):
        self.store.delete(
            namespace=("thread", thread_id),
            key="memory"
        )
    
    def prepare_tool_input(self, thread_id: str, state: Any) -> dict[str, Any]:
        """
        Prepare the input for the tool node.
        The input is like state of the MessagesState.
        """
        return { "messages": self.get_messages(thread_id), **state }
    
    # AGUI utils
    def send_agui_event(self, thread_id: str, session_id: str, event: Event, order: int):
        assert self.sio is not None
        self.sio.emit(
            event="agui_event",
            data = {
                "threadId": thread_id,
                "sessionId": session_id,
                "event": encode_event(event),
                "order": order
            }
        )
        
    def handle_client_response(self, data: ClientResponseData):
        tool_call_id = cast(str, data.get("toolCallId"))
        keys = self.pending_responses.keys()
        if tool_call_id in keys:
            self.response_data[tool_call_id] = data.get("response")
            if self._loop:
                self._loop.call_soon_threadsafe(self.pending_responses[tool_call_id].set)
        
    async def wait_for_client_response(self, tool_call_id: str, timeout: int = 30) -> Any:
        """
        Wait for client response for a specific tool call ID.
        Returns the response data or None if timeout occurs.
        """
        event = asyncio.Event()
        self.pending_responses[tool_call_id] = event
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            response = self.response_data.get(tool_call_id)
            return response
        except asyncio.TimeoutError:
            return { "success": False, "reason": "Execution timeout" }
        finally:
            self.pending_responses.pop(tool_call_id, None)
            self.response_data.pop(tool_call_id, None)
