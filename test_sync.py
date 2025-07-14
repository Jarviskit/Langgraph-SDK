import os
import asyncio
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from src import SocketConfig, default_message_handler, InitRuntimeParams, JarvisKitRuntimeManager
from typing import cast

# Agents
from examples.simple_agent import get_graph as get_simple_agent_graph
from examples.no_stream_mode import get_graph as get_no_stream_mode_graph
from examples.agent_with_client_tool_call import get_graph as get_agent_with_client_tool_call_graph
from examples.agent_with_base_tool_sub_class import get_graph as get_agent_with_base_tool_sub_class_graph
from examples.custom_event_graph import get_graph as get_custom_event_graph
from src.classes import RabbitMQConfig

load_dotenv()


async def bootstrap():
    checkpointer = MemorySaver()
    runtime_endpoint = cast(str, os.getenv('JARVIS_KIT_RUNTIME'))
    socket_config = SocketConfig(
        reconnection=True,
        reconnection_attempts=10,   
        reconnection_delay=1,
        reconnection_delay_max=5,
        url=runtime_endpoint,
        timeout=10
    )
    
    rabbitmq_config = RabbitMQConfig(
        url=cast(str, os.getenv('JARVIS_KIT_RABBITMQ_URI'))
    )

    agents = {
        "simple_agent": get_simple_agent_graph(checkpointer=checkpointer),
        "no_stream_mode": get_no_stream_mode_graph(checkpointer=checkpointer),
        "agent_with_client_tool_call": get_agent_with_client_tool_call_graph(checkpointer=checkpointer),
        "agent_with_base_tool_sub_class": get_agent_with_base_tool_sub_class_graph(checkpointer=checkpointer),
        "custom_event_graph": get_custom_event_graph(checkpointer=checkpointer)
    }

    params = InitRuntimeParams(
        namespace=cast(str, os.getenv('JARVIS_KIT_NAMESPACE')),
        namespace_api_key=cast(str, os.getenv('JARVIS_KIT_NAMESPACE_SECRET')),
        runtime_endpoint=runtime_endpoint,
        socket_config=socket_config,
        max_concurrent_workers=5,
        rabbitmq_config=rabbitmq_config
    )
    runtime = await JarvisKitRuntimeManager.init_runtime(params, agents=agents)
    
    await runtime.serve(default_message_handler)
    
if __name__ == "__main__":
    asyncio.run(bootstrap())