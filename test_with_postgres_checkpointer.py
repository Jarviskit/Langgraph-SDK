import asyncio
import ssl
import os
from typing import cast
from dotenv import load_dotenv
from psycopg.rows import DictRow
from src import SocketConfig, default_message_handler, RabbitMQConfig, InitRuntimeParams, JarvisKitRuntimeManager

# Agents
from examples.simple_agent import get_graph as get_simple_agent_graph
from examples.no_stream_mode import get_graph as get_no_stream_mode_graph
from examples.agent_with_client_tool_call import get_graph as get_agent_with_client_tool_call_graph
from examples.agent_with_base_tool_sub_class import get_graph as get_agent_with_base_tool_sub_class_graph

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
load_dotenv()


async def bootstrap():
    async with await AsyncConnection[DictRow].connect(os.getenv('POSTGRES_CONNECTION_STRING') or '') as conn:
        await conn.set_autocommit(True)
        checkpointer = AsyncPostgresSaver(conn)
        await checkpointer.setup()
        
        runtime_endpoint = cast(str, os.getenv('JARVIS_KIT_RUNTIME'))
        
        socket_config = SocketConfig(
            reconnection=True,
            reconnection_attempts=10,
            reconnection_delay=1,
            reconnection_delay_max=5,
            url=runtime_endpoint
        )
        
        agents = {
            "simple_agent": get_simple_agent_graph(checkpointer=checkpointer),
            "no_stream_mode": get_no_stream_mode_graph(checkpointer=checkpointer),
            "agent_with_client_tool_call": get_agent_with_client_tool_call_graph(checkpointer=checkpointer),
            "agent_with_base_tool_sub_class": get_agent_with_base_tool_sub_class_graph(checkpointer=checkpointer)
        }
        
        rabbitmq_config = RabbitMQConfig(url=cast(str, os.getenv('RABBITMQ_URL')))

        params = InitRuntimeParams(
            runtime_endpoint=runtime_endpoint,
            namespace=cast(str, os.getenv('JARVIS_KIT_NAMESPACE')),
            namespace_api_key=cast(str, os.getenv('JARVIS_KIT_NAMESPACE_SECRET')),
            socket_config=socket_config,
            max_concurrent_workers=10,
            rabbitmq_config=rabbitmq_config
        )
        agent_runtime = await JarvisKitRuntimeManager.init_runtime(params, agents=agents)
        
        await agent_runtime.serve(default_message_handler)

if __name__ == "__main__":
    asyncio.run(bootstrap())
    