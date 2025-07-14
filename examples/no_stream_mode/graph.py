from langchain_core.tools import tool
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer, Command
from langchain.chat_models import init_chat_model
from typing import Literal
from langchain_core.messages import SystemMessage

from src import JarvisKitRuntimeManager, JarvisKitToolNode, JarvisKitRuntime

# When running the agent you should get the following output:
# ----
# User: Hi
# Assistant: Hello! How can I assist you today?
# ----
# The response from the `no_stream_node` and `no_stream_by_llm_config` will never be streamed to the runtime
# And of course, user will not see the response from these nodes in the conversation

@tool
def scan_cv_tool(cv_url: str):
    """Extract structured data from a CV file (PDF, DOC, image)."""
    return {
        "name": "Nghia Pham",
        "experiences": "12 năm",
        "skills": ["Python", "Java", "C#"],
        "education": "Đại học Bách Khoa Hà Nội"
    }


class State(TypedDict):
    current_page: str
    protected_key: str = "protected_key"

async def agent(state: State, config: RunnableConfig) -> Command[Literal['no_stream_node', 'tool_execution_handler']]:
    agent_runtime: JarvisKitRuntime = JarvisKitRuntimeManager().get_runtime()
    thread_id = config.get("configurable", {}).get("thread_id", "")
    llm = init_chat_model(
        model="openai:openai:gpt-4.1-nano",
        temperature=0,
        streaming=True
    ).bind_tools([scan_cv_tool])
    
    response = await llm.ainvoke(
        [
            SystemMessage("You are a helpful assistant that can help me with my tasks. * **IMPORTANT**: Before using any tool, you **MUST FIRST** explain to the user what you're about to do. Only then should you call the appropriate tool."),
            *agent_runtime.get_messages(thread_id)
        ]
    )
    
    # Only put the stream response to the store, to ensure data consistency between actual conversation content and content in message store
    agent_runtime.put_store_message(thread_id, response)
    
    if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        return Command( goto="tool_execution_handler" )
    else:
        return Command( goto="no_stream_node" )
    
async def no_stream_node(state: State, config: RunnableConfig) -> Command[Literal['no_stream_by_llm_config']]:
    agent_runtime: JarvisKitRuntime = JarvisKitRuntimeManager().get_runtime()
    thread_id = config.get("configurable", {}).get("thread_id", "")
    llm = init_chat_model(
        model="openai:openai:gpt-4.1-nano",
        temperature=0,
        streaming=True
    ).bind_tools([scan_cv_tool])
    
    response = await llm.ainvoke(
        [
            SystemMessage("You are a helpful assistant that can help me with my tasks. * **IMPORTANT**: Before using any tool, you **MUST FIRST** explain to the user what you're about to do. Only then should you call the appropriate tool."),
            *agent_runtime.get_messages(thread_id)
        ]
    )
    
    print("No stream node response: ", response)
    
    return Command( goto="no_stream_by_llm_config" )

async def no_stream_by_llm_config(state: State, config: RunnableConfig) -> Command[Literal['__end__']]:
    agent_runtime: JarvisKitRuntime = JarvisKitRuntimeManager().get_runtime()
    thread_id = config.get("configurable", {}).get("thread_id", "")
    llm = init_chat_model(
        model="openai:gpt-4o-mini",
        temperature=0,
        streaming=True
    ).bind_tools([scan_cv_tool])
    
    # Set .with_config(tags=["no_stream"]) to disable streaming in the llm
    response = await llm.with_config(tags=["no_stream"]).ainvoke(
        [
            SystemMessage("You are a helpful assistant that can help me with my tasks. * **IMPORTANT**: Before using any tool, you **MUST FIRST** explain to the user what you're about to do. Only then should you call the appropriate tool."),
            *agent_runtime.get_messages(thread_id)
        ]
    )
    
    print("No stream by llm node response: ", response)
    
    return Command( goto="__end__" )


def get_graph(checkpointer: Checkpointer):
    # Build the graph
    workflow = StateGraph(State)
    workflow.add_node("agent", agent)
    
    # This node and all the sub-nodes of it will not be streamed to the runtime
    workflow.add_node("no_stream_node", no_stream_node, metadata={"no_stream": True})
    
    # The only llm with tags `no_stream` will not be streamed to the runtime
    # If there is a llm without tags `no_stream` in the node, it will be streamed as normal
    workflow.add_node("no_stream_by_llm_config", no_stream_by_llm_config)
    
    workflow.add_node("tool_execution_handler", JarvisKitToolNode([scan_cv_tool]))
    
    # Edges
    workflow.add_edge("tool_execution_handler", "agent")

    # Entry point
    workflow.set_entry_point("agent")
    
    return workflow.compile(checkpointer=checkpointer, name="no_stream_mode")