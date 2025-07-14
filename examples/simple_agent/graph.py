from langchain_core.tools import tool
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer, Command
from langchain.chat_models import init_chat_model
from typing import Literal
from langchain_core.messages import SystemMessage

from src import JarvisKitToolNode, JarvisKitRuntimeManager

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
    current_page: str | None
    protected_key: str | None = "protected_key"

async def agent(state: State, config: RunnableConfig) -> Command[Literal['__end__', 'tool_execution_handler']]:
    jarviskit_runtime = JarvisKitRuntimeManager().get_runtime()
    thread_id = config.get("configurable", {}).get("thread_id", "")
    
    llm = init_chat_model(
        model="openai:gpt-4.1-nano",
        temperature=0,
        streaming=True
    ).bind_tools([scan_cv_tool])
    
    response = await llm.ainvoke(
        [
            SystemMessage("You are a helpful assistant that can help me with my tasks. * **IMPORTANT**: Before using any tool, you **MUST FIRST** explain to the user what you're about to do. Only then should you call the appropriate tool."),
            *jarviskit_runtime.get_messages(thread_id)
        ]
    )
    
    jarviskit_runtime.put_store_message(thread_id, response) # This is important to save the response to the store

    
    if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        return Command( goto="tool_execution_handler" )
    else:
        return Command( goto="__end__" )


def get_graph(checkpointer: Checkpointer):
    # Build the graph
    workflow = StateGraph(State)
    workflow.add_node("agent", agent)
    workflow.add_node("tool_execution_handler", JarvisKitToolNode([scan_cv_tool]))
    
    workflow.add_edge("tool_execution_handler", "agent")
    workflow.set_entry_point("agent")
    
    return workflow.compile(checkpointer=checkpointer, name="simple_agent")