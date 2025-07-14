from typing_extensions import Annotated
from langchain_core.tools import ArgsSchema, InjectedToolCallId, BaseTool
from typing import TypedDict, override
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer, Command
from langchain.chat_models import init_chat_model
from typing import Literal
from langchain_core.messages import SystemMessage

from src import JarvisKitRuntimeManager, JarvisKitToolNode, JarvisKitRuntime

class ScanCVRequest(BaseModel):
    cv_url: str
    tool_call_id: Annotated[str, InjectedToolCallId]

class ScanCVTool(BaseTool):
    """Tool that scans a CV file (PDF, DOC, image)."""
 
    name: str = "scan_cv_tool"
    description: str = (
        "Use this tool to scan a CV file (PDF, DOC, image)."
    )
    args_schema: ArgsSchema | None = ScanCVRequest
    
    @override
    def _run(
        self,
        cv_url: str,
        tool_call_id: str
    ):
        return self._arun(cv_url, tool_call_id)
    
    
    @override
    async def _arun(
        self,
        cv_url: str,
        tool_call_id: str
    ):
        jarviskit_runtime: JarvisKitRuntime = JarvisKitRuntimeManager().get_runtime()
        response = await jarviskit_runtime.wait_for_client_response(tool_call_id, timeout=120)
        return response


class State(TypedDict):
    pass

async def agent(state: State, config: RunnableConfig) -> Command[Literal['__end__', 'tool_execution_handler']]:
    jarviskit_runtime: JarvisKitRuntime = JarvisKitRuntimeManager().get_runtime()
    thread_id = config.get("configurable", {}).get("thread_id", "")
    llm = init_chat_model(
        model="openai:gpt-4.1-nano",
        temperature=0,
        streaming=True
    ).bind_tools([ScanCVTool()])
    
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
    workflow.add_node("tool_execution_handler", JarvisKitToolNode([ScanCVTool()]))
    
    workflow.add_edge("tool_execution_handler", "agent")
    workflow.set_entry_point("agent")
    
    return workflow.compile(checkpointer=checkpointer, name="agent_with_base_tool_sub_class")