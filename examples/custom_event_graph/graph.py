import asyncio
import random
from typing import TypedDict

from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer, Command
from langchain.chat_models import init_chat_model
from typing import Literal
from langchain_core.messages import SystemMessage

from src import JarvisKitRuntimeManager, JarvisKitRuntime

class State(TypedDict):
    current_page: str | None
    protected_key: str | None = "protected_key"

async def agent(state: State, config: RunnableConfig) -> Command[Literal['__end__']]:
    agent_runtime: JarvisKitRuntime = JarvisKitRuntimeManager().get_runtime()
    thread_id = config.get("configurable", {}).get("thread_id", "")
    
    llm = init_chat_model(
        model="openai:gpt-4.1-nano",
        temperature=0,
        streaming=True
    )
    
    # Send a custom event to the runtime. Better use "replace" strategy to replace the event with the new one in the conversation.
    i = 0
    while i < 100:
        dispatch_custom_event(
            name="progress_event",
            data={
                "progress": i / 100,
                "strategy": "append" if i == 0 else "replace",
                "message": f"The progress is {i}%"
            },
            config=config
        )
        
        i += random.randint(3, 10)
        if i > 100:
            i = 100
        await asyncio.sleep(0.1)
    
    i = 0
    while i < 100:
        dispatch_custom_event(
            name="progress_event",
            data={
                "progress": i / 100,
                "strategy": "append" if i == 0 else "replace",
                "message": f"The progress is {i}%"
            },
            config=config
        )

        i += random.randint(3, 10)
        if i > 100:
            i = 100
        await asyncio.sleep(0.1)
    
    # Process the message
    response = await llm.ainvoke(
        [
            SystemMessage("You are a helpful assistant that can help me with my tasks."),
            *agent_runtime.get_messages(thread_id)
        ]
    )
    
    agent_runtime.put_store_message(thread_id, response) # This is important to save the response to the store
    
    # Send a custom event to the runtime. Better use "append" strategy to append the event to the conversation.
    await asyncio.sleep(1)
    dispatch_custom_event(
        name="alert_event",
        data={
            "strategy": "append",
            "message": "This is a warning message!!!"
        },
        config=config
    )
    
    await asyncio.sleep(1)
    dispatch_custom_event(
        name="alert_event",
        data={
            "strategy": "append",
            "message": "This is an error message!!!"
        },
        config=config
    )

    # Simulate a long running task because the conversation will clear all the custom events when the run finishes
    await asyncio.sleep(5)

    return Command( goto="__end__" )


def get_graph(checkpointer: Checkpointer):
    # Build the graph
    workflow = StateGraph(State)
    workflow.add_node("agent", agent)
    
    workflow.set_entry_point("agent")
    
    return workflow.compile(checkpointer=checkpointer, name="custom_event_graph")