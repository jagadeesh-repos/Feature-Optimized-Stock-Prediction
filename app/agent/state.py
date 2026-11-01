"""Typed LangGraph state for conversation-scoped assistant messages."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Message history and the stable thread identity for injected tools."""

    messages: Annotated[list[AnyMessage], add_messages]
    thread_id: str
    response_mode: str
