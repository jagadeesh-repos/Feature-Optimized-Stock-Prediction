"""LangGraph workflow with Groq tool calling and a data-only fallback."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from sqlalchemy import select

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import AgentState
from app.agent.tools import available_tools
from app.models.database_models import AnalysisReport, ConversationReport, db
from app.services.report_service import report_to_dict

logger = logging.getLogger(__name__)


def _latest_thread_reports(thread_id: str, limit: int = 2) -> list[AnalysisReport]:
    """Read only reports explicitly associated with the active conversation."""
    return list(
        db.session.scalars(
            select(AnalysisReport)
            .join(
                ConversationReport,
                ConversationReport.report_id == AnalysisReport.report_id,
            )
            .where(ConversationReport.thread_id == thread_id)
            .order_by(AnalysisReport.created_at.desc())
            .limit(limit)
        )
    )


def fallback_response(thread_id: str, user_text: str) -> str:
    """Answer using linked stored results only when Groq is unavailable."""
    reports = _latest_thread_reports(thread_id)
    if not reports:
        return (
            "Fallback mode: Groq is unavailable, and no saved analysis is linked "
            "to this conversation. I cannot infer or invent market results. Run an "
            "analysis from the dashboard or configure GROQ_API_KEY for tool-assisted chat."
        )

    latest = reports[0]
    data = report_to_dict(latest)
    chunks = [
        "Fallback mode: Groq is unavailable. The following facts come only from "
        "the latest report linked to this conversation.",
        f"Report {latest.report_id}: {latest.ticker}, {latest.start_date} to {latest.end_date}.",
        f"Requested horizon: {latest.prediction_horizon} sessions; threshold: {latest.crisis_threshold:.2%}.",
    ]
    model_data = data.get("models", {})
    for model_name, model_result in model_data.items():
        metrics = model_result.get("metrics", {})
        latest_drawdown = model_result.get("latest_predicted_drawdown")
        if latest_drawdown is not None:
            chunks.append(
                f"{model_name.upper()} latest estimated drawdown: {latest_drawdown:.2%}; "
                "this is a regression estimate, not a probability."
            )
        if metrics:
            chunks.append(
                f"{model_name.upper()} held-out metrics: "
                + ", ".join(
                    f"{name.upper()} {value:.5f}" for name, value in metrics.items()
                )
                + "."
            )
    if len(reports) > 1 and any(
        word in user_text.casefold() for word in ("compare", "previous", "earlier")
    ):
        chunks.append(
            f"Another linked report is available for comparison: {reports[1].report_id}. "
            "Configure Groq to have the agent compare the stored metrics in natural language."
        )
    chunks.append(
        "Latest forecast outcomes are not yet known; held-out metrics refer only to historical test dates."
    )
    return "\n".join(chunks)


def conversation_messages(graph: Any, thread_id: str) -> list[dict[str, str]]:
    """Serialize persisted human/assistant turns for the conversation UI."""
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", []) if state and state.values else []
    serialized = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        content = message.content
        if isinstance(content, str) and content.strip():
            serialized.append({"role": role, "content": content})
    return serialized


def build_agent_graph(
    checkpointer: Any,
    *,
    groq_api_key: str | None,
    model_name: str,
) -> Any:
    """Compile an agent graph with SQLite-backed message history."""
    tools = available_tools()
    groq_model = None
    if groq_api_key:
        from langchain_groq import ChatGroq

        groq_model = ChatGroq(
            model=model_name,
            api_key=groq_api_key,
            temperature=0,
            max_retries=1,
        ).bind_tools(tools)

    def assistant_node(state: AgentState) -> dict[str, Any]:
        """Call Groq, or return a clearly marked database-backed fallback."""
        if groq_model is None:
            return {
                "messages": [
                    AIMessage(
                        content=fallback_response(
                            state["thread_id"], state["messages"][-1].content
                        )
                    )
                ],
                "response_mode": "fallback",
            }
        try:
            response = groq_model.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
            )
            return {"messages": [response], "response_mode": "groq"}
        except Exception:
            logger.exception("Groq request failed; returning stored-report fallback")
            return {
                "messages": [
                    AIMessage(
                        content=fallback_response(
                            state["thread_id"], state["messages"][-1].content
                        )
                    )
                ],
                "response_mode": "fallback",
            }

    def next_node(state: AgentState) -> str:
        """Route actual LLM tool calls to the LangGraph ToolNode."""
        if groq_model is not None and state["messages"][-1].tool_calls:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("assistant", assistant_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("assistant")
    workflow.add_conditional_edges("assistant", next_node, {"tools": "tools", END: END})
    workflow.add_edge("tools", "assistant")
    return workflow.compile(checkpointer=checkpointer)
