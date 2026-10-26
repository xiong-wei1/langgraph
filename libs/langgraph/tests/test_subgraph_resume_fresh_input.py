"""A finished `checkpointer=True` subgraph called again during a parent
resume must run on its new input instead of replaying its last checkpoint.

The parent propagates its resuming flag to every subgraph call. That flag
describes the parent's run, not the subgraph's: when the subgraph's own
previous run already finished, a later call carrying fresh input is a new
invocation. Runs that are still in flight — paused on a pending
`interrupt()` or at a static breakpoint inside the subgraph — must keep
resuming instead.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict, total=False):
    x: int
    rounds: int
    note: str


class ChildState(TypedDict, total=False):
    x: int
    note: str


def _increment(state: ChildState) -> ChildState:
    return {"x": state["x"] + 1}


def _looping_parent() -> StateGraph:
    child = StateGraph(ChildState)
    child.add_node("increment", _increment)
    child.add_edge(START, "increment")
    child.add_edge("increment", END)

    builder = StateGraph(State)
    builder.add_node("child", child.compile(checkpointer=True))
    builder.add_node("tick", lambda state: {"rounds": state["rounds"] + 1})
    builder.add_edge(START, "child")
    builder.add_edge("child", "tick")
    builder.add_conditional_edges(
        "tick",
        lambda state: "child" if state["rounds"] < 2 else END,
        ["child", END],
    )
    return builder


@pytest.mark.parametrize("breakpoint_kwarg", ["interrupt_before", "interrupt_after"])
def test_finished_subgraph_applies_new_input_after_parent_breakpoint(
    breakpoint_kwarg: str,
) -> None:
    graph = _looping_parent().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    kwargs = {breakpoint_kwarg: ["child"]}
    result = graph.invoke({"x": 0, "rounds": 0}, config, **kwargs)
    for _ in range(6):
        if not graph.get_state(config).next:
            break
        result = graph.invoke(None, config, **kwargs)
    # Two calls, each incrementing once: the second call must not return
    # the first call's checkpointed result.
    assert result["x"] == 2


@pytest.mark.anyio
async def test_finished_subgraph_applies_new_input_after_parent_breakpoint_async() -> (
    None
):
    graph = _looping_parent().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    result = await graph.ainvoke(
        {"x": 0, "rounds": 0}, config, interrupt_before=["child"]
    )
    for _ in range(6):
        state = await graph.aget_state(config)
        if not state.next:
            break
        result = await graph.ainvoke(None, config, interrupt_before=["child"])
    assert result["x"] == 2


def test_subgraph_with_pending_interrupt_still_resumes() -> None:
    calls = {"ask": 0, "bump": 0}

    def ask(state: ChildState) -> ChildState:
        calls["ask"] += 1
        return {"note": interrupt("approve?")}

    def bump(state: ChildState) -> ChildState:
        calls["bump"] += 1
        return {"x": state["x"] + 1}

    child = StateGraph(ChildState)
    child.add_node("ask", ask)
    child.add_node("bump", bump)
    child.add_edge(START, "ask")
    child.add_edge("ask", "bump")
    child.add_edge("bump", END)

    builder = StateGraph(State)
    builder.add_node("child", child.compile(checkpointer=True))
    builder.add_edge(START, "child")
    builder.add_edge("child", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"x": 10}, config)
    result = graph.invoke(Command(resume="yes"), config)
    assert result["x"] == 11
    assert result["note"] == "yes"
    # `ask` re-runs so `interrupt()` can return the resume value; `bump`
    # runs exactly once, after the resume.
    assert calls == {"ask": 2, "bump": 1}


def test_subgraph_paused_at_own_breakpoint_still_continues() -> None:
    calls = {"a": 0, "b": 0}

    def node_a(state: ChildState) -> ChildState:
        calls["a"] += 1
        return {"x": state["x"] + 1}

    def node_b(state: ChildState) -> ChildState:
        calls["b"] += 1
        return {"x": state["x"] * 10}

    child = StateGraph(ChildState)
    child.add_node("a", node_a)
    child.add_node("b", node_b)
    child.add_edge(START, "a")
    child.add_edge("a", "b")
    child.add_edge("b", END)

    builder = StateGraph(State)
    builder.add_node("child", child.compile(checkpointer=True, interrupt_before=["b"]))
    builder.add_edge(START, "child")
    builder.add_edge("child", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}}

    graph.invoke({"x": 1}, config)
    result = graph.invoke(None, config)
    # The paused run continues: `a` is not re-run and `b` completes it.
    assert result["x"] == 20
    assert calls == {"a": 1, "b": 1}
