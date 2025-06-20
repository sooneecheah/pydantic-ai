import asyncio

import pytest

from pydantic_ai import Agent
from pydantic_ai._agent_graph import InputGuardrail
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext

pytestmark = pytest.mark.anyio


def simple_model(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='ok')])


async def test_output_guardrail_called():
    triggered = asyncio.Event()

    async def gr(messages: list[ModelMessage], ctx: RunContext[None]) -> None:
        await asyncio.sleep(0)
        triggered.set()

    agent = Agent(FunctionModel(simple_model), output_guardrails=[gr])
    await agent.run('hi')
    assert triggered.is_set()


async def test_input_guardrail_called():
    triggered = asyncio.Event()

    async def gr(messages: list[ModelMessage], ctx: RunContext[None]) -> None:
        await asyncio.sleep(0)
        triggered.set()

    agent = Agent(FunctionModel(simple_model), input_guardrails=[gr])
    await agent.run('hi')
    assert triggered.is_set()


async def test_output_guardrail_blocks_until_complete():
    done = False

    async def gr(messages: list[ModelMessage], ctx: RunContext[None]) -> None:
        nonlocal done
        await asyncio.sleep(0.05)
        done = True

    agent = Agent(FunctionModel(simple_model), output_guardrails=[gr])
    await agent.run('hi')
    assert done


async def test_input_guardrail_blocks_until_complete_when_specified():
    events: list[str] = []

    async def gr(messages: list[ModelMessage], ctx: RunContext[None]) -> None:
        events.append('gr_start')
        await asyncio.sleep(0.05)
        events.append('gr_end')

    def model(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        events.append('model')
        return ModelResponse(parts=[TextPart(content='ok')])

    agent = Agent(FunctionModel(model), input_guardrails=[InputGuardrail(gr, is_blocking=True)])
    await agent.run('hi')
    assert events == ['gr_start', 'gr_end', 'model']


async def test_input_guardrail_runs_in_parallel_by_default():
    events: list[str] = []

    async def gr(messages: list[ModelMessage], ctx: RunContext[None]) -> None:
        events.append('gr_start')
        await asyncio.sleep(0.05)
        events.append('gr_end')

    def model(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        events.append('model')
        return ModelResponse(parts=[TextPart(content='ok')])

    agent = Agent(FunctionModel(model), input_guardrails=[gr])
    await agent.run('hi')
    assert events[0] == 'gr_start'
    assert events[1] == 'model'
    assert events[-1] == 'gr_end'
