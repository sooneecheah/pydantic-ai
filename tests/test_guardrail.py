import asyncio
import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext

pytestmark = pytest.mark.anyio


def simple_model(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content="ok")])


async def test_guardrail_called():
    triggered = asyncio.Event()

    async def gr(messages: list[ModelMessage], ctx: RunContext[None]) -> None:
        await asyncio.sleep(0)
        triggered.set()

    agent = Agent(FunctionModel(simple_model), guardrails=[gr])
    await agent.run("hi")
    assert triggered.is_set()


async def test_guardrail_blocks_until_complete():
    done = False

    async def gr(messages: list[ModelMessage], ctx: RunContext[None]) -> None:
        nonlocal done
        await asyncio.sleep(0.05)
        done = True

    agent = Agent(FunctionModel(simple_model), guardrails=[gr])
    await agent.run("hi")
    assert done

