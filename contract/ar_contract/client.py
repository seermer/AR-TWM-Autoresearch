"""Preconfigured clients for the two kernel services, reached over Unix sockets.

The container has no network. The gateway and tool server listen on sockets in
SOCKET_DIR. Details verified on this stack (see the Plan 2 facts):
- the chat model uses httpx over the gateway socket; the MCP 2.x client REQUIRES httpx2;
- the chat model speaks Chat Completions, the most widely supported OpenAI-compatible
  endpoint, and round-trips a reply's reasoning when it returns one (see `reasoning_of`);
- the host is 'localhost' (the MCP server's rebinding guard rejects anything else);
- MCP calls such as data_ingest run the dataset checker and take minutes, and recipe_check
  can wait on the GPU lock then run 3600 s gate steps while hf_download fetches 20 GiB, so
  the session read timeout is long (4 h) -- a client timeout shorter than a server op still
  in progress would just make the agent retry work that hasn't failed. The phase hard cap
  (enforced elsewhere) bounds real hangs;
- the gateway owns upstream retries, so clients never retry.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import httpx2
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

MCP_TIMEOUT_S = 14400.0
LLM_TIMEOUT_S = 900.0


def socket_dir() -> str:
    return os.environ.get("AR_SOCKET_DIR", "/run/ar")


def token() -> str:
    return os.environ.get("AR_TOKEN", "")


def default_model() -> str:
    return os.environ.get("AR_DEFAULT_MODEL", "mock-model")


def reasoning_of(message: dict) -> str | None:
    """A Chat Completions message's reasoning: `reasoning` (OpenAI's field), or `reasoning_content`
    only when `reasoning` is empty. None when the message has neither. The same rule for every
    provider: no provider-specific fields."""
    if "reasoning" not in message and "reasoning_content" not in message:
        return None
    return message.get("reasoning") or message.get("reasoning_content") or ""


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI (Chat Completions) that keeps a reply's reasoning (`reasoning_of`) in
    `additional_kwargs["reasoning"]` and sends it back as `reasoning` with that assistant message,
    so the model keeps its earlier reasoning across turns. Stock langchain-openai drops it both
    ways. A no-op for replies that carry no reasoning."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, raw.get("choices") or []):
            reasoning = reasoning_of(choice.get("message") or {})
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning"] = reasoning
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # payload["messages"] is built one-to-one from these messages
        for message, sent in zip(self._convert_input(input_).to_messages(), payload.get("messages", [])):
            if isinstance(message, AIMessage) and "reasoning" in message.additional_kwargs:
                sent["reasoning"] = message.additional_kwargs["reasoning"]
        return payload


def chat_model(model: str | None = None, **kwargs) -> ReasoningChatOpenAI:
    """A LangChain chat model bound to the gateway (Chat Completions, non-streaming).

    Both clients go over the socket: without an explicit sync client, a sync invoke()
    would try TCP localhost:80 and fail inside the network-less container. Reasoning effort
    is not set here: the gateway enforces OPENAI_EFFORT."""
    sock = os.path.join(socket_dir(), "gateway.sock")
    return ReasoningChatOpenAI(model=model or default_model(), base_url="http://localhost/v1",
                               api_key=token(), use_responses_api=False, max_retries=0,
                               http_client=httpx.Client(transport=httpx.HTTPTransport(uds=sock),
                                                        timeout=LLM_TIMEOUT_S),
                               http_async_client=httpx.AsyncClient(
                                   transport=httpx.AsyncHTTPTransport(uds=sock), timeout=LLM_TIMEOUT_S),
                               **kwargs)


@asynccontextmanager
async def mcp_session(sockets: str | Path | None = None, auth_token: str | None = None):
    """An initialized MCP ClientSession on the kernel tool server."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    sock = os.path.join(str(sockets or socket_dir()), "tools.sock")
    client = httpx2.AsyncClient(transport=httpx2.AsyncHTTPTransport(uds=sock),
                                headers={"Authorization": f"Bearer {auth_token or token()}"},
                                timeout=MCP_TIMEOUT_S)
    async with client, streamable_http_client("http://localhost/mcp", http_client=client) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=MCP_TIMEOUT_S) as session:
            await session.initialize()
            yield session
