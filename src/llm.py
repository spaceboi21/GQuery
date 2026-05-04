"""
Unified LLM adapter — OpenAI, Anthropic, and Ollama behind one interface.

Ollama uses the OpenAI-compatible endpoint (base_url override).
Anthropic requires minor format normalisation for tool calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Generator

from .config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.cfg = settings
        self._client = self._build_client()
        # Updated after each stream call; accumulated by the agent
        self.last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        if self.cfg.llm_provider == "anthropic":
            return self._anthropic_chat(messages, tools)
        return self._openai_chat(messages, tools)

    def stream_chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Generator[str | ToolCall, None, None]:
        """Yields str chunks during text generation, or ToolCall objects."""
        if self.cfg.llm_provider == "anthropic":
            yield from self._anthropic_stream(messages, tools)
        else:
            yield from self._openai_stream(messages, tools)

    # ------------------------------------------------------------------
    # OpenAI / Ollama
    # ------------------------------------------------------------------

    def _build_client(self):
        if self.cfg.llm_provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=self.cfg.llm_api_key or None)

        from openai import OpenAI
        kwargs: dict = {}
        key = self.cfg.llm_api_key or "ollama"
        kwargs["api_key"] = key
        if self.cfg.llm_base_url:
            kwargs["base_url"] = self.cfg.llm_base_url
        elif self.cfg.llm_provider == "ollama":
            kwargs["base_url"] = "http://localhost:11434/v1"
        return OpenAI(**kwargs)

    def _openai_chat(self, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
        kwargs: dict = dict(
            model=self.cfg.llm_model,
            messages=messages,
            temperature=self.cfg.llm_temperature,
            max_tokens=self.cfg.llm_max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tcs = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tcs.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments or "{}"),
                ))
        return LLMResponse(
            content=msg.content,
            tool_calls=tcs,
            stop_reason=resp.choices[0].finish_reason or "stop",
        )

    def _openai_stream(
        self, messages: list[dict], tools: list[dict] | None
    ) -> Generator[str | ToolCall, None, None]:
        kwargs: dict = dict(
            model=self.cfg.llm_model,
            messages=messages,
            temperature=self.cfg.llm_temperature,
            max_tokens=self.cfg.llm_max_tokens,
            stream=True,
        )
        if tools:
            kwargs["tools"] = tools

        # Request usage stats in the final chunk (OpenAI-native; silently ignored by Ollama)
        try:
            kwargs["stream_options"] = {"include_usage": True}
        except Exception:
            pass

        tool_call_chunks: dict[int, dict] = {}

        for chunk in self._client.chat.completions.create(**kwargs):
            # Capture token usage from the final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                pt = getattr(chunk.usage, "prompt_tokens", 0) or 0
                ct = getattr(chunk.usage, "completion_tokens", 0) or 0
                self.last_usage = {
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": pt + ct,
                }

            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Text token
            if delta.content:
                yield delta.content

            # Tool call fragments (may arrive across multiple chunks)
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    idx = tc_chunk.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {"id": "", "name": "", "args": ""}
                    if tc_chunk.id:
                        tool_call_chunks[idx]["id"] = tc_chunk.id
                    if tc_chunk.function:
                        if tc_chunk.function.name:
                            tool_call_chunks[idx]["name"] += tc_chunk.function.name
                        if tc_chunk.function.arguments:
                            tool_call_chunks[idx]["args"] += tc_chunk.function.arguments

            finish = chunk.choices[0].finish_reason if chunk.choices else None
            if finish in ("tool_calls", "stop") and tool_call_chunks:
                for tc in tool_call_chunks.values():
                    try:
                        args = json.loads(tc["args"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    yield ToolCall(id=tc["id"], name=tc["name"], args=args)
                tool_call_chunks.clear()

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------

    def _anthropic_chat(self, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
        system, ant_msgs = _to_anthropic_messages(messages)
        kwargs: dict = dict(
            model=self.cfg.llm_model,
            max_tokens=self.cfg.llm_max_tokens,
            messages=ant_msgs,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _openai_tools_to_anthropic(tools)

        resp = self._client.messages.create(**kwargs)
        tcs: list[ToolCall] = []
        text_parts: list[str] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tcs.append(ToolCall(id=block.id, name=block.name, args=block.input or {}))
        return LLMResponse(
            content="\n".join(text_parts) or None,
            tool_calls=tcs,
            stop_reason=resp.stop_reason or "stop",
        )

    def _anthropic_stream(
        self, messages: list[dict], tools: list[dict] | None
    ) -> Generator[str | ToolCall, None, None]:
        import anthropic

        system, ant_msgs = _to_anthropic_messages(messages)
        kwargs: dict = dict(
            model=self.cfg.llm_model,
            max_tokens=self.cfg.llm_max_tokens,
            messages=ant_msgs,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _openai_tools_to_anthropic(tools)

        current_tool: dict | None = None
        with self._client.messages.stream(**kwargs) as stream:
            for event in stream:
                if isinstance(event, anthropic.types.RawContentBlockStartEvent):
                    if event.content_block.type == "tool_use":
                        current_tool = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "args": "",
                        }
                elif isinstance(event, anthropic.types.RawContentBlockDeltaEvent):
                    delta = event.delta
                    if hasattr(delta, "text"):
                        yield delta.text
                    elif hasattr(delta, "partial_json") and current_tool is not None:
                        current_tool["args"] += delta.partial_json
                elif isinstance(event, anthropic.types.RawContentBlockStopEvent):
                    if current_tool is not None:
                        try:
                            args = json.loads(current_tool["args"] or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        yield ToolCall(
                            id=current_tool["id"],
                            name=current_tool["name"],
                            args=args,
                        )
                        current_tool = None
            # Capture usage from final message
            try:
                final = stream.get_final_message()
                pt = getattr(final.usage, "input_tokens", 0) or 0
                ct = getattr(final.usage, "output_tokens", 0) or 0
                self.last_usage = {
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": pt + ct,
                }
            except Exception:
                pass


# ------------------------------------------------------------------
# Format helpers
# ------------------------------------------------------------------

def _to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split system prompt out and normalise tool results for Anthropic format."""
    system = ""
    result: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "tool":
            result.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": m["content"],
                    }
                ],
            })
        elif m["role"] == "assistant" and "tool_calls" in m:
            content = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"] or "{}"),
                })
            result.append({"role": "assistant", "content": content})
        else:
            result.append(m)
    return system, result


def _openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI tool schema format to Anthropic format."""
    out = []
    for t in tools:
        fn = t.get("function", {})
        out.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out
