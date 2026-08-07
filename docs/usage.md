# Usage

The Agents SDK automatically tracks token usage for every run. You can access it from the run context and use it to monitor costs, enforce limits, or record analytics.

## What is tracked

- **requests**: number of LLM API calls made
- **input_tokens**: total input tokens sent
- **output_tokens**: total output tokens received
- **total_tokens**: input + output
- **request_usage_entries**: list of per-request usage breakdowns
- **details**:
  - `input_tokens_details.cached_tokens`
  - `output_tokens_details.reasoning_tokens`

## Accessing usage from a run

After `Runner.run(...)`, access usage via `result.context_wrapper.usage`.

```python
result = await Runner.run(agent, "What's the weather in Tokyo?")
usage = result.context_wrapper.usage

print("Requests:", usage.requests)
print("Input tokens:", usage.input_tokens)
print("Output tokens:", usage.output_tokens)
print("Total tokens:", usage.total_tokens)
```

Usage is aggregated across all model calls during the run (including tool calls and handoffs).

### Preserving provider usage fields

Normalized `Usage` values intentionally default missing counters to zero so they can be aggregated safely. If your application must distinguish a field omitted by the provider from a field explicitly reported as zero, opt in to the provider usage snapshot:

```python
agent = Agent(
    name="Assistant",
    model_settings=ModelSettings(preserve_raw_usage=True),
)
result = await Runner.run(agent, "Hello")

raw_usage = result.raw_responses[-1].raw_usage
if raw_usage is not None:
    input_details = raw_usage.get("input_tokens_details")
    cached_tokens_was_reported = (
        isinstance(input_details, dict) and "cached_tokens" in input_details
    )
```

`ModelResponse.raw_usage` is a JSON-compatible snapshot of the usage object received by the model adapter before the Agents SDK normalizes it. Omitted keys remain absent, while explicit zero and null values remain present. The snapshot reflects any transformations already performed by an upstream provider adapter; it is `None` when preservation is disabled or no usage payload reaches the Agents SDK.

This option does not ask the provider to return usage. For streamed Chat Completions backends that require it, also set `include_usage=True`. Raw usage is retained only on completed model responses and is not persisted through `RunState` serialization.

### Enabling usage with third-party adapters

Usage reporting varies across third-party adapters and provider backends. If you rely on adapter-backed models and need accurate `result.context_wrapper.usage` values:

- With `AnyLLMModel`, usage is propagated automatically when the upstream provider returns it. For streamed Chat Completions backends, you may need `ModelSettings(include_usage=True)` before usage chunks are emitted.
- With `LitellmModel`, some provider backends do not report usage by default, so `ModelSettings(include_usage=True)` is often required.

Review the adapter-specific notes in the [Third-party adapters](models/index.md#third-party-adapters) section of the Models guide and validate the exact provider backend you plan to deploy.

## Per-request usage tracking

The SDK automatically tracks usage for each API request in `request_usage_entries`, useful for detailed cost calculation and monitoring context window consumption.

```python
result = await Runner.run(agent, "What's the weather in Tokyo?")

for i, request in enumerate(result.context_wrapper.usage.request_usage_entries):
    print(f"Request {i + 1}: {request.input_tokens} in, {request.output_tokens} out")
```

## Accessing usage with sessions

When you use a `Session` (e.g., `SQLiteSession`), each call to `Runner.run(...)` returns usage for that specific run. Sessions maintain conversation history for context, but each run's usage is independent.

```python
session = SQLiteSession("my_conversation")

first = await Runner.run(agent, "Hi!", session=session)
print(first.context_wrapper.usage.total_tokens)  # Usage for first run

second = await Runner.run(agent, "Can you elaborate?", session=session)
print(second.context_wrapper.usage.total_tokens)  # Usage for second run
```

Note that while sessions preserve conversation context between runs, the usage metrics returned by each `Runner.run()` call represent only that particular execution. In sessions, previous messages may be re-fed as input to each run, which affects the input token count in subsequent turns.

## Using usage in hooks

If you're using `RunHooks`, the `context` object passed to each hook contains `usage`. This lets you log usage at key lifecycle moments.

```python
class MyHooks(RunHooks):
    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        u = context.usage
        print(f"{agent.name} → {u.requests} requests, {u.total_tokens} total tokens")
```

## API reference

For detailed API documentation, see:

-   [`Usage`][agents.usage.Usage] - Usage tracking data structure
-   [`RequestUsage`][agents.usage.RequestUsage] - Per-request usage details
-   [`RunContextWrapper`][agents.run.RunContextWrapper] - Access usage from run context
-   [`RunHooks`][agents.run.RunHooks] - Hook into usage tracking lifecycle
