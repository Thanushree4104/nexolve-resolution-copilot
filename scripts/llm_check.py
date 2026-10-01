from app.llm.factory import get_provider

provider = get_provider()
resp = provider.complete(
    [{"role": "user", "content": "Reply with one short sentence about broadband."}],
    max_tokens=500,
)
print(resp)