import math

# Pricing (OpenAI embedding model) - https://developers.openai.com/api/docs/models/text-embedding-3-small
COST_PER_1K_TOKENS = 0.00002  # text-embedding-3-small

def estimate_tokens(text: str) -> int:
    # Approx: 1 token ≈ 4 chars
    return math.ceil(len(text) / 4)

def calculate_cost(chunks):
    total_tokens = 0

    for chunk in chunks:
        tokens = estimate_tokens(chunk.page_content)
        total_tokens += tokens

    cost = (total_tokens / 1000) * COST_PER_1K_TOKENS

    return {
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "estimated_cost": round(cost, 5),
        "avg_tokens_per_chunk": total_tokens // max(len(chunks), 1)
    }