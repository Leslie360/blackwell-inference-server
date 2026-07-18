"""Self-speculative decoding with n-gram draft proposals.

Implements the "prompt lookup" / self-speculative decoding method: use n-gram
matches from the current context as draft tokens, then verify them with the
target model in a single forward pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class SpecResult:
    text: str
    tokens: int
    wall_time_s: float
    tokens_per_s: float
    draft_acceptance_rate: float


class NGramProposer:
    def __init__(self, n: int = 3, gamma: int = 4):
        self.n = n
        self.gamma = gamma

    def propose(self, context: list[int], next_token: int) -> list[int]:
        """Return draft tokens after next_token, using the most recent n-gram match.

        The token immediately following the matched n-gram must equal next_token;
        otherwise the draft is invalid.
        """
        if len(context) < self.n + 1:
            return []
        key = context[-self.n:]
        for i in range(len(context) - self.n - 1, -1, -1):
            if context[i : i + self.n] == key and context[i + self.n] == next_token:
                return context[i + self.n + 1 : i + self.n + 1 + self.gamma]
        return []


def _truncate_cache(past_key_values, length: int):
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(length)
        return past_key_values
    for i in range(len(past_key_values.key_cache)):
        past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, :length, :]
        past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, :length, :]
    return past_key_values


def speculative_generate(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    max_new_tokens: int = 128,
    gamma: int = 4,
    ngram: int = 3,
    eos_token_id: int | None = None,
) -> SpecResult:
    """Greedy decoding with n-gram self-speculation."""
    device = input_ids.device
    proposer = NGramProposer(n=ngram, gamma=gamma)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1)  # pending, not appended yet
        generated = input_ids.clone()
        new_tokens = 0
        total_draft = 0
        accepted_draft = 0

        while new_tokens < max_new_tokens:
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            context = generated[0].tolist()
            draft = proposer.propose(context, next_token.item())

            if not draft:
                # greedy: verify next_token only
                candidate = next_token.unsqueeze(1)
                out = model(input_ids=candidate, past_key_values=past, use_cache=True)
                past = out.past_key_values
                generated = torch.cat([generated, candidate], dim=1)
                next_token = out.logits[:, -1, :].argmax(dim=-1)
                new_tokens += 1
                continue

            # verify next_token + draft in one forward
            draft_tensor = torch.tensor([draft], device=device, dtype=generated.dtype)
            candidate = torch.cat([next_token.unsqueeze(1), draft_tensor], dim=1)
            total_draft += len(draft)

            out = model(input_ids=candidate, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0]

            # next_token is always accepted (it was model's greedy choice)
            accepted = 0
            for j in range(candidate.shape[1] - 1):
                if logits[j].argmax() == candidate[0, j + 1]:
                    accepted += 1
                else:
                    break
            accepted_draft += accepted

            if accepted < candidate.shape[1] - 1:
                add = accepted + 1
                if new_tokens + add > max_new_tokens:
                    add = max_new_tokens - new_tokens
                    generated = torch.cat([generated, candidate[:, :add]], dim=1)
                    new_tokens += add
                    break
                keep = past.get_seq_length() - (candidate.shape[1] - 1 - accepted)
                past = _truncate_cache(past, keep)
                generated = torch.cat([generated, candidate[:, : accepted + 1]], dim=1)
                next_token = logits[accepted].argmax().unsqueeze(0)
                new_tokens += add
            else:
                add = candidate.shape[1]
                if new_tokens + add > max_new_tokens:
                    add = max_new_tokens - new_tokens
                    generated = torch.cat([generated, candidate[:, :add]], dim=1)
                    new_tokens += add
                    break
                generated = torch.cat([generated, candidate], dim=1)
                next_token = logits[-1].argmax().unsqueeze(0)
                new_tokens += add

        if eos_token_id is not None and next_token.item() == eos_token_id and new_tokens < max_new_tokens:
            generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            new_tokens += 1

    wall = time.perf_counter() - t0
    text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
    return SpecResult(
        text=text,
        tokens=new_tokens,
        wall_time_s=wall,
        tokens_per_s=new_tokens / wall,
        draft_acceptance_rate=accepted_draft / total_draft if total_draft else 0.0,
    )


def greedy_generate(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    max_new_tokens: int = 128,
    eos_token_id: int | None = None,
) -> SpecResult:
    """Standard greedy decoding for baseline."""
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1)
        generated = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
        new_tokens = 1

        while new_tokens < max_new_tokens:
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            out = model(input_ids=next_token.unsqueeze(1), past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            new_tokens += 1

    wall = time.perf_counter() - t0
    text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
    return SpecResult(
        text=text,
        tokens=new_tokens,
        wall_time_s=wall,
        tokens_per_s=new_tokens / wall,
        draft_acceptance_rate=0.0,
    )
