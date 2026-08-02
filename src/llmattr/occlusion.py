"""Occlusion attribution: mask each token, measure the drop in score.

score_fn can be anything — a lexicon scorer here, a HuggingFace model's
log-prob in Phase 2. Only score_fn changes; the interface stays the same."""
import numpy as np


class OcclusionAttributor:
    def __init__(self, score_fn, mask_token: str = "[MASK]"):
        self.score_fn = score_fn   # (list[str]) -> float
        self.mask_token = mask_token

    def attribute(self, tokens: list[str]) -> list[dict]:
        base = self.score_fn(tokens)
        out = []
        for i, tok in enumerate(tokens):
            masked = tokens[:i] + [self.mask_token] + tokens[i+1:]
            delta = base - self.score_fn(masked)   # positive => the token supported the output
            out.append({"token": tok, "attribution": round(float(delta), 4)})
        return out

    @staticmethod
    def render_terminal(attributions: list[dict]) -> str:
        """ANSI heatmap: green = supports the output, red = opposes it."""
        vals = np.array([a["attribution"] for a in attributions])
        scale = max(abs(vals).max(), 1e-9)
        parts = []
        for a in attributions:
            x = a["attribution"] / scale
            code = "\033[42m" if x > 0.5 else "\033[102m" if x > 0.1 \
                   else "\033[41m" if x < -0.5 else "\033[101m" if x < -0.1 else "\033[47m"
            parts.append(f"{code}\033[30m {a['token']} \033[0m")
        return " ".join(parts)
