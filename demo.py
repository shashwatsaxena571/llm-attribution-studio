"""Demo: occlusion attribution over a sentiment lexicon scorer.

In Phase 2, score_fn becomes a HuggingFace model's target-token log-prob —
the rest of this file does not change."""
import sys; sys.path.insert(0, "src")
from llmattr import OcclusionAttributor

POS = {"amazing": 2, "love": 2, "great": 1.5, "good": 1}
NEG = {"terrible": -2, "hate": -2, "slow": -1, "buggy": -1.5}

def sentiment_score(tokens):
    return sum(POS.get(t.lower(), 0) + NEG.get(t.lower(), 0) for t in tokens)

sentence = "I love this phone but the software is buggy and slow".split()
attr = OcclusionAttributor(sentiment_score)
result = attr.attribute(sentence)

print("Token attributions (how much each token pushed the sentiment prediction):\n")
for r in sorted(result, key=lambda x: -abs(x["attribution"])):
    bar = "#" * int(abs(r["attribution"]) * 5)
    sign = "+" if r["attribution"] >= 0 else "-"
    print(f"  {r['token']:>10}  {sign}{abs(r['attribution']):.2f}  {bar}")

print("\nHeatmap view:")
print(attr.render_terminal(result))
print("\nNext: replace score_fn with GPT-2's log P(next_token) — that is the only change needed.")
