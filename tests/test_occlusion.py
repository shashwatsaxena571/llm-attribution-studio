import sys; sys.path.insert(0, "src")
from llmattr import OcclusionAttributor

def test_important_token_gets_high_attribution():
    score = lambda toks: 2.0 if "love" in toks else 0.0
    res = OcclusionAttributor(score).attribute(["i", "love", "it"])
    by_tok = {r["token"]: r["attribution"] for r in res}
    assert by_tok["love"] == 2.0 and by_tok["i"] == 0.0
