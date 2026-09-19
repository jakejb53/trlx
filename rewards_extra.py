"""Verification fixture for [rewards] path.py:function entries."""


# 1 for completions under 20 words, else 0.
def brevity(completions, **kwargs):
    from trlx.rewards import completion_text

    return [1.0 if len(completion_text(c).split()) < 20 else 0.0 for c in completions]
