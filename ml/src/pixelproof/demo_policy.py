"""Presentation policy preserving every original E92 AI alert.

The separately measured E93 stability guard remains unchanged. Its disagreement
may annotate an original alert but may never suppress that alert in this demo.
"""
from pixelproof.e92_demo import AI_CUT, guarded_outcome

DISPLAY_POLICY = 'e92-preserve-alerts-v1'


def display_result(scores, reference):
    guarded, reason = guarded_outcome(scores, reference)  # validates all inputs
    outcome = 'ai_signal' if scores[0] >= AI_CUT else guarded
    return {'outcome': outcome, 'reason': reason, 'guard_outcome': guarded,
            'review_required': guarded == 'uncertain', 'display_policy': DISPLAY_POLICY}


class DemoEngine:
    def __init__(self):
        from pixelproof.e92_demo import E92Engine
        self.engine = E92Engine()

    def analyze(self, image):
        from pixelproof.e92_demo import social_view
        scores, reference = self.engine.score_views([image, social_view(image)])
        return display_result(scores, reference)
