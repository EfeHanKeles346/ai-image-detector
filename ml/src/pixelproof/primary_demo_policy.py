"""E92 paired decision with a separate, non-vetoing E43 advisory.

Weights, preprocessing and both cutoffs are inherited unchanged. The historical
demo_policy/demo_scores adapters remain available for reproducing old reports.
"""
from pixelproof.demo_scores import ScoredDemoEngine, scored_display_result
from pixelproof.e92_demo import AI_CUT, guarded_outcome, social_view
from pixelproof.verified_demo_runtime import VerifiedE92Engine

DISPLAY_POLICY = 'e92-primary-reference-advisory-v2'
GUARD_ID = 'e92-paired-v2'


def primary_result(scores, reference):
    result = scored_display_result(scores, reference)  # validates all four scores
    guarded, reason = guarded_outcome(scores, [0., 0.])
    result.update(outcome='ai_signal' if scores[0] >= AI_CUT else guarded,
                  reason=reason, guard_outcome=guarded, review_required=guarded == 'uncertain',
                  display_policy=DISPLAY_POLICY,
                  reference_ai_warning=bool(max(reference) >= AI_CUT))
    return result


class PrimaryDemoEngine(ScoredDemoEngine):
    def __init__(self):
        self.engine = VerifiedE92Engine()

    def analyze(self, image):
        scores, reference = self.engine.score_views([image, social_view(image)])
        return primary_result(scores, reference)
