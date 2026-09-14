"""Presentation-only raw scores; frozen E92 inference and alert policy are unchanged."""
from pixelproof.demo_policy import display_result
from pixelproof.e92_demo import AI_CUT, E92Engine, social_view


def scored_display_result(scores, reference):
    result = display_result(scores, reference)  # validates both paired arrays
    result['model_score'] = {
        'kind': 'raw_e92_score',
        'calibrated': False,
        'original': float(scores[0]),
        'social_q75': float(scores[1]),
        'ai_cut': AI_CUT,
    }
    return result


class ScoredDemoEngine:
    def __init__(self):
        self.engine = E92Engine()

    def analyze(self, image):
        scores, reference = self.engine.score_views([image, social_view(image)])
        return scored_display_result(scores, reference)
