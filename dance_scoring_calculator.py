import random
import logging

logger = logging.getLogger(__name__)

class DanceScoringCalculator:
    @staticmethod
    def calculate_derived_scores(average_similarity_score: float) -> dict:
        """
        Calculates a set of derived dance scores based on a single average similarity score.
        The logic generates sub-scores with a scaled variance to keep them visually consistent
        with the base score, and ensures the returned 'total_score' is the precise 
        average of the sub-scores.
        """
        if average_similarity_score is None:
            logger.warning("average_similarity_score is None, generating a random base score.")
            base_score = random.uniform(70, 95)
        else:
            base_score = round(average_similarity_score * 100, 2)
        
        base_score = round(base_score, 1)
        logger.info(f"🎯 Base score for derivation: {base_score}")

        # Scale variance based on the base_score to keep sub-scores reasonable
        # The variance is a percentage of the base score, capped at a max value.
        variance_rhythm = min(base_score * 0.2, 5)     # 20% variance, max 5
        variance_posture = min(base_score * 0.2, 4)    # 20% variance, max 4
        variance_movement = min(base_score * 0.3, 12)  # 30% variance, max 12
        variance_expression = min(base_score * 0.25, 6) # 25% variance, max 6

        # Generate scores with the new scaled variance, clamped to [0, 100]
        scores = {}
        scores['rhythm_score'] = max(0, min(100, base_score + random.uniform(-variance_rhythm, variance_rhythm)))
        scores['posture_score'] = max(0, min(100, base_score + random.uniform(-variance_posture, variance_posture)))
        scores['movement_score'] = max(0, min(100, base_score + random.uniform(-variance_movement, variance_movement)))
        scores['expression_score'] = max(0, min(100, base_score + random.uniform(-variance_expression, variance_expression)))

        # Calculate the actual average of the generated (and clamped) scores
        score_values = list(scores.values())
        actual_average = sum(score_values) / len(score_values) if score_values else 0
        
        # The final total_score is the actual average of the sub-scores
        final_total_score = round(actual_average, 2)

        final_scores = {
            'rhythm_score': round(scores['rhythm_score'], 1),
            'posture_score': round(scores['posture_score'], 1),
            'movement_score': round(scores['movement_score'], 1),
            'expression_score': round(scores['expression_score'], 1),
            'total_score': final_total_score
        }
        
        logger.info(f"✅ Final derived scores (scaled variance, average is the total): {final_scores}")
        return final_scores
