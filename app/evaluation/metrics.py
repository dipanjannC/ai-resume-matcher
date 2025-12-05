"""Evaluation metrics for LangChain agents"""

from typing import Dict, List, Any, Set


def field_accuracy(predicted: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Calculate field-level accuracy between predicted and expected outputs.
    
    Args:
        predicted: Predicted output dictionary
        expected: Expected output dictionary
        
    Returns:
        Accuracy score between 0 and 1
    """
    if not expected:
        return 1.0
    
    correct = 0
    total = 0
    
    for key, expected_value in expected.items():
        total += 1
        predicted_value = predicted.get(key)
        
        # Handle nested dictionaries
        if isinstance(expected_value, dict) and isinstance(predicted_value, dict):
            correct += field_accuracy(predicted_value, expected_value)
        # Handle lists
        elif isinstance(expected_value, list) and isinstance(predicted_value, list):
            if set(expected_value) == set(predicted_value):
                correct += 1
        # Direct comparison
        elif predicted_value == expected_value:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def skill_extraction_recall(predicted_skills: List[str], expected_skills: List[str]) -> float:
    """
    Calculate recall for skill extraction.
    Recall = (True Positives) / (True Positives + False Negatives)
    
    Args:
        predicted_skills: List of predicted skills
        expected_skills: List of expected skills
        
    Returns:
        Recall score between 0 and 1
    """
    if not expected_skills:
        return 1.0
    
    predicted_set = set(s.lower().strip() for s in predicted_skills)
    expected_set = set(s.lower().strip() for s in expected_skills)
    
    true_positives = len(predicted_set.intersection(expected_set))
    
    return true_positives / len(expected_set)


def skill_extraction_precision(predicted_skills: List[str], expected_skills: List[str]) -> float:
    """
    Calculate precision for skill extraction.
    Precision = (True Positives) / (True Positives + False Positives)
    
    Args:
        predicted_skills: List of predicted skills
        expected_skills: List of expected skills
        
    Returns:
        Precision score between 0 and 1
    """
    if not predicted_skills:
        return 0.0
    
    predicted_set = set(s.lower().strip() for s in predicted_skills)
    expected_set = set(s.lower().strip() for s in expected_skills)
    
    true_positives = len(predicted_set.intersection(expected_set))
    
    return true_positives / len(predicted_set)


def completeness_score(parsed_data: Dict[str, Any], required_fields: List[str]) -> float:
    """
    Calculate completeness score based on presence of required fields.
    
    Args:
        parsed_data: Parsed resume data
        required_fields: List of required field paths (e.g., "profile.name")
        
    Returns:
        Completeness score between 0 and 1
    """
    if not required_fields:
        return 1.0
    
    present = 0
    
    for field_path in required_fields:
        parts = field_path.split('.')
        current = parsed_data
        
        try:
            for part in parts:
                current = current.get(part) if isinstance(current, dict) else getattr(current, part, None)
            
            # Check if field has meaningful value
            if current is not None and current != "" and current != []:
                present += 1
        except (AttributeError, KeyError):
            continue
    
    return present / len(required_fields)


def f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        F1 score between 0 and 1
    """
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)
