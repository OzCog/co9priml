"""
Mathematical utilities for cognitive science components.
Provides basic math functions to replace numpy dependency.
"""

import math
from typing import List, Dict, Any, Union

def clip(value: float, min_val: float, max_val: float) -> float:
    """Clip value to be within min_val and max_val."""
    return max(min_val, min(max_val, value))

def mean(values: List[float]) -> float:
    """Calculate mean of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)

def std(values: List[float]) -> float:
    """Calculate standard deviation of values."""
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)

def normalize(values: List[float]) -> List[float]:
    """Normalize values to sum to 1."""
    total = sum(values)
    if total == 0:
        return [1.0 / len(values)] * len(values) if values else []
    return [v / total for v in values]

def entropy(probabilities: List[float]) -> float:
    """Calculate Shannon entropy of probability distribution."""
    if not probabilities:
        return 0.0
    
    # Normalize to ensure probabilities sum to 1
    total = sum(probabilities)
    if total == 0:
        return 0.0
    
    norm_probs = [p / total for p in probabilities]
    return -sum(p * math.log2(p) for p in norm_probs if p > 0)

def weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate weighted average."""
    if not values or not weights or len(values) != len(weights):
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return mean(values)
    
    return sum(v * w for v, w in zip(values, weights)) / total_weight

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2) or not vec1 or not vec2:
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + math.exp(-clip(x, -500, 500)))

def softmax(values: List[float]) -> List[float]:
    """Softmax activation function."""
    if not values:
        return []
    
    # Subtract max for numerical stability
    max_val = max(values)
    exp_values = [math.exp(clip(v - max_val, -500, 500)) for v in values]
    sum_exp = sum(exp_values)
    
    if sum_exp == 0:
        return [1.0 / len(values)] * len(values)
    
    return [v / sum_exp for v in exp_values]

def confidence_interval(values: List[float], confidence_level: float = 0.95) -> tuple:
    """Calculate confidence interval for values."""
    if len(values) < 2:
        return (0.0, 0.0)
    
    avg = mean(values)
    std_dev = std(values)
    n = len(values)
    
    # Use t-distribution approximation for small samples
    if n < 30:
        t_value = 2.0  # Approximation for 95% confidence
    else:
        t_value = 1.96  # Standard normal for large samples
    
    margin_of_error = t_value * std_dev / math.sqrt(n)
    return (avg - margin_of_error, avg + margin_of_error)

def moving_average(values: List[float], window_size: int) -> List[float]:
    """Calculate moving average with given window size."""
    if window_size <= 0 or window_size > len(values):
        return values.copy()
    
    result = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        end = i + 1
        result.append(mean(values[start:end]))
    
    return result

def correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)
    sum_y2 = sum(yi * yi for yi in y)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator