import numpy as np
from random import randint

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True, mask: np.ndarray = None) -> np.ndarray:
    """
    Apply inverted dropout. If mask is provided, use it; otherwise generate one.
    """
    if training == True:
        if mask is None:
            mask = [n for n in np.random.choice([0, 1], p=[p, 1-p])]
        
        x = (x / (1-p)) * mask

    return x

    