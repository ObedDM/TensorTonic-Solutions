import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    length = np.prod(x.shape, axis=0)
    
    if rng is None:
        random = np.random.random(length)

    else:
        random = rng.random(length)

    mask = np.where(random > 1-p, 0.0, 1.0)
    mask = mask.reshape(x.shape)

    x = x * mask / (1-p)

    return (x, (mask / (1-p)))

    