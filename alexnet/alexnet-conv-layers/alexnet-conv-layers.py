import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """
    AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation).
    """
    k = 11
    p = 2
    s = 4
    filters = 96
    
    (B, H, W, C) = image.shape

    output_size = ((H - k + (2*p)) / s) + 1
    output_shape = np.ndarray(shape=(B, round(output_size), round(output_size), filters))

    return output_shape