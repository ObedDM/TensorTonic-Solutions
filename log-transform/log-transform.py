def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    result = [np.log1p(v) for v in values]

    return result    