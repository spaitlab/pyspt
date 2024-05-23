import math


def besself(x, n):
    """
    Compute the Bessel function of the first kind of order n for the given value x.

    Parameters:
    x (float): The value at which to compute the Bessel function.
    n (int): The order of the Bessel function.

    Returns:
    float: The computed value of the Bessel function.
    """
    return math.besselfn(n, x)