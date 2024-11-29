import numpy as np
from scipy.signal import tf2zpk


def is_min_phase(b, a=1.0):
    # Convert transfer function coefficients to zeros, poles, and gain
    z, p, k = tf2zpk(b, a)

    # Check if all zeros are inside the unit circle
    return np.all(np.abs(z) < 1) & np.all(np.abs(p) < 1)


# Example usage
b = [1, -2, 2]  # Coefficients of the numerator
a = [1, 0.5]    # Coefficients of the denominator
result = is_min_phase(b, a)
print(np.abs(np.roots(b)))
print(np.abs(np.roots(a)))
print(f"The system is minimum phase: {result}")
