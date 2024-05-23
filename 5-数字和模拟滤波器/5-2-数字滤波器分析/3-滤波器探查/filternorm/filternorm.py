import numpy as np
from scipy.signal import freqz, butter,firwin
from scipy.linalg import norm
from impz import impz
from impzlength import impzlength

def filternorm(b, a, pnorm):
    # Check number of input arguments
    if not (2 <= len(locals()) <= 4):
        raise ValueError("Wrong number of input arguments. Expected 2 to 4.")

    # Set pnorm to 2 if only two input arguments are provided
    if len(locals()) == 2:
        pnorm = 2

    # Validate input data types and values
    if not isinstance(b, np.ndarray) or not isinstance(a, np.ndarray):
        raise TypeError("Inputs 'b' and 'a' must be numpy arrays.")
    if not np.issubdtype(b.dtype, np.float64) or not np.issubdtype(a.dtype, np.float64):
        raise TypeError("Inputs 'b' and 'a' must be of data type float64.")

    # Validate pnorm
    pnormScalar = pnorm
    if not (np.isinf(pnormScalar) or pnormScalar == 2):
        raise ValueError("Invalid value for pnorm. Must be either Inf or 2.")

    # Convert b and a to row vectors
    bRow = np.squeeze(b)
    aRow = a

    # Check for zero-leading denominator coefficient
    # if aRow[0] == 0:
    #     raise ValueError("Zero-leading denominator coefficient is not allowed.")

    # Check stability and FIR/IIR
    isStable, isFIR = isstableTF(bRow, aRow)
    if not isStable:
        raise ValueError("Filter is unstable.")

    if np.isinf(pnormScalar):
        # Compute inf-norm (maximum magnitude of frequency response)
        _, h = freqz(bRow, aRow, worN=1024)
        s = np.max(np.abs(h))
    else:  # pnormScalar == 2
        if isFIR:
            # Compute 2-norm for FIR filter
            s = norm(bRow, ord=-2)
        else:
            # Compute 2-norm for IIR filter
            tol = 1e-8  # Default tolerance for IIR filter
            maxradius = np.max(np.abs(np.roots(aRow)))
            # Check stability using maximum radius of poles
            if maxradius >= 1:
                raise ValueError("Filter is unstable (numerical precision issue).")

            # Determine length of impulse response
            N = impzlength(bRow, aRow, tol)
            H = impz(bRow, aRow, N)
            s = norm(H, ord=-2)

    return s

def isstableTF(num, den):
    # Determine if filter is FIR or IIR
    firFlag = np.isscalar(den) or len(den) == 1
    if firFlag:
        # FIR filter is always stable
        return True, True
    else:
        # Check stability of IIR filter
        return np.all(np.abs(np.roots(den)) < 1), False


# # Example usage:
# b = np.array([1, -0.5, 0.2])  # Numerator coefficients of filter
# a = np.array([1, -0.8, 0.3])   # Denominator coefficients of filter
# pnorm = 2                      # P-norm (default: 2)
# norm_value = filternorm(b, a, pnorm)
# print(f"Norm value: {norm_value}")