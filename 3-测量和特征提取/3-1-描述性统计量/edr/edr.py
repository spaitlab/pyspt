import numpy as np

def edr(seq1, seq2, tolerance):
    """
    Calculates the Edit Distance on Real sequence (EDR) between two sequences.
    This function assumes that the sequences are lists of real numbers.
    
    :param seq1: First sequence of real numbers
    :param seq2: Second sequence of real numbers
    :param tolerance: Tolerance within which matches are considered
    :return: EDR distance and the aligned sequences
    """
    len_seq1 = len(seq1)
    len_seq2 = len(seq2)
    # Create a 2D matrix of size (len_seq1+1) x (len_seq2+1) for dynamic programming
    dp = np.zeros((len_seq1 + 1, len_seq2 + 1))

    # Initialize the first row and column of the matrix
    for i in range(len_seq1 + 1):
        dp[i][0] = i
    for j in range(len_seq2 + 1):
        dp[0][j] = j

    # Populate the matrix based on the EDR distance rule
    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            if abs(seq1[i - 1] - seq2[j - 1]) <= tolerance:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # Insertion
                                    dp[i][j - 1],  # Deletion
                                    dp[i - 1][j - 1])  # Match/Mismatch

    # Reconstruct the aligned sequences
    aligned_seq1 = []
    aligned_seq2 = []
    i, j = len_seq1, len_seq2
    while i > 0 and j > 0:
        if abs(seq1[i - 1] - seq2[j - 1]) <= tolerance:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(0)
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            aligned_seq1.append(0)
            aligned_seq2.append(seq2[j - 1])
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1

    # Finish the remaining sequence if any
    while i > 0:
        aligned_seq1.append(seq1[i - 1])
        aligned_seq2.append(0)
        i -= 1
    while j > 0:
        aligned_seq1.append(0)
        aligned_seq2.append(seq2[j - 1])
        j -= 1

    aligned_seq1.reverse()
    aligned_seq2.reverse()

    return dp[-1][-1], aligned_seq1, aligned_seq2