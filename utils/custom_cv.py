import numpy as np
from numba import jit


@jit(nopython=True)
def custom_cv(y: np.array, n_folds: int, random_state: int = 10):
    """
    Perform custom cross-validation.

    Returns
    -------
    cv_indices: np.array, counts: np.array
        Indices for each fold and the counts of indices for each fold.
    """
    n_samples = len(y)

    if n_folds < 2:
        raise ValueError('n_folds must be at least 2')
    if n_folds > n_samples:
        raise ValueError(
            'n_folds cannot be greater than the number of samples')

    np.random.seed(random_state)
    cv_indices = np.empty((n_folds, n_samples), dtype=np.int64)
    cv_indices[:] = -1
    counts = np.zeros(n_folds, dtype=np.int64)
    y = y.flatten()
    y_idx = np.argsort(y)
    n_bins = int(np.ceil(n_samples / n_folds))
    indices = np.array_split(y_idx, n_bins)

    # Distribute the indices over the folds
    for i, idx in enumerate(indices):
        shuffled_indices = np.random.permutation(idx)
        fold = i % n_folds
        cv_indices[fold, counts[fold]:counts[fold] +
                   len(shuffled_indices)] = shuffled_indices
        counts[fold] += len(shuffled_indices)

    return cv_indices, counts


class CustomCV:
    def __init__(self, y: np.array, n_folds: int = 10, random_state: int = 10):
        self.y = np.array(y)
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv_indices, self.counts = custom_cv(y, n_folds)

    def get_indices(self):
        """
        Retrieve the indices for each fold in the desired format.

        Returns
        -------
        List of tuples, where each tuple contains the training and test indices for one fold.
        """
        index_pairs = []
        for i in range(self.n_folds):
            train_idx = np.concatenate(
                [self.cv_indices[j, :self.counts[j]] for j in range(self.n_folds) if j != i])
            test_idx = self.cv_indices[i, :self.counts[i]]
            index_pairs.append((train_idx, test_idx))
        return index_pairs
