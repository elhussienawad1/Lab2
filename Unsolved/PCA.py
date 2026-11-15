import numpy as np

class PCA():
    def __init__(self,  new_dim:int, solver:str='eig') -> None:
        # hyperparameter representing the number of dimensions after reduction
        #   - int  L      → keep the top L principal components
        #   - float in (0,1) → [Extra] keep the fewest components whose variance ratio is at least new_dim
        self.new_dim = new_dim
        # [Extra] 'eig' = eigendecomposition of Σ (the TODO path), 'svd' = SVD of Z/sqrt(m-1)
        assert solver in ['eig', 'svd'], f"solver must be 'eig' or 'svd' but you passed {solver}"
        self.solver = solver
        # for standardization
        self.μ:np.ndarray
        self.σ:np.ndarray
        # for PCA
        self.A:np.ndarray 
        self.λs:np.ndarray                  # sorted eigenvalues (= variances of the new features)
        self.explained_variance_ratio:np.ndarray

    # x_train is (m,n) matrix where each row is an n-dimensional vector of features
    def fit(self, x_train):
        # TODO 1: Find μ and σ of each feature in x_train
        self.μ = np.mean(x_train, axis=0)
        self.σ = np.std(x_train, axis=0)
        # if a column has zero std (useless constant) set σ=1 (skip their standardization)
        self.σ = np.where(self.σ == 0, 1, self.σ)
        
        # TODO 2: Standardize the training data
        z_train = (x_train - self.μ) / self.σ

        if self.solver == 'eig':
            # TODO 3: Compute the covariance matrix
            Σ = np.cov(z_train, rowvar=False)           # (n, n); rowvar=False → columns are the variables
            
            # TODO 4: Compute eigenvalues and eigenvectors using Numpy
            λs, U = np.linalg.eig(Σ)                    # U[:, i] is the eigenvector of λs[i]
            λs, U = λs.real, U.real           # sometimes a zero imaginary part can appear due to approximations
            
            # TODO 5: Sort eigenvalues and eigenvectors
            # TODO 5.1: Find the sequence of indices that sort λs in descending order
            sorting_inds = np.argsort(λs)[::-1]
            # TODO 5.2: Use it to sort λs and U
            λs = λs[sorting_inds]
            U = U[:, sorting_inds]                      # reorder the COLUMNS (eigenvectors), not the rows
        else:
            # [Extra] SVD route: Σ = ZᵀZ/(m-1), so right singular vectors of Z/sqrt(m-1) are the
            # eigenvectors of Σ and the squared singular values are its eigenvalues (already descending)
            m = z_train.shape[0]
            _, S, Vt = np.linalg.svd(z_train / np.sqrt(m - 1), full_matrices=False)
            λs, U = S ** 2, Vt.T

        self.λs = λs
        self.explained_variance_ratio = λs / np.sum(λs)
        
        # TODO 6: Select the top L eigenvectors and set A accordingly
        L = self.new_dim
        if isinstance(L, float) and 0 < L < 1:
            # [Extra] smallest L such that cumulative variance ratio ≥ new_dim
            L = int(np.searchsorted(np.cumsum(self.explained_variance_ratio), L) + 1)
        self.L = L
        self.A = U[:, :L].T                             # (L, n): rows are the top-L principal components
        
        return self
    
    # x_val is (m,n) matrix where each row is an n-dimensional vector of features
    def transform(self, x_val):
        z_val = (x_val - self.μ) / self.σ
        # TODO 7: Apply the transformation equation
        # Z_(m×L) = X_(m×n) · Aᵀ_(n×L)
        return z_val @ self.A.T
    
    def inverse_transform(self, z_val):
        # TODO 8: Apply the inverse transformation equation (including destandardization)
        # X_(m×n) ≈ Z_(m×L) · A_(L×n), then undo standardization: x = z·σ + μ
        return (z_val @ self.A) * self.σ + self.μ

    def fit_transform(self, x_train):
        return self.fit(x_train).transform(x_train)
