import numpy as np
from pysindy.optimizers import BaseOptimizer


class RecursiveLeastSquaresOptimizer(BaseOptimizer):
    """
    Online Recursive Least Squares optimizer for PySINDy.
    Updates coefficients incrementally with forgetting factor.
    """
    
    def __init__(self, lam=0.99, alpha=100.0, threshold=0.0, unbias=True):
        """
        Parameters
        ----------
        lam : float
            Forgetting factor in (0,1]; smaller means faster adaptation.
        alpha : float
            Initial covariance scaling (larger = more uncertainty initially).
        threshold : float
            Optional threshold for small coefficients (to preserve sparsity).
        unbias : bool
            Whether to perform unbiasing (PySINDy convention).
        """
        super().__init__(unbias=unbias)
        self.lam = lam
        self.alpha = alpha
        self.threshold = threshold
        self.P = None
        self.initialized_ = False
    
    def _reduce(self, x, y):
        """
        Required by PySINDy BaseOptimizer API.
        For RLS, no reduction is needed - just pass through.
        
        Parameters
        ----------
        x : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        """
        # No reduction needed for RLS
        pass
    
    def _fit(self, x, y):
        """
        Internal fit method called by BaseOptimizer.fit().
        Initializes and performs RLS updates on batch data.
        
        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features)
            Feature matrix
        y : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target values
        """
        n_samples, n_features = x.shape
        
        # Handle 1D target
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_targets = y.shape[1]
        
        # Initialize covariance matrix and coefficients
        self.P = np.eye(n_features) * self.alpha
        self.coef_ = np.zeros((n_features, n_targets))
        self.initialized_ = True
        
        # Perform RLS updates for each sample
        for i in range(n_samples):
            self.update(x[i], y[i])
        
        # Apply threshold for sparsity if needed
        if self.threshold > 0:
            self.coef_[np.abs(self.coef_) < self.threshold] = 0.0
    
    def update(self, phi_t, y_t):
        """
        Perform one recursive least squares update.
        
        Parameters
        ----------
        phi_t : np.ndarray, shape (n_features,)
            Feature vector at time t
        y_t : float or np.ndarray, shape (n_targets,)
            Target value(s) at time t
            
        Returns
        -------
        self : RecursiveLeastSquaresOptimizer
        """
        # Auto-initialize if not yet fitted (useful for ensemble copies)
        if not self.initialized_ or self.P is None or self.coef_ is None:
            # Infer dimensions from input
            phi_t_arr = np.atleast_1d(phi_t)
            y_t_arr = np.atleast_1d(y_t)
            
            n_features = len(phi_t_arr)
            n_targets = len(y_t_arr)
            
            self.P = np.eye(n_features) * self.alpha
            self.coef_ = np.zeros((n_features, n_targets))
            self.initialized_ = True
        
        # Ensure proper shapes and convert to plain numpy arrays
        phi_t = np.atleast_1d(phi_t)
        if hasattr(phi_t, 'values'):
            phi_t = phi_t.values
        phi_t = np.asarray(phi_t).reshape(-1, 1)  # (n_features, 1)
        
        y_t = np.atleast_1d(y_t)
        if hasattr(y_t, 'values'):
            y_t = y_t.values
        y_t = np.asarray(y_t).reshape(1, -1)      # (1, n_targets)
        
        # RLS update equations - convert P and coef to plain numpy
        lam = self.lam
        P = np.asarray(self.P)
        coef = np.asarray(self.coef_)
        
        # Calculate gain
        P_phi = P @ phi_t
        denom = lam + float((phi_t.T @ P @ phi_t)[0, 0])
        K = P_phi / denom            # (n_features, 1)
        
        # Calculate prediction error
        err = y_t - phi_t.T @ coef   # (1, n_targets)
        
        # Update coefficients
        self.coef_ = coef + K @ err  # (n_features, n_targets)
        
        # Update covariance matrix (Joseph form for numerical stability)
        self.P = (P - K @ phi_t.T @ P) / lam
        
        # Apply threshold
        if self.threshold > 0:
            self.coef_[np.abs(self.coef_) < self.threshold] = 0.0
        
        return self
    
    @property
    def complexity(self):
        """
        Compute complexity as number of nonzero coefficients.
        Required by PySINDy BaseOptimizer.
        """
        return np.count_nonzero(self.coef_)