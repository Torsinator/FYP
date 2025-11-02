import pysindy as ps
import numpy as np


dt = 0.02
feature_names = [r'$x$', r'$y$', r'vx', r'$vy$', r'$\theta$', r'\omega']


# 1. Library ensemble and calculate inclusion probabilities
n_candidates_to_drop = 1
n_models = 10
library_ensemble_optimizer = ps.STLSQ()
model = ps.SINDy(feature_names=feature_names,
                 optimizer=library_ensemble_optimizer)
model.fit(x_train, t=dt, library_ensemble=True, n_models=n_models,
          n_candidates_to_drop=n_candidates_to_drop, quiet=True)
model.print()
library_ensemble_coefs = np.asarray(model.coef_list)
n_targets = len(feature_names)
n_features = len(model.get_feature_names())

# Calculate inclusion probabilities (note you may want to add another factor here because
# each term is only present for a portion of the fits)
inclusion_probabilities = np.count_nonzero(model.coef_list, axis=0) / n_models

# 2. Chop inclusion probabilities <= 50% (this is rather drastic for illustration)
inclusion_probabilities[inclusion_probabilities <= 0.3] = 0.0

# Find indices that are chopped for all three equations
# since we pass the same library for all.
chopped_inds = np.any(inclusion_probabilities != 0.0, axis=0)
chopped_inds = np.ravel(np.where(~chopped_inds))

# 3. Pass truncated library and then do normal ensembling
library = ps.PolynomialLibrary(degree=2, library_ensemble=True,
                               ensemble_indices=chopped_inds)
ensemble_optimizer = ps.STLSQ()
model = ps.SINDy(
    feature_names=feature_names,
    optimizer=ensemble_optimizer,
    feature_library=library
)
model.fit(x_train, t=dt, ensemble=True, n_models=n_models, quiet=True)
two_step_ensemble_coefs = np.asarray(model.coef_list)
two_step_mean = np.mean(two_step_ensemble_coefs, axis=0)
two_step_std = np.std(two_step_ensemble_coefs, axis=0)
two_step_median = np.median(two_step_ensemble_coefs, axis=0)

# Add zeros to get coefficient matrices to original full size
for i in range(len(chopped_inds)):
    two_step_mean = np.insert(two_step_mean, chopped_inds[i], 0.0, axis=-1)
    two_step_std = np.insert(two_step_std, chopped_inds[i], 0.0, axis=-1)
    two_step_median = np.insert(two_step_median, chopped_inds[i], 0.0, axis=-1)