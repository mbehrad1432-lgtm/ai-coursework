# California Housing data provenance

Acquired with scikit-learn 1.9.1 `fetch_california_housing`, then stored as compressed NumPy arrays `X`, `y` and `feature_names`. No target rounding or feature scaling was applied to this bundled raw dataset. Shape: X=(20640,8), y=(20640,).

Source: 1990 census block-group data; reference Pace, R. Kelley and Ronald Barry, *Sparse Spatial Autoregressions*, Statistics and Probability Letters, 33 (1997), 291–297.

Official description: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/descr/california_housing.rst

Official loading/conversion code: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_california_housing.py

Raw-data archive used by the loader: https://ndownloader.figshare.com/files/5976036

Bundled NPZ SHA-256: `c289ef3e14e1e9f0e550a7e1de781c38b4952b17de77e69b010084de0bce4242`
