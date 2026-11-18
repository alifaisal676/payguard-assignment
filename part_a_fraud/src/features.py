"""Feature preparation, fit on train only inside a sklearn Pipeline.

Only Logistic Regression needs preprocessing:
- `Amount` is heavily right-skewed -> log1p, then standardise.
- V1-V28 -> standardise (LR's regularisation assumes comparable scales).
(`Time` is not a model feature; see MODEL_FEATURES in data.py.)

LightGBM gets raw features: tree splits don't care about scale or monotone transforms.

Because the transformer lives inside the model Pipeline, its means/stds are learned from
the training rows only and re-used unchanged on validation/test (no leakage).
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def lr_preprocessor() -> ColumnTransformer:
    log_then_scale = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [("amount", log_then_scale, ["Amount"])],
        remainder=StandardScaler(),          # V1..V28
        verbose_feature_names_out=False,
    )
