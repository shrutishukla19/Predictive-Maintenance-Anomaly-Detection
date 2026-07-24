"""
Custom preprocessing transformers for the Predictive Maintenance project.

This module contains reusable transformers for:
1. Feature engineering
2. Dropping unnecessary columns

These transformers are used in both the training and inference pipelines
to ensure consistent preprocessing.
"""

from sklearn.base import BaseEstimator, TransformerMixin

COLUMNS_TO_DROP = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

class ColumnDropTransformer(BaseEstimator, TransformerMixin):
    """
    Drops columns that are not required for model training or inference,
    such as identifiers, target, and failure-specific columns.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(
            columns=COLUMNS_TO_DROP, 
            errors="ignore")


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Creates engineered features used by the predictive maintenance model.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["Temperature_Difference"] = (
            X["Process temperature [K]"]
            - X["Air temperature [K]"]
        )

        X["Mechanical_Load"] = (
            X["Torque [Nm]"]
            * X["Rotational speed [rpm]"]
        )

        X["Wear_Stress"] = (
            X["Tool wear [min]"]
            * X["Torque [Nm]"]
        )

        X["Wear_Heat"] = (
            X["Tool wear [min]"]
            * X["Process temperature [K]"]
        )

        return X