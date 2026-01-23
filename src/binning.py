from enum import StrEnum, auto

import pandas as pd
import numpy as np
from typing import List, Optional

class BinningStrat(StrEnum):
    QUANTILE = auto()
    PREDEFINED = auto()
    UNIFORM = auto()

class BinningTransformer:
    """
    Transforms numerical features into categorical bins.
    Supports quantile, uniform, and predefined binning strategies.
    """

    def __init__(
            self,
            n_bins: int = 4,
            strategy: BinningStrat = BinningStrat.QUANTILE,
            predefined_bins: Optional[List[float]] = None,
            labels: Optional[List[str]] = None
    ):
        # validate input
        if strategy is BinningStrat.PREDEFINED and predefined_bins is None:
            raise ValueError("predefined_bins must be provided for 'predefined' strategy")

        self.n_bins = n_bins
        self.strategy = strategy
        self.predefined_bins = predefined_bins

        self._bin_edges: Optional[np.ndarray] = None
        self._user_labels = labels

    def fit(self, X: pd.Series, y=None):
        """Learns bin edges from data."""
        X = pd.Series(X)

        if self.strategy == BinningStrat.PREDEFINED:
            self._bin_edges = np.array(self.predefined_bins)

        elif self.strategy == BinningStrat.QUANTILE:
            _, self._bin_edges = pd.qcut(X, q=self.n_bins, retbins=True, duplicates='drop')

        elif self.strategy == BinningStrat.UNIFORM:
            _, self._bin_edges = pd.cut(X, bins=self.n_bins, retbins=True, include_lowest=True)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """Applies binning using learned edges."""
        if self._bin_edges is None:
            raise RuntimeError("Transformer has not been fit yet.")

        X = pd.Series(X)
        binned = pd.cut(
            X,
            bins=self._bin_edges,
            include_lowest=True,
            duplicates='drop'
        )
        final_labels = self._generate_labels(binned.cat.categories)
        binned = binned.cat.rename_categories(final_labels)
        return binned.astype(str)

    def fit_transform(self, X: pd.Series, y=None) -> pd.Series:
        return self.fit(X).transform(X)

    def _generate_labels(self, categories: pd.Index) -> List[str]:
        """
        Generates labels matching the actual categories found.
        If user provided custom labels, we try to use them, but ensure length matches.
        """
        # If user provided exact labels and lengths match, use them
        if self._user_labels is not None:
            if len(self._user_labels) == len(categories):
                return self._user_labels
            print("User labels does not match number of categories, generating new labels")

        # Otherwise, generate simplified 'Int-Int' labels derived from the Interval objects
        generated_labels = []
        for i, interval in enumerate(categories):
            low = int(interval.left)
            high = int(interval.right)
            if i < len(categories) - 1:
                label = f"{low}-{high - 1}"
            else:
                label = f"{low}-{high}"
            generated_labels.append(label)

        return generated_labels