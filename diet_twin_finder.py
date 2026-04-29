import numpy as np
import pandas as pd


class DietTwinFinder:
    """
    k-Nearest Neighbor Diet Twin retrieval.

    Implemented from scratch using NumPy only — no sklearn NearestNeighbors.

    Pipeline (mirrors the SVD & Vector Retrieval lecture):
        1. Standardise  : z = (x - μ) / σ          (numpy, per-feature)
        2. Re-weight    : z' = z * w                (amplify behavioural features)
        3. Retrieve     : rank by Cosine / Euclidean / Manhattan distance

    Cosine Similarity derivation (from lecture):
        cos(u, v) = ⟨u, v⟩ / (‖u‖ · ‖v‖)
        Cosine Distance = 1 - cos(u, v)

    Choosing argmin over the dataset is equivalent to the top-k retrieval
    formula from the lecture:
        r_k = argmin_{v ∈ D} dist(v, q)
    """

    def __init__(self, df, metric="cosine", feature_cols=None, weights=None):
        """
        Args:
            df           : DataFrame of diet/lifestyle profiles.
            metric       : 'cosine' | 'euclidean' | 'manhattan'
            feature_cols : Ordered list of feature column names.
            weights      : Per-feature multipliers applied after standardisation.
                           Lets behavioural features (adherence, stress …) outweigh
                           body-stat features in the distance metric.
        """
        self.df = df.copy()
        self.metric = metric

        # ── 1. Feature selection ─────────────────────────────────────────────
        if feature_cols is not None:
            self.numerical_cols = list(feature_cols)
        else:
            self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for drop_col in ("profile_id", "id"):
                if drop_col in self.numerical_cols:
                    self.numerical_cols.remove(drop_col)

        raw = self.df[self.numerical_cols].fillna(
            self.df[self.numerical_cols].median()
        ).to_numpy(dtype=float)

        # ── 2. Manual StandardScaler  z = (x - μ) / σ ───────────────────────
        self._mean = raw.mean(axis=0)               # shape (d,)
        self._std  = raw.std(axis=0)                # shape (d,)
        self._std[self._std == 0] = 1.0             # avoid division by zero

        self.scaled_features = (raw - self._mean) / self._std   # (N, d)

        # ── 3. Post-scaling behavioural weights  z' = z * w ─────────────────
        self.weights = None
        if weights is not None:
            self.weights = np.asarray(weights, dtype=float)
            self.scaled_features = self.scaled_features * self.weights

        # ── 4. Pre-compute L2 norms for cosine search (speed optimisation) ──
        norms = np.linalg.norm(self.scaled_features, axis=1, keepdims=True)  # (N,1)
        norms[norms == 0] = 1.0
        self._unit_features = self.scaled_features / norms     # (N, d), unit vectors

    # ── Private distance kernels ─────────────────────────────────────────────

    def _cosine_distances(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Cosine Distance = 1 - ⟨u, v⟩ / (‖u‖ · ‖v‖)

        Because both the dataset matrix and the query are already unit-
        normalised, the inner product equals the cosine similarity directly:
            dist = 1 - (unit_features @ unit_query)
        This avoids an O(N·d) norm recomputation on every query.
        """
        norm = np.linalg.norm(query_vec)
        unit_q = query_vec / (norm if norm > 0 else 1.0)
        # Matrix-vector dot product: shape (N,)
        similarities = self._unit_features @ unit_q
        return 1.0 - similarities

    def _euclidean_distances(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Squared Euclidean: ‖v - q‖² = ‖v‖² + ‖q‖² - 2⟨v, q⟩  (from lecture)
        Return √ for interpretable distance values.
        """
        diff = self.scaled_features - query_vec      # broadcasting (N, d)
        return np.sqrt((diff ** 2).sum(axis=1))      # (N,)

    def _manhattan_distances(self, query_vec: np.ndarray) -> np.ndarray:
        """L1 distance: Σ |vᵢ - qᵢ|"""
        return np.abs(self.scaled_features - query_vec).sum(axis=1)  # (N,)

    # ── Public API ───────────────────────────────────────────────────────────

    def find_twin(self, user_vector, k: int = 1):
        """
        Top-k nearest-neighbour retrieval.

        Implementation follows the lecture formula:
            r_k = argmin_{v ∈ D} dist(v, q)    for k = 1 … K

        Args:
            user_vector : 1-D array of raw feature values (same order as
                          feature_cols, BEFORE scaling).
            k           : Number of twins to return.

        Returns:
            indices   : np.ndarray of shape (k,) — dataset row indices
            distances : np.ndarray of shape (k,) — corresponding distances
        """
        # ── Apply the same z-score transform fitted on the dataset ───────────
        user_arr = np.asarray(user_vector, dtype=float)
        user_scaled = (user_arr - self._mean) / self._std

        # ── Apply the same behavioural weights ───────────────────────────────
        if self.weights is not None:
            user_scaled = user_scaled * self.weights

        # ── Compute pairwise distances (O(N·d)) ──────────────────────────────
        if self.metric == "cosine":
            dists = self._cosine_distances(user_scaled)
        elif self.metric == "euclidean":
            dists = self._euclidean_distances(user_scaled)
        elif self.metric == "manhattan":
            dists = self._manhattan_distances(user_scaled)
        else:
            raise ValueError(f"Unsupported metric: '{self.metric}'. "
                             f"Choose from 'cosine', 'euclidean', 'manhattan'.")

        # ── Partial-sort: O(N + k·log k) instead of O(N·log N) ──────────────
        k = min(k, len(dists))
        top_k_idx = np.argpartition(dists, k)[:k]          # unordered top-k
        top_k_idx = top_k_idx[np.argsort(dists[top_k_idx])]  # sort by dist

        return top_k_idx, dists[top_k_idx]
