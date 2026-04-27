import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class DietTwinFinder:
    def __init__(self, df, metric='cosine', feature_cols=None, weights=None):
        """
        Initialize the DietTwinFinder.
        
        Args:
            df (pd.DataFrame): The dataframe containing diet and lifestyle profiles.
            metric (str): The distance metric to use (e.g., 'cosine', 'euclidean').
            feature_cols (list[str], optional): Explicit list of columns to use as features.
                If None, all numerical columns (excluding identifiers) are used.
            weights (np.ndarray, optional): Per-feature weights applied AFTER StandardScaler.
                This ensures lifestyle flags genuinely dominate over body stats.
                Must have the same length as feature_cols.
        """
        self.df = df.copy()
        
        # Determine which columns to use as features
        if feature_cols is not None:
            self.numerical_cols = list(feature_cols)
        else:
            # Fallback: auto-detect all numerical columns
            self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove identifier columns from feature calculation if they exist
            if 'profile_id' in self.numerical_cols:
                self.numerical_cols.remove('profile_id')
            if 'id' in self.numerical_cols:
                self.numerical_cols.remove('id')
            
        # Handle missing values to prevent algorithm failure (using median imputation)
        self.features_df = self.df[self.numerical_cols].fillna(self.df[self.numerical_cols].median())
        
        # VERY IMPORTANT: Standardize the data.
        # This prevents large-scale features (like weight or calories) from dominating 
        # small-scale features (like age or adherence_score) in the distance metric.
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features_df)
        
        # Apply post-scaling weights so that lifestyle features genuinely carry
        # more mathematical importance than body stats in the distance metric.
        self.weights = weights
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)
            self.scaled_features = self.scaled_features * self.weights
        
        # Initialize and fit the k-NN model.
        self.model = NearestNeighbors(metric=metric, algorithm='brute')
        self.model.fit(self.scaled_features)
        
    def find_twin(self, user_vector, k=1):
        """
        Find the closest match(es) for the given user profile.
        
        Args:
            user_vector (list or np.ndarray): A 1D array of user features matching 
                                              the order of self.numerical_cols.
            k (int): Number of twins to retrieve.
            
        Returns:
            tuple: (indices of the nearest twins, distances/loss to the twins)
        """
        # Ensure the input is a 2D array as expected by sklearn
        user_array = np.array(user_vector).reshape(1, -1)
        
        # Transform the user vector using the previously fitted scaler
        user_scaled = self.scaler.transform(user_array)
        
        # Apply the same weights that were used during fit
        if self.weights is not None:
            user_scaled = user_scaled * self.weights
        
        # Find the k nearest neighbors
        distances, indices = self.model.kneighbors(user_scaled, n_neighbors=k)
        
        # Return 1D arrays for ease of use
        return indices[0], distances[0]
