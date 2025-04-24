import Setup
df = Setup.dataset()
print("Setup complete.")
print("DataFrame head:")
print(df.head())

# List of features to use for clustering
X_columns = [
    'num_words',
    'num_unique_words',
    'num_stopwords',
    'num_links',
    'num_unique_domains',
    'num_email_addresses',
    'num_spelling_errors',
    'num_urgent_keywords'
]
Y = 'label'

from sklearn.model_selection import train_test_split

# Get features and target
X = df[X_columns]
y = df[Y]

# Create train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.X_columns = X_columns
        self.trained = False
        
    def train(self, X, y):
        """Train the model with labeled data"""
        from sklearn.cluster import DBSCAN
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        X_features = X[self.X_columns]
        self.labels_ = self.model.fit_predict(X_features)
        self.true_labels = y
        self.trained = True
        return self.labels_

    def fit(self, X):
        """Fit the model to new data"""
        if not self.trained:
            print("Warning: Model hasn't been trained with labeled data")
        X_features = X[self.X_columns]
        self.labels_ = self.model.fit_predict(X_features)
        return self.labels_
    
    def evaluate(self):
        """Evaluate clustering performance if model was trained"""
        if not self.trained:
            return "Model hasn't been trained with labeled data"
        from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
        ari = adjusted_rand_score(self.true_labels, self.labels_)
        ami = adjusted_mutual_info_score(self.true_labels, self.labels_)
        return {"ARI": ari, "AMI": ami}

    def get_clusters(self):
        return set(self.labels_)

# Initialize and train model
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.train(X_train, y_train)
predictions = dbscan.fit(X_test)
performance = dbscan.evaluate()
print("Model Performance:", performance)