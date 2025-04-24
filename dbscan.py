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

# Clean and validate data
X = df[X_columns].dropna()  # Remove rows with missing values
y = df[Y][X.index]  # Keep only corresponding labels

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
        X = X.fillna(X.mean())  # Handle any remaining missing values
        self.labels_ = self.model.fit_predict(X)
        self.true_labels = y
        self.trained = True
        return self.labels_

    def fit(self, X):
        """Fit the model to new data"""
        if not self.trained:
            print("Warning: Model hasn't been trained with labeled data")
        X = X.fillna(X.mean())  # Handle any missing values
        self.model.fit(X)  # Only fit, don't predict
        return self
    
    def predict(self, X):
        """Predict clusters for new data using the fitted model"""
        X = X.fillna(X.mean())
        self.test_labels_ = self.model.fit_predict(X)  # Store predictions
        return self.test_labels_
    
    def evaluate(self):
        """Evaluate clustering performance with detailed metrics"""
        if not self.trained:
            return "Model hasn't been trained with labeled data"
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, adjusted_rand_score, adjusted_mutual_info_score,
            classification_report
        )
        
        # Calculate clustering specific metrics
        ari = adjusted_rand_score(y_test, self.test_labels_)
        ami = adjusted_mutual_info_score(y_test, self.test_labels_)
        
        # Calculate classification metrics with 'macro' averaging
        accuracy = accuracy_score(y_test, self.test_labels_)
        precision = precision_score(y_test, self.test_labels_, average='macro')
        recall = recall_score(y_test, self.test_labels_, average='macro')
        f1 = f1_score(y_test, self.test_labels_, average='macro')
        
        # Get detailed classification report
        report = classification_report(y_test, self.test_labels_, output_dict=True)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, self.test_labels_)
        
        return {
            "Clustering Metrics": {
                "ARI": f"{float(ari):.2%}",
                "AMI": f"{float(ami):.2%}",
            },
            "Classification Metrics": {
                "Accuracy": f"{accuracy:.2%}",
                "Macro Precision": f"{precision:.2%}",
                "Macro Recall": f"{recall:.2%}",
                "Macro F1-Score": f"{f1:.2%}",
            },
            "Detailed Report": report,
            "Confusion Matrix": cm.tolist(),
            "Number of Clusters": len(set(self.test_labels_))
        }

    def get_clusters(self):
        return set(self.labels_)

# Initialize and train model
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.train(X_train, y_train)
dbscan.fit(X_test)  # Fit the model to test data
predictions = dbscan.predict(X_test)  # Get predictions
performance = dbscan.evaluate()
print("Model Performance:", performance)