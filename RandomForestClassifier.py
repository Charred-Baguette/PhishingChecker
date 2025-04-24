import Setup
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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

class PhishingRandomForestClassifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.2f}")

    def predict(self, X):
        return self.model.predict(X)

# Create and use the classifier with our prepared data
if __name__ == "__main__":
    classifier = PhishingRandomForestClassifier(X_train, X_test, y_train, y_test)
    classifier.train()
    classifier.evaluate()