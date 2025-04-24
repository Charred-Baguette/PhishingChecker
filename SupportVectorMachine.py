import Setup
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.utils import resample

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

# Balance the dataset
df_majority = pd.DataFrame(X[y == 0])
df_minority = pd.DataFrame(X[y == 1])

# Undersample majority class
df_majority_undersampled = resample(df_majority,
                                  replace=False,
                                  n_samples=len(df_minority),
                                  random_state=42)

# Combine minority class with undersampled majority class
X_balanced = pd.concat([df_majority_undersampled, df_minority])
y_balanced = pd.concat([pd.Series([0] * len(df_majority_undersampled)), 
                       pd.Series([1] * len(df_minority))])

# Create train-test split with balanced data
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

from sklearn.svm import SVC

# Create and train SVM classifier
svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Print results
print("\nSupport Vector Machine Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

