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

# Create MLP Neural Network
from sklearn.neural_network import MLPClassifier

# Define different architectures to test
hidden_layers = [
    (50,),
    (100,),
    (200,),
    (50, 25),
    (100, 50),
    (200, 100),
    (100, 50, 25),
    (200, 100, 50),
    (300, 200, 100),
    (200, 150, 100, 50),
    (300, 200, 150, 100, 50),
    (400, 300, 200, 100, 50, 25),
    (500, 400, 300, 200, 100, 50, 25),
    (600, 500, 400, 300, 200, 100, 50, 25),
    (700, 600, 500, 400, 300, 200, 100, 50, 25),
    (800, 700, 600, 500, 400, 300, 200, 100, 50, 25),
]

# Dictionary to store results
results = {}

# Test each architecture
for layers in hidden_layers:
    print(f"\nTesting architecture with layers: {layers}")
    
    mlp = MLPClassifier(hidden_layer_sizes=layers,
                        max_iter=500,
                        random_state=42)
    
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    
    acc_score = accuracy_score(y_test, y_pred)
    results[layers] = {
        'model': mlp,
        'accuracy': acc_score,
        'predictions': y_pred
    }
    
    print(f"Architecture {layers}:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {acc_score}\n")

# Find best model
best_layers = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
best_accuracy = results[best_layers]['accuracy']

print(f"\nBest Architecture: {best_layers}")
print(f"Best Accuracy: {best_accuracy}")

# Save best model for future use
best_model = results[best_layers]['model']