import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf
from train_dpsgd_model import load_and_preprocess_data  # Import the data loading function

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Load the DPSGD results
results_df = pd.read_csv('dpsgd_results.csv')


# Function to create attack data
def prepare_attack_data(model, X_train, X_test):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_labels = np.ones(len(train_preds))
    test_labels = np.zeros(len(test_preds))

    preds = np.concatenate([train_preds, test_preds])
    labels = np.concatenate([train_labels, test_labels])

    return preds, labels


# Function to train and evaluate attack model
def train_attack_model(preds, labels):
    attack_model = LogisticRegression()
    attack_model.fit(preds.reshape(-1, 1), labels)

    preds_labels = attack_model.predict(preds.reshape(-1, 1))
    accuracy = accuracy_score(labels, preds_labels)
    precision = precision_score(labels, preds_labels)
    recall = recall_score(labels, preds_labels)

    return accuracy, precision, recall


# Run membership inference attack
attack_results = []

for _, row in results_df.iterrows():
    epsilon = row['Epsilon']
    epochs = int(row['Epochs'])  # Ensure epochs is an integer

    # Load the model with the corresponding epsilon and epochs
    model_filename = f'model_epsilon_{epsilon}_epochs_{epochs}.h5'
    model = tf.keras.models.load_model(model_filename, compile=False)

    # Prepare attack data
    preds, labels = prepare_attack_data(model, X_train, X_test)

    # Train attack model
    accuracy, precision, recall = train_attack_model(preds, labels)

    attack_results.append((epsilon, epochs, accuracy, precision, recall))
    print(
        f"Epsilon: {epsilon}, Epochs: {epochs}, Attack Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

# Save attack results
attack_results_df = pd.DataFrame(attack_results, columns=['Epsilon', 'Epochs', 'Accuracy', 'Precision', 'Recall'])
attack_results_df.to_csv('attack_results.csv', index=False)

print("Attack simulation completed and results saved.")
