import sqlite3
import pandas as pd
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import re
import time


# Load and preprocess data (same as before)
def load_and_preprocess_data():
    # Load data from SQLite database
    conn = sqlite3.connect('synthetic_university.db')
    students = pd.read_sql_query('SELECT * FROM Students', conn)
    courses = pd.read_sql_query('SELECT * FROM Courses', conn)
    enrollments = pd.read_sql_query('SELECT * FROM Enrollments', conn)
    conn.close()

    # Combine the data for modeling
    data = pd.merge(enrollments, students, on='Student_ID').merge(courses, on='Course_ID')

    # One-hot encode categorical variables
    data = pd.get_dummies(data, columns=['Gender', 'Department_x', 'Department_y', 'Grade'])

    # Ensure all feature columns are numeric and convert to float32
    X = data.drop(columns=['Enrollment_ID', 'Student_ID', 'Course_ID', 'Name', 'Instructor', 'Course_Name', 'GPA'])
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    X = X.astype('float32')

    # Normalize the feature data
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Target variable
    y = data['GPA'].astype('float32')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, X

# Function to extract epsilon value from the statement string
def extract_epsilon(statement):
    print(f"Privacy Statement: {statement}")  # Debug print statement
    match = re.search(r'Epsilon with each example occurring once per epoch:\s+([0-9.]+)', statement)
    if match:
        return float(match.group(1))
    else:
        return None

# Function to create and train model with DPSGD
def train_dpsgd_model(epsilon, delta, epochs, learning_rate, X_train, y_train, X_test, y_test):
    noise_multiplier = 0.1 / epsilon  # Adjusted noise multiplier
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=1.0,
        noise_multiplier=noise_multiplier,
        num_microbatches=1,
        learning_rate=learning_rate  # Adjusted learning rate
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Linear output for regression
    ])

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    # Compute epsilon value
    n = len(X_train)
    statement = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy_statement(
        n,
        32,
        epochs,
        noise_multiplier,
        delta,
        False,
        None,
        compute_dp_sgd_privacy_lib.AccountantType.RDP,
    )
    epsilon_value = extract_epsilon(statement)

    return model, history, loss, mae, epsilon_value

# Function to perform membership inference attack
def membership_inference_attack(model, X_train, X_test, y_train, y_test):
    # Predict values for train and test data
    train_preds = model.predict(X_train).flatten()  # Ensure 1D array
    test_preds = model.predict(X_test).flatten()  # Ensure 1D array

    # Compute confidence scores for membership inference
    train_confidence = np.abs(train_preds - y_train)
    test_confidence = np.abs(test_preds - y_test)

    # Determine a threshold based on the median of train confidence
    threshold = np.median(train_confidence)

    # Predict membership
    train_membership_pred = train_confidence <= threshold
    test_membership_pred = test_confidence <= threshold

    # True membership (1 if the sample is in the training set, 0 otherwise)
    train_true_membership = np.ones(len(y_train))
    test_true_membership = np.zeros(len(y_test))

    # Combine predictions and true membership labels
    membership_pred = np.concatenate([train_membership_pred, test_membership_pred])
    true_membership = np.concatenate([train_true_membership, test_true_membership])

    # Compute attack accuracy
    attack_accuracy = np.mean(membership_pred == true_membership)

    return attack_accuracy

# Function to perform model inversion attack
def model_inversion_attack(model, X_test, sensitive_feature_indices):
    # Predict values for test data
    test_preds = model.predict(X_test).flatten()  # Ensure 1D array

    # Extract the sensitive features from the input data
    sensitive_test = X_test.iloc[:, sensitive_feature_indices]

    # Initialize a list to store inversion accuracies for each sensitive feature
    inversion_accuracies = []

    # Loop over each sensitive feature column
    for i, col in enumerate(sensitive_test.columns):
        # Predict sensitive feature based on the model predictions
        reconstructed_sensitive_feature = np.sign(test_preds)

        # Compute inversion accuracy for the current feature
        actual_sensitive_feature = np.sign(sensitive_test[col])
        inversion_accuracy = np.mean(reconstructed_sensitive_feature == actual_sensitive_feature)

        # Store the inversion accuracy
        inversion_accuracies.append(inversion_accuracy)

    # Compute the average inversion accuracy across all sensitive features
    average_inversion_accuracy = np.mean(inversion_accuracies)

    return average_inversion_accuracy


# Function to evaluate performance
def evaluate_performance(epsilon, delta, epochs, learning_rate, batch_size, X_train, y_train, X_test, y_test,
                         sensitive_feature_indices):
    start_time = time.time()
    model, history, loss, mae, epsilon_value = train_dpsgd_model(epsilon, delta, epochs, learning_rate, X_train,
                                                                 y_train, X_test, y_test)
    end_time = time.time()
    computation_time = end_time - start_time

    # Perform membership inference attack
    membership_attack_accuracy = membership_inference_attack(model, X_train, X_test, y_train, y_test)

    # Perform model inversion attack
    inversion_attack_accuracy = model_inversion_attack(model, X_test, sensitive_feature_indices)

    return loss, mae, epsilon_value, computation_time, membership_attack_accuracy, inversion_attack_accuracy

# Main experiment
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X = load_and_preprocess_data()
    epsilons = [0.5]  # Different epsilon values for experimentation
    deltas = [1e-6]  # Different delta values for experimentation
    epochs_list = [100]
    learning_rates = [0.01]  # Different learning rates for experimentation
    batch_sizes = [8,16,32,64,128]  # Different batch sizes for experimentation

    # Identify the indices of the one-hot encoded 'Grade' columns
    grade_columns = [col for col in X.columns if col.startswith('Grade_')]
    sensitive_feature_indices = [X.columns.get_loc(col) for col in grade_columns]

    # Debug print statements to check the columns and their indices
    print(f"Columns: {list(X.columns)}")
    print(f"Grade Columns: {grade_columns}")
    print(f"Sensitive Feature Indices: {sensitive_feature_indices}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    results = []

    for epsilon in epsilons:
        for delta in deltas:
            for epochs in epochs_list:
                for learning_rate in learning_rates:
                    for batch_size in batch_sizes:
                        loss, mae, epsilon_value, computation_time, membership_attack_accuracy, inversion_attack_accuracy = evaluate_performance(
                            epsilon, delta, epochs, learning_rate, batch_size, X_train, y_train, X_test, y_test,
                            sensitive_feature_indices)
                        results.append((
                            epsilon, delta, epochs, learning_rate, batch_size, loss, mae, epsilon_value,
                            computation_time, membership_attack_accuracy, inversion_attack_accuracy))
                        print(
                            f"Epsilon: {epsilon}, Delta: {delta}, Epochs: {epochs}, Learning Rate: {learning_rate}, Batch Size: {batch_size}, Loss: {loss}, MAE: {mae}, Epsilon Value: {epsilon_value}, Time: {computation_time}, Membership Attack Accuracy: {membership_attack_accuracy}, Inversion Attack Accuracy: {inversion_attack_accuracy}")

    # Convert results to DataFrame for better visualization
    results_df = pd.DataFrame(results,
                              columns=['Epsilon', 'Delta', 'Epochs', 'Learning Rate', 'Batch Size', 'Loss', 'MAE',
                                       'Epsilon Value', 'Computation Time', 'Membership Attack Accuracy',
                                       'Inversion Attack Accuracy'])

    # Save results
    results_df.to_csv('dpsgd_delta06_results.csv', index=False)

    # Plot results
    plt.figure(figsize=(12, 8))
    for epsilon in epsilons:
        for delta in deltas:
            for learning_rate in learning_rates:
                filtered_results = results_df[(results_df['Epsilon'] == epsilon) & (results_df['Delta'] == delta) & (
                        results_df['Learning Rate'] == learning_rate)]
                plt.plot(filtered_results['Epochs'], filtered_results['MAE'], marker='o',
                         label=f'Epsilon {epsilon}, Delta {delta}, LR {learning_rate}')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Effect of Epochs, Epsilon, Delta, and Learning Rate on MAE')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 8))
    for epsilon in epsilons:
        for delta in deltas:
            for learning_rate in learning_rates:
                filtered_results = results_df[(results_df['Epsilon'] == epsilon) & (results_df['Delta'] == delta) & (
                        results_df['Learning Rate'] == learning_rate)]
                plt.plot(filtered_results['Epochs'], filtered_results['Computation Time'], marker='o',
                         label=f'Epsilon {epsilon}, Delta {delta}, LR {learning_rate}')
    plt.xlabel('Epochs')
    plt.ylabel('Computation Time (s)')
    plt.title('Effect of Epochs, Epsilon, Delta, and Learning Rate on Computation Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 8))
    for epsilon in epsilons:
        for delta in deltas:
            for learning_rate in learning_rates:
                filtered_results = results_df[(results_df['Epsilon'] == epsilon) & (results_df['Delta'] == delta) & (
                        results_df['Learning Rate'] == learning_rate)]
                plt.plot(filtered_results['Epochs'], filtered_results['Membership Attack Accuracy'], marker='o',
                         label=f'Epsilon {epsilon}, Delta {delta}, LR {learning_rate}')
    plt.xlabel('Epochs')
    plt.ylabel('Membership Attack Accuracy')
    plt.title('Effect of Epochs, Epsilon, Delta, and Learning Rate on Membership Inference Attack Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 8))
    for epsilon in epsilons:
        for delta in deltas:
            for learning_rate in learning_rates:
                filtered_results = results_df[(results_df['Epsilon'] == epsilon) & (results_df['Delta'] == delta) & (
                        results_df['Learning Rate'] == learning_rate)]
                plt.plot(filtered_results['Epochs'], filtered_results['Inversion Attack Accuracy'], marker='o',
                         label=f'Epsilon {epsilon}, Delta {delta}, LR {learning_rate}')
    plt.xlabel('Epochs')
    plt.ylabel('Inversion Attack Accuracy')
    plt.title('Effect of Epochs, Epsilon, Delta, and Learning Rate on Model Inversion Attack Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
