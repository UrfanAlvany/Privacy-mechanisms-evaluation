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