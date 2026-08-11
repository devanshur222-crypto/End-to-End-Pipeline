import pandas as pd
import numpy as np
import logging
import os
import pickle
import json

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# Logging configuration
# --------------------------------------------------

logdir = 'log'
os.makedirs(logdir, exist_ok=True)

logger = logging.getLogger('Model_evaluation')
logger.setLevel(logging.DEBUG)

console_log = logging.StreamHandler()
console_log.setLevel(logging.DEBUG)

log_file_path = os.path.join(
    logdir,
    'Model_evaluation.log'
)

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

console_log.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_log)
logger.addHandler(file_handler)


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model(model_path):

    try:

        logger.debug('Model loading started.')

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        logger.debug('Model loaded successfully.')

        return model

    except Exception as e:

        logger.error(
            f'Unexpected error occurred while loading model: {e}'
        )

        raise


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

def evaluate_model(model, features, target):

    try:

        logger.debug('Model evaluation started.')

        # Make predictions
        predictions = model.predict(features)

        # Calculate metrics
        mae = mean_absolute_error(
            target,
            predictions
        )

        mse = mean_squared_error(
            target,
            predictions
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            target,
            predictions
        )

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }

        logger.debug(
            f'Model evaluation completed. Metrics: {metrics}'
        )

        return metrics

    except Exception as e:

        logger.error(
            f'Unexpected error occurred during model evaluation: {e}'
        )

        raise


# --------------------------------------------------
# Save metrics
# --------------------------------------------------

def save_metrics(metrics, metrics_path):

    try:

        logger.debug('Saving evaluation metrics.')

        # Create reports directory
        os.makedirs(
            os.path.dirname(metrics_path),
            exist_ok=True
        )

        # Save metrics as JSON
        with open(metrics_path, 'w') as file:
            json.dump(
                metrics,
                file,
                indent=4
            )

        logger.debug(
            f'Metrics saved successfully at {metrics_path}'
        )

    except Exception as e:

        logger.error(
            f'Unexpected error occurred while saving metrics: {e}'
        )

        raise


# --------------------------------------------------
# Main function
# --------------------------------------------------

def main():

    try:

        logger.debug('Model evaluation process started.')

        # --------------------------------------------------
        # Paths
        # --------------------------------------------------

        model_path = 'models/model.pkl'

        test_data_path = (
            r'C:\Users\devan\OneDrive\Desktop'
            r'\End-to-End-Pipeline\data\interim\test_data.csv'
        )

        metrics_path = 'reports/metrics.json'


        # --------------------------------------------------
        # Load model
        # --------------------------------------------------

        model = load_model(model_path)


        # --------------------------------------------------
        # Load test data
        # --------------------------------------------------

        test_data = pd.read_csv(test_data_path)

        logger.debug('Test data loaded successfully.')


        # --------------------------------------------------
        # Separate features and target
        # --------------------------------------------------

        target = test_data['selling_price']

        features = test_data.drop(
            columns='selling_price'
        )


        # --------------------------------------------------
        # Evaluate model
        # --------------------------------------------------

        metrics = evaluate_model(
            model,
            features,
            target
        )


        # --------------------------------------------------
        # Save metrics
        # --------------------------------------------------

        save_metrics(
            metrics,
            metrics_path
        )


        logger.debug(
            'Model evaluation process completed successfully.'
        )


    except Exception as e:

        logger.error(
            f'Unable to complete model evaluation: {e}'
        )

        raise


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == '__main__':
    main()