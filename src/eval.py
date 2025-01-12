

from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf


def evaluate_model(model_path, X_val, y_val):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    

    # make the predictions
    predictions = model.predict(X_val)
    
    # calculate the error metrics
    mse = mean_squared_error(y_val, predictions)
    mae = mean_absolute_error(y_val, predictions)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
