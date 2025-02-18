from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

def evaluate_predictions(y_true, y_pred):
    return {"Pearson r": pearsonr(y_true, y_pred), "MSE": mean_squared_error(y_true, y_pred), "MAE": mean_absolute_error(y_true, y_pred)}
