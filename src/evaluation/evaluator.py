import numpy as np
from sklearn.metrics import mean_squared_error
from src.utils.helpers import ensure_dir

def evaluate_model(model, X_test, y_test, config):
    output_dir = config["paths"]["output_dir"]

    ensure_dir(output_dir + "/metrics")

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)

    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics_text = (
        f"Test Loss (MSE): {test_loss:.4f}\n"
        f"Test MAE: {test_mae:.4f}\n"
        f"Test RMSE: {rmse:.4f}\n"
    )

    with open(f"{output_dir}/metrics/metrics.txt", "w") as f:
        f.write(metrics_text)

    print(metrics_text)