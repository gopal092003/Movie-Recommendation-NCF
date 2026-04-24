import matplotlib.pyplot as plt
import tensorflow as tf
from src.utils.helpers import ensure_dir

def train_model(model, X_train, y_train, X_test, y_test, config):
    output_dir = config["paths"]["output_dir"]
    model_dir = config["paths"]["model_dir"]

    ensure_dir(output_dir + "/plots")
    ensure_dir(model_dir)

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    patience = config["training"]["patience"]

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(f"{model_dir}/ncf_best_model.h5", save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(f"{output_dir}/plots/loss.png")
    plt.close()

    return model