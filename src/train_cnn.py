"""Keras CNN on VV, VH and ratio channels."""
from pathlib import Path


def build_model(input_shape, n_classes):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    def block(x, f):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        return layers.ReLU()(x)

    inp = layers.Input(input_shape)
    x = layers.RandomFlip("horizontal_and_vertical")(inp)
    x = block(x, 32); x = block(x, 32); x = layers.MaxPooling2D()(x)
    x = block(x, 64); x = block(x, 64); x = layers.MaxPooling2D()(x)
    x = block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def run_cnn(Xc, y, tr, va, te, class_names, out_dir, epochs=40, batch_size=32, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score)

    tf.keras.utils.set_random_seed(seed)
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    model = build_model(Xc.shape[1:], len(class_names))
    cb = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
          tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)]
    hist = model.fit(Xc[tr], y[tr], validation_data=(Xc[va], y[va]),
                     epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=2)
    pred = model.predict(Xc[te], verbose=0).argmax(1)
    res = {"test_accuracy": float(accuracy_score(y[te], pred)),
           "test_macro_f1": float(f1_score(y[te], pred, labels=sorted(set(y[te].tolist())), average="macro", zero_division=0)),
           "epochs_run": len(hist.history["loss"])}

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(hist.history["accuracy"], label="train"); ax[1].plot(hist.history["val_accuracy"], label="val")
    ax[1].set_title("accuracy"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "06_cnn_training_curves.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y[te], pred, labels=list(range(len(class_names))), display_labels=class_names, ax=ax,
                                            xticks_rotation=45, colorbar=False)
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "05_cnn_confusion.png", dpi=130); plt.close(fig)
    model.save(out_dir / "models" / "cnn.keras")
    return res
