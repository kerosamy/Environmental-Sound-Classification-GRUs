import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def evaluate_model(model, X_test, y_test, class_names=None):


    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("Accuracy:", acc)
    print("Macro F1-score:", f1)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        cmap="Blues",
        values_format="d",
        xticks_rotation=45
    )

    plt.title("Confusion Matrix")
    plt.show()

    return acc, f1, cm