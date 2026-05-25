from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_model(y_true, y_pred):

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0,1,2]
    )

    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    # =========================
    # ACCURACY
    # =========================
    acc = accuracy_score(y_true, y_pred)

    print(f"\nAccuracy : {acc:.2f}")

    # =========================
    # PRECISION
    # =========================
    precision = precision_score(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0
    )

    print(f"Precision: {precision:.2f}")

    # =========================
    # RECALL
    # =========================
    recall = recall_score(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0
    )

    print(f"Recall   : {recall:.2f}")

    # =========================
    # CLASSIFICATION REPORT
    # =========================
    print("\n=== CLASSIFICATION REPORT ===")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=[
                "AMAN",
                "WASPADA",
                "BAHAYA"
            ],
            zero_division=0
        )
    )

    # =========================
    # VISUALISASI MATRIX
    # =========================
    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[
            'AMAN',
            'WASPADA',
            'BAHAYA'
        ],
        yticklabels=[
            'AMAN',
            'WASPADA',
            'BAHAYA'
        ]
    )

    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix")

    plt.show()