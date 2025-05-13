

import pandas as pd
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    classification_report,
)




if __name__ == "__main__":
    mode = "lwc"

    column = ['drug_name', 'profile', 'justification', 'summary', 'risk_score',
              'rag_summary', 'rag_recommendation_text', 'rag_Risk_score',
              'lwc_summary', 'lwc_recommendation_text', 'lwc_Risk_score',
              'profile_id', 'retrieved_chunk']



    df = pd.read_csv('/home/tom/Bureau/phd/mistral_training/hackaton_medi/result_baseline/res_lwc_mistral-small-latest_2025-05-12 19:20:40.csv')
    print(df.columns)

    if mode == "rag":
        list_pred = list(df["rag_Risk_score"])
        list_gr = list(df["risk_score"])
    elif mode == "lwc":
        list_pred = list(df["lwc_Risk_score"])
        list_gr = list(df["risk_score"])
    else:
        raise ValueError("Invalid mode. Choose either 'rag' or 'lcw'.")
    list_pred_cleaned = []
    list_ground_truth = []
    list_not_cleaned = []
    for i in range(len(list_pred)):
        if type(list_pred[i]) == float and np.isnan(list_pred[i]):
            list_not_cleaned.append(list_pred[i])
            continue
        if type(list_pred[i]) == float:
            list_pred[i] = str(list_pred[i])
        if list_pred[i] == 0 or list_pred[i][0] == '0' or list_pred[i][0] == 0:
            list_pred_cleaned.append(0)
            list_ground_truth.append(list_gr[i])
        elif list_pred[i] == 1 or list_pred[i][0] == '1' or list_pred[i][0] == 1:
            list_pred_cleaned.append(1)
            list_ground_truth.append(list_gr[i])
        elif list_pred[i] == 1 or list_pred[i][0] == '2' or list_pred[i][0] == 2:
            list_pred_cleaned.append(2)
            list_ground_truth.append(list_gr[i])
        else:
            list_not_cleaned.append(list_pred[i])

    print(f"number of cleaned: {len(list_pred_cleaned)}")
    print(f"number of not cleaned: {len(list_not_cleaned)}")
    acc = accuracy_score(list_ground_truth, list_pred_cleaned)
    print(f"Accuracy: {acc:.3f}")




    mode = "rag_hybride"

    per_class_acc = recall_score(y_true=list_ground_truth, y_pred=list_pred_cleaned,
                                 labels=[0, 1, 2], average=None)  # one value per class

    for cls, acc in zip([0, 1, 2], per_class_acc):
        print(f"Accuracy for class '{cls}': {acc:.3f}")

    classes = [0, 1, 2]
    cm = confusion_matrix(y_true=list_ground_truth, y_pred=list_pred_cleaned, labels=classes)
    per_class_acc_manual = cm.diagonal() / cm.sum(axis=1)
    assert np.allclose(per_class_acc, per_class_acc_manual)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from matplotlib import pyplot as plt
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[0, 1, 2])
    disp.plot()
    ## save a svg
    plt.savefig(f"/home/tom/Bureau/phd/mistral_training/hackaton_medi/result_baseline/confusion_matrix_{mode}.svg",
                bbox_inches='tight', dpi=300)
    plt.show()

    # 4⃣  Nicest single call — includes precision, recall (= per‑class acc), f1
    print("\nFull classification report:\n")
    print(classification_report(y_true=list_ground_truth, y_pred=list_pred_cleaned, digits=3))




