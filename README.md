# 🎥 Shoplifting Detection 

This computer vision system detects theft incidents in video footage. Using deep learning models, it analyzes video sequences to identify suspicious activities and potential stealing behavior. The solution provides real-time monitoring and alert capabilities for security applications.

---

## 🧠 Project Overview

- **Goal:** Classify surveillance videos into:
  - `shop_lifter`
  - `non_shop_lifter`
- **Dataset:** > 800 videos total, each with 16 frames (224x224, RGB)
- **Model:** Pretrained `EfficientNetB0_LSTM` 
- **Strategy:**
  - Extract embeddings from each video using frozen R3D-18.
  - Train a small binary classifier (MLP) or logistic regression on top.
  - Evaluate performance using accuracy, precision, recall, and F1-score.

---

## 🧾 Evaluation Metrics

The following **metrics** are used to evaluate model performance:

- Accuracy

- Precision

- Recall

- F1 Score
