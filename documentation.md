# 📌 Technical Documentation: Movie Recommendation System (NCF)

---

## 🧠 1. Introduction

This project implements a **Neural Collaborative Filtering (NCF)** based recommendation system using deep learning techniques to model user-item interactions.

---

## 🏗️ 2. Model Architecture

![Model Architecture](assets/model_architecture.png)

### Explanation

The architecture consists of:

1. **User Embedding Layer**

   * Converts user IDs into dense vectors

2. **Movie Embedding Layer**

   * Converts movie IDs into latent representations

3. **Concatenation Layer**

   * Combines user and movie embeddings

4. **Fully Connected Layers**

   * Learns non-linear interaction patterns

5. **Output Layer**

   * Sigmoid activation to predict normalized ratings

---

## 🔁 3. System Pipeline

```
Raw Data → Preprocessing → Encoding → Model → Training → Evaluation → Recommendation
```

---

## 📂 4. Module Breakdown

### 4.1 Data Layer (`src/data/`)

* Loads and preprocesses ratings dataset
* Encodes users and movies

---

### 4.2 Model Layer (`src/models/`)

* Defines Neural Collaborative Filtering model

---

### 4.3 Training Layer (`src/training/`)

* Handles model training and checkpointing

---

### 4.4 Evaluation Layer (`src/evaluation/`)

* Computes MSE, MAE, RMSE

---

### 4.5 Inference Layer (`src/inference/`)

* Generates top-N recommendations

---

### 4.6 Utility Layer (`src/utils/`)

* Config handling and helper functions

---

## ⚙️ 5. Configuration

Controlled via `config.yaml`:

* Data paths
* Model parameters
* Training settings
* Output paths

---

## 🧪 6. Training Workflow

1. Load dataset
2. Preprocess and encode
3. Split user-wise
4. Train model
5. Save best model
6. Generate plots

---

## 📊 7. Evaluation Metrics

* **MSE**
* **MAE**
* **RMSE**

---

## 🎯 8. Recommendation Logic

* Identify unseen movies
* Predict ratings
* Rank and return top-N

---

## ⚠️ 9. Limitations

* Cold start problem
* No metadata usage
* No ranking-based metrics

---

## 🚀 10. Future Enhancements

* Hybrid recommendation system
* Add content-based features
* Deploy API/UI
* Add ranking metrics

---

## 🧾 11. Conclusion

This project demonstrates a **modular, scalable, and production-style implementation** of a Neural Collaborative Filtering recommendation system.
