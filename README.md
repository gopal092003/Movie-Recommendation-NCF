# 🎬 Movie Recommendation System (NCF)

A deep learning-based movie recommendation system built using **Neural Collaborative Filtering (NCF)** with TensorFlow.

---

## 🧠 Model Architecture

![Model Architecture](assets/model_architecture.png)

The model learns user-item interactions using embedding layers followed by a deep neural network to predict ratings.

---

## 🚀 Features

* Personalized movie recommendations
* Neural network-based collaborative filtering
* User & movie embeddings
* Modular and scalable pipeline
* Config-driven architecture

---

## 🧠 Tech Stack

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Scikit-learn
* Matplotlib

---

## 📂 Project Structure

```
Movie-Recommendation-NCF/
│
├── data/
├── notebooks/
├── models/
├── outputs/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── utils/
│
├── config.yaml
├── main.py
├── requirements.txt
├── README.md
└── documentation.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/gopal092003/Movie-Recommendation-NCF.git
cd Movie-Recommendation-NCF
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the model

```bash
python main.py --mode train
```

### Evaluate the model

```bash
python main.py --mode evaluate
```

### Get recommendations

```bash
python main.py --mode recommend
```

---

## 📊 Outputs

* Training loss plot → `outputs/plots/`
* Evaluation metrics → `outputs/metrics/`

---

## ⚠️ Limitations

* Cold start problem
* Uses only rating data
* No ranking-based evaluation

---

## 🚀 Future Improvements

* Hybrid recommendation system
* Add metadata (genres, tags)
* Deploy with Streamlit
* Add ranking metrics

---

## 🧾 About

This project demonstrates a **modular deep learning pipeline** for recommendation systems using Neural Collaborative Filtering.

---

## 👤 Author

**Gopal Gupta**
GitHub: https://github.com/gopal092003

---

## ⭐ If you found this useful

Give it a star ⭐ on GitHub!
