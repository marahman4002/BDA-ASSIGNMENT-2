# 🚀 Spark ML Models in Google Colab

This project demonstrates three essential types of Machine Learning models using **Apache Spark** in a **Google Colab** environment. Each model is built with PySpark and includes detailed steps and explanations for learning purposes.

---

## 🎯 Scope

The goal of this project is to explore large-scale machine learning using Spark’s MLlib on real-world datasets.

You will learn how to:
- Configure Spark in Google Colab
- Build a classification model using Logistic Regression
- Perform clustering using KMeans
- Create a recommendation engine using collaborative filtering (ALS)

---

## 🧠 Models Overview

| Model Type       | Algorithm           | Dataset         | Goal                                 |
|------------------|---------------------|------------------|--------------------------------------|
| Classification   | Logistic Regression | Titanic Dataset  | Predict survival of passengers       |
| Clustering       | KMeans              | Iris Dataset     | Group flowers into species-like clusters |
| Recommendation   | ALS                 | MovieLens 100k   | Recommend movies to users            |

---

## 🧰 Workflow

1. **Install Spark + Java** (in Colab using `apt-get` and `wget`)
2. **Configure Spark** using `findspark`
3. **Load & explore dataset** using Pandas and PySpark
4. **Preprocess data** (label encoding, assembling features)
5. **Build ML model** using PySpark MLlib
6. **Train + Evaluate** the model
7. **Explain results** and metrics (e.g. accuracy, silhouette score)

---

## 📦 Datasets Used

| Dataset        | Source                                                                 |
|----------------|------------------------------------------------------------------------|
| Titanic        | [`Seaborn`](https://github.com/mwaskom/seaborn-data) or Kaggle         |
| Iris           | [`Scikit-learn`](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) |
| MovieLens 100k | [GroupLens](https://grouplens.org/datasets/movielens/100k/)             |

---

## ⚙️ Setup in Google Colab

Before running the models, each notebook installs and configures Spark like this:

```bash
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz
!tar -xvzf spark-3.4.1-bin-hadoop3.tgz
!pip install -q findspark
```

## 📘 Notebook Included

- `BSA_ASSIGNMENT_2.ipynb` – A single Colab notebook containing:
  - Logistic Regression on Titanic dataset (Classification)
  - KMeans Clustering on Iris dataset
  - ALS Recommendation on MovieLens dataset

## 📊 Evaluation Metrics

Each model in this notebook is evaluated using an appropriate metric:

- **Classification** (Titanic Dataset)
  - ✅ Accuracy
  - ✅ Precision
  - ✅ Recall

- **Clustering** (Iris Dataset)
  - ✅ Silhouette Score (measures cohesion & separation)

- **Recommendation** (MovieLens Dataset)
  - ✅ Root Mean Squared Error (RMSE)
