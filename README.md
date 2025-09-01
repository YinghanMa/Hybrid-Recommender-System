# Hybrid Recommender System  

This project builds a **hybrid recommender system** that predicts user ratings for products by combining collaborative filtering with meta-learning. It demonstrates **data preprocessing, model training, evaluation, and ensembling** to improve recommendation accuracy.  

---

##  Project Overview  
- **Dataset**: Amazon product reviews (user, product, rating, and metadata).  
- **Goal**: Predict user–product ratings in the test set.  
- **Approach**:  
  1. **Exploratory Data Analysis**: Distribution of ratings, user activity, and item popularity.  
  2. **Baseline Models**: Global mean, user mean, and item mean predictors.  
  3. **Matrix Factorisation**: Trained with Surprise SVD to capture latent features.  
  4. **Hybrid Ensembling**: A meta-learner (XGBoost / linear regressor) combines baseline and SVD predictions.  
  5. **Evaluation**: RMSE for rating prediction, using K-Fold cross-validation.  

---

##  Repository Structure  
```text
Hybrid-Recomme
nder-System/
│
├── README.md
│
├── notebooks/
│ └── hybrid_recommender.ipynb # full pipeline: EDA → models → ensemble → submission
│
├── outputs/
│ ├── prediction_result.csv
│
└── data
├── train.csv
└── test.csv
```


---

##  Results & Insights  
- **Baselines**: Simple global/user/item averages give a useful benchmark.  
- **SVD**: Captures hidden patterns in user–item interactions, reducing RMSE significantly.  
- **Hybrid**: The stacked model outperforms individual models, balancing accuracy and generalisation.  
- **Insights**:  
  - Frequent users/items drive recommendation accuracy.  
  - Cold-start cases remain challenging → hybrid helps smooth predictions.  

---

## Requirements
- pandas
- numpy
- scikit-learn
- scikit-surprise
- xgboost
- matplotlib
- seaborn
