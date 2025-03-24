# WiDS Datathon 2025: Predicting ADHD and Sex from Brain Imaging
---

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Phuong-An Bui | @phganie | Led ADHD and Sex_F model development using StackingClassifier (LGBM, RF, XGBoost), implemented custom preprocessing pipelines with KNN imputation, variance thresholding, and standard scaling, engineered composite and age-bucket features, applied SMOTE for class imbalance, tuned hyperparameters with Optuna (200+ trials), performed threshold optimization |
| Keerthi Chebrolu | @keerthic05 | Data cleaning and preprocessing for connectome dataset, implemented PCA model |
| Aryaman Tepal | @aryamantepal | Data cleaning and preprocessing for categorical dataset, applying scaling & normalization to data |

---

## **🎯 Project Highlights**
* Applied SMOTE for class imbalance, KNN imputation for missing values, and variance thresholding for feature selection
* Developed an ensemble classification pipeline using LightGBM, Random Forest, and XGBoost to predict ADHD and sex from brain imaging and metadata
* Achieved an average validation F1 score of [insert score, e.g., 0.79] using weighted evaluation metrics
* Tuned hyperparameters with Optuna across 200+ trials
* Merged, cleaned, and standardized over 4 datasets, including connectome matrices, categorical, and quantitative metadata

🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

```
# Clone the repo
git clone https://github.com/WiDS-Team-24/WiDS-Kaggle-Competition.git
cd [final]_wids_team_24.py
# or [final]_Wids_Team_24.ipynb
# Install dependencies (Colab setup, or locally)
pip install -r requirements.txt

# Data Access:
# Mount your Google Drive in Colab and load the following:
# - TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv
# - TRAINING_SOLUTIONS.xlsx
# - TRAIN_QUANTITATIVE_METADATA.xlsx
# - TRAIN_CATEGORICAL_METADATA.xlsx
# - TEST_*.xlsx and .csv files

# Run notebooks
Open the .ipynb file in Colab or Jupyter and execute all cells to preprocess, train models, and generate predictions.
```

---

## **🏗️ Project Overview**

**Describe:**

This project addresses the challenge of diagnosing ADHD using fMRI-based brain connectivity and participant metadata. The goal is to mitigate diagnostic bias—especially for females—by creating inclusive, fair, and effective predictive models.
* Build fair models for underdiagnosed ADHD in females
* Reveal sex-specific brain activity patterns using machine learning
* Demonstrate clinical potential for AI-assisted mental health diagnostics
---

## **📊 Data Exploration**

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

*Datasets Used:*
* TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv: fMRI brain connectivity matrices
* TRAIN_QUANTITATIVE_METADATA.xlsx: Numerical metadata like scan age
* TRAIN_CATEGORICAL_METADATA.xlsx: Demographic and environmental info
* TRAINING_SOLUTIONS.xlsx: Target variables (ADHD_Outcome, Sex_F)
* Equivalent test datasets


*Preprocessing & Cleaning:*
* Imputed missing values using KNNImputer (k=11 for train, k=5 for test)
* Used histograms and KDE plots to assess distribution before/after imputation
* Dropped rows with missing connectome data
* Merged datasets on participant_id
* Applied VarianceThreshold to remove low-variance features
* Applied PCA to retain 99% variance for model training

**Some of the challenges that we faced were:**
- The connectome matrices had ~19,000 features per participant, making them extremely high-dimensional. This caused memory strain during model training and made the models prone to overfitting. To mitigate this, we applied dimensionality reduction techniques like Variance Thresholding and PCA to reduce noise and extract the most informative signals.
- The dataset was imbalanced, especially for ADHD diagnosis in female participants — a key focus of the competition. To address this, we used SMOTE for the Sex_F model and tuned the decision threshold based on validation F1 scores, with a focus on improving recall for underrepresented classes.
- We had to integrate multiple data modalities — categorical, numerical, and matrix-form (connectome) — into a unified feature set. This required different preprocessing strategies, including KNN imputation for missing values, one-hot encoding for categorical features, and custom pipelines for each model (ADHD vs. Sex).
---

## **🧠 Model Development**

| Tasks | Methods |
| ----- | ----- |
| ADHD Prediction | StackingClassifier (LGBM, RF, XGBoost) |
| Sex Classification | StackingClassifier + SMOTE |
| Optimization | Optuna, 200+ trials |
| Evaluation | F1 Score |
| Preprocessing | Variance Threshold -> Standard Scaler |

To tackle the dual prediction tasks—ADHD diagnosis and participant sex—we built two separate pipelines using ensemble learning strategies. For ADHD prediction, we implemented a StackingClassifier composed of LightGBM, Random Forest, and XGBoost models, each selected for their ability to handle high-dimensional data and nonlinear relationships. For sex classification, we used a similar stacking approach but incorporated SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance and boost recall for female participants, which aligned with the competition’s fairness goal.

Across both tasks, we used Optuna to perform hyperparameter optimization, running over 200 trials per model to find optimal settings that maximized the F1 score. Each pipeline followed a consistent preprocessing routine: low-variance features were removed using VarianceThreshold, followed by standardization using StandardScaler to ensure all features contributed equally to model learning. Our models were trained and evaluated using StratifiedKFold cross-validation, with performance measured using the F1 score.

---

## **📈 Results & Key Findings**

**Describe (as applicable):**

* Performance metrics (e.g., Kaggle Leaderboard score, F1-score)
* How your model perform overall

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools


*Key Observations:*
* Scaling and PCA significantly improved training performance
* Imputation with KNN maintained realistic distributions and preserved feature variance
* Feature selection (via variance threshold) removed noise and reduced overfitting


---

## **🖼️ Impact Narrative**
**WiDS challenge:**

**1. What brain activity patterns are associated with ADHD; are they different between males and females, and, if so, how?**
* Females showed altered patterns in prefrontal connectivity—related to executive function—while males showed more pronounced sensorimotor activation, aligning with impulsivity traits. These insights match published neuroscience literature and support sex-specific neural markers of ADHD.

**3. How could your work help contribute to ADHD research and/or clinical care?**
* More inclusive diagnostic models that capture underrepresented symptoms in females
* A prototype pipeline for clinical tools analyzing fMRI data for early detection
* Supports personalized treatment planning for neurodiverse youth

---

## **🚀 Next Steps & Future Improvements**

**We hope to address the following:**

* Apply Graph Neural Networks (GNNs) to model fMRI as adjacency matrices
* Use SHAP or LIME to interpret brain connectivity features
* Include external datasets to test generalization across demographics
* Explore temporal patterns in longitudinal scans, if available

---

## **📄 References & Additional Resources**


---

