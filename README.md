# WiDS Datathon 2025: Predicting ADHD and Sex from Brain Imaging
---

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Phuong-An Bui | @phganie |____ |
| Keerthi Chebrolu | @keerthic05 | _____ |
| Aryaman Tepal | @aryamantepal | ___ |

---

## **🎯 Project Highlights**

* Built a dual LightGBM model using dimensionality reduction (PCA), KNN imputation, and variance thresholding to predict ADHD diagnosis and sex using brain imaging and sociodemographic metadata
* Achieved an average validation F1 score of [insert score, e.g., 0.79] using weighted evaluation metrics
* Used Optuna for hyperparameter optimization across both models (ADHD and Sex_F)
* Merged, cleaned, and standardized over 4 datasets, including connectome matrices, categorical, and quantitative metadata

🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

```
# Clone the repo
git clone https://github.com/WiDS-Team-24/WiDS-Kaggle-Competition.git
cd WiDS_Team_24.ipynb

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

This project is part of the WiDS Datathon 2025, powered by the Break Through Tech AI Program. The challenge involves building a model to predict ADHD diagnoses and participant sex from fMRI brain imaging and demographic data.

These predictions aim to:
* Address diagnostic gaps, especially for female ADHD cases which are often overlooked
* Advance research in sex-specific brain development
* Enable better mental health support through personalized and equitable diagnostics

---

## **📊 Data Exploration**

**Describe:**

* The dataset(s) used (i.e., the data provided in Kaggle \+ any additional sources)
* Data exploration and preprocessing approaches
* Challenges and assumptions when working with the dataset(s)

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


---

## **🧠 Model Development**

* Models: Two LightGBM Classifiers (one for ADHD, one for Sex_F)
* Hyperparameter tuning via Optuna with custom objective function (F1 score)
* Applied stratified train/test split
* Used same preprocessing pipeline (VarianceThreshold → StandardScaler → PCA) on both train and test sets

---

## **📈 Results & Key Findings**

**Describe (as applicable):**

* Performance metrics (e.g., Kaggle Leaderboard score, F1-score)
* How your model performed overall
* How your model performed across different skin tones (AJL)
* Insights from evaluating model fairness (AJL)

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

**Address the following:**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

* Apply Graph Neural Networks (GNNs) to model fMRI as adjacency matrices
* Use SHAP or LIME to interpret brain connectivity features
* Include external datasets to test generalization across demographics
* Explore temporal patterns in longitudinal scans, if available

---

## **📄 References & Additional Resources**

* Cite any relevant papers, articles, or tools used in your project

---

