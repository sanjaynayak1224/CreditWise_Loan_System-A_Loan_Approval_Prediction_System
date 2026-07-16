# 💼 CreditWise: Predicting Loan Approvals with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9%2B-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0%2B-darkblue?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning project designed to automate credit underwriting, evaluate borrower risk, and predict loan approvals. Using a dataset of 1,000 credit profiles ([loan_approval_data.csv](loan_approval_data.csv)), I developed and tuned classification models, achieving a peak accuracy of **88.0%** and a recall of **83.6%** with feature-engineered Logistic Regression.

---

## 📌 Project Overview & Problem Statement
In traditional banking, credit underwriting can be slow, manual, and prone to subjective bias. Automating this process helps lenders make faster, data-driven decisions while minimizing default rates. This repository showcases a full machine learning lifecycle—from data exploration and cleaning to feature engineering, model training, evaluation, and future roadmap planning.

---

## 🔍 The Pipeline & Modeling Workflow
The project follows a standard machine learning workflow, from clean-up to evaluation:

![The Pipeline & Modeling Workflow](pipeline.png)

### Behind the Scenes: Preprocessing & Pipeline Construction
To prepare raw application data for modeling and prevent data leakage:
- **Noise Reduction**: Dropped `Applicant_ID` immediately since it is a database key and would cause the model to overfit.
- **Missing Value Imputation**: Numeric column missing values are imputed using the mean, while categorical columns are filled with the most frequent value (mode).
- **Categorical Encoding**:
  - Ordinal features (e.g., `Education_Level`) are encoded using label encoding.
  - Nominal features (e.g., `Employment_Status`, `Marital_Status`) are encoded using one-hot encoding, with the first category dropped to avoid the **dummy variable trap** (multicollinearity).
- **Feature Scaling**: Applied `StandardScaler` to normalize continuous variables, ensuring that distance-based algorithms (like KNN) and coefficient-based estimators (like Logistic Regression) treat all features equitably.

---

## 📈 Exploratory Data Analysis (EDA)
Our initial exploration revealed key patterns in applicant profiles:
- **Class Imbalance**: Approximately **70%** of applications in the dataset were approved, while **30%** were rejected. This imbalance naturally skews model training and is a target for future optimization.
- **Credit Score Signal**: There is a distinct approval boundary; applicants with credit scores above `650` are significantly more likely to receive approvals.

![Exploratory Data Analysis](eda.png)
![Feature Correlation Heatmap](feature_correlation_heatmap.png)

---

## 📊 Model Evaluation & Results
We trained and evaluated three classification models under two configurations:
1. **Baseline**: Trained on standard cleaned and scaled features.
2. **Fine-Tuned / Engineered**: Trained on features augmented with polynomial variables—specifically $DTI^2$ (`DTI_Ratio_sq`) and $Credit\_Score^2$ (`Credit_Score_sq`)—to capture non-linear relationships, while dropping the original linear variables to prevent collinearity issues.

### 🏆 Model Performance Comparison

The exact metrics computed from our test set split (20% holdout) are summarized below:

| Model | Configuration | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Baseline | 86.5% | 78.3% | 77.1% | 77.7% |
| | **Fine-Tuned (Polynomial)** | **88.0%** | **78.5%** | **83.6%** | **81.0%** |
| **Naive Bayes** | Baseline | 86.5% | 80.4% | 73.8% | 76.9% |
| | Fine-Tuned (Polynomial) | 86.0% | 81.1% | 70.5% | 75.4% |
| **K-Nearest Neighbors** | Baseline | 76.0% | 62.7% | 52.5% | 57.1% |
| | Fine-Tuned (Polynomial) | 78.5% | 67.3% | 57.4% | 61.9% |

![Model Performance Comparison](model_performance_comparison.png)

### 💡 Key Insights
- **Non-Linear Feature Engineering Elevated Logistic Regression**: By adding quadratic features for the Debt-to-Income (DTI) Ratio and Credit Score ($DTI^2$ and $Score^2$), the model got better at capturing non-linear behavior. This push raised the accuracy to **88.0%** and recall to **83.6%**, which is highly desirable for lenders who want to maximize the detection of qualified borrowers while keeping defaults low.
- **Naive Bayes is Stable but Conservative**: Naive Bayes achieved the highest precision (**81.1%** in the fine-tuned config), proving highly reliable when it predicts approval, though its recall dropped to **70.5%**.
- **KNN Suffers from the Curse of Dimensionality**: After one-hot encoding categorical variables, the feature space expanded to 28 dimensions. Since KNN relies on Euclidean distance, this high-dimensional space made data points feel far apart, causing baseline accuracy to drop to **76.0%**.

---

## 🚀 Next Steps: Production Roadmap
To move this model closer to a production-ready system, we propose the following improvements:
1. **Addressing Class Imbalance**: Use `SMOTE` (Synthetic Minority Over-sampling Technique) or set `class_weight='balanced'` in estimators to handle the 70/30 approval skew.
2. **Advanced Modeling (Tree Ensembles)**: Transition from linear/distance-based models to tree ensembles like **XGBoost**, **LightGBM**, or **Random Forests** to capture complex feature interactions without manual polynomial engineering.
3. **Automated Hyperparameter Tuning**: Implement systematic `GridSearchCV` or `RandomizedSearchCV` with 5-fold cross-validation to search for optimal hyperparameter settings (e.g. regularization strength, tree depth).
4. **Domain-Specific Feature Engineering**: Create realistic financial metrics:
   - *Total Household Income*: Combine applicant and co-applicant incomes.
   - *Loan-to-Value (LTV) Ratio*: Ratio of loan amount to asset/collateral value.
   - *Debt-Service Coverage Ratio (DSCR)*: For deeper cash-flow analysis.
5. **Stratified K-Fold Cross-Validation**: Migrate to a 5-fold or 10-fold Stratified CV scheme to ensure model validation metrics are stable and generalized across different subsets of data.

---

## 🛠️ How to Run the Project Locally
To run the notebook and replicate this analysis locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/sanjaynayak1224/CreditWise_Loan_System-A_Loan_Approval_Prediction_System.git
cd CreditWise_Loan_System-A_Loan_Approval_Prediction_System
```

### 2. Set Up a Virtual Environment
* **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **macOS/Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Key dependencies listed in [requirements.txt](requirements.txt) include: `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, and `ipykernel`)*

### 4. Run the Jupyter Notebook
Open [CreditWiseLoanSystem.ipynb](CreditWiseLoanSystem.ipynb) in your IDE of choice (such as VS Code or Jupyter Lab), select the `.venv` kernel, and execute the cells.
