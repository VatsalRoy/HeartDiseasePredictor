# Heart Disease Predictor

A machine learning project to predict the likelihood of heart disease in a patient based on various clinical parameters.

The project includes an exploratory data analysis (EDA) and model training notebook, a trained model, a feature scaler, and a Streamlit web application for interactive predictions.

-----

## 🚀 Technologies Used

  * **Python**
  * **Data Analysis & Manipulation:** Pandas
  * **Machine Learning:** Scikit-learn (used for various models including Logistic Regression, KNN, GaussianNB, Decision Tree, and SVM)
  * **Deployment:** Streamlit
  * **Serialization:** Joblib

-----

## 📦 Repository Contents

| File | Description |
| :--- | :--- |
| `HeartDiseaseML.ipynb` | Jupyter notebook containing the full data cleaning, EDA, model training, and evaluation process. |
| `app.py` | The Streamlit web application. |
| `heart.csv` | The dataset used for training the model. |
| `KNN_heart.pkl` | The serialized **K-Nearest Neighbors (KNN)** classification model used for deployment. |
| `scaler_heart.pkl` | The fitted `StandardScaler` used to preprocess numerical input features. |
| `columns.pkl` | A list of feature column names used to ensure consistent input for the model. |

-----

## 🧠 Model & Performance

The project compared five classification models: Logistic Regression, K-Nearest Neighbors (KNN), Naive Bayes, Decision Tree, and SVM.

**Logistic Regression** showed the highest accuracy and F1-score on the test set, while **KNN** was selected for deployment.

| Model | Accuracy (Test Set) | F1 Score (Test Set) |
| :--- | :--- | :--- |
| **Logistic Regression** | 0.8696 | 0.8857 |
| **KNN (Deployed)** | 0.8641 | 0.8815 |
| Naive Bayes | 0.8533 | 0.8683 |
| SVM | 0.8478 | 0.8679 |
| Decision Tree | 0.7772 | 0.8000 |

-----

## ⚙️ Installation and Setup

1.  **Install dependencies (requires Python):**

    ```bash
    pip install pandas numpy scikit-learn joblib streamlit
    ```

    *(Note: The full notebook also requires `seaborn` and `matplotlib` for EDA.)*

2.  **Ensure all model and data files are in the same directory:**

      * `app.py`
      * `KNN_heart.pkl`
      * `scaler_heart.pkl`
      * `columns.pkl`

-----

## 🚀 Running the Web Application

Start the interactive prediction application using the Streamlit CLI:

```bash
streamlit run app.py
```

This command will start the application and open the prediction interface in your default web browser.

-----

## 📝 Data Preprocessing Highlights

The following key steps were performed in the `HeartDiseaseML.ipynb` notebook prior to model training:

1.  **Missing Value Imputation:** Zero values found in `Cholesterol` and `RestingBP` were replaced with the mean of the non-zero values for those respective columns.
2.  **Encoding:** Categorical features (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) were converted into numerical features using **one-hot encoding** (`pd.get_dummies`).
3.  **Scaling:** All numerical features were scaled using **`StandardScaler`** to normalize the data before training the models.
