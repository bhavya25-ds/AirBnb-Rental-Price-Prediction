# 🏠 AirBnb Rental Price Prediction
Predicting Airbnb rental prices in New York City from listing features using Linear Regression and Random Forest Regressor, with outlier removal, cross-validation, and GridSearchCV hyperparameter tuning.

## ✨ Features
* **Exploratory Data Analysis:** Checks for null values and duplicates, visualizes price distribution, and plots relationships between key features (neighbourhood group, room type, minimum nights) and price.
* **Outlier Detection and Removal:** Identifies and removes extreme price listings using the IQR method, reducing the dataset from 27,379 to 25,720 rows for cleaner model training.
* **Label Encoding:** Converts categorical text columns (neighbourhood group, room type) into numeric values so the model can process them.
* **Multi-Model Cross Validation:** Compares Linear Regression and Random Forest Regressor using K-Fold cross validation to identify the best performing algorithm.
* **Hyperparameter Tuning:** Uses GridSearchCV to find the optimal `max_depth` and `n_estimators` for the Random Forest model across multiple combinations.
* **Actual vs Predicted Visualization:** Evaluates model performance visually using a hexbin plot with a perfect-prediction reference line.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Data Analytics:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Machine Learning:** Scikit-Learn (RandomForestRegressor, LinearRegression, StandardScaler, LabelEncoder, GridSearchCV, cross_val_score)
* **Dataset:** New York City Airbnb listings (`data.csv`) — 8 features per listing (neighbourhood group, latitude, longitude, room type, price, minimum nights, calculated host listings count, availability 365)

## 🚀 Setup
1. Clone the repo.
2. Install dependencies: `pip install pandas numpy matplotlib scikit-learn`
3. Place `data.csv` in the same folder as `main.ipynb`.
4. Run `main.ipynb` cell by cell.
5. Find predictions in `Predictions.csv`!

## 📊 Results
* **Best Model:** Random Forest Regressor (`max_depth=10`, `n_estimators=300`)
* **Cross-Validation Scores:** Linear Regression 44.6% · Random Forest 54.5%
* **Train R2:** ~100% · **Test R2:** 56.2% · **MAE:** $32.74
* **Key Observation:** The model predicts mid-range Airbnb prices ($50–$200) well but becomes less accurate for higher-priced listings, since expensive listings are rarer in the dataset and harder to learn from. Airbnb pricing is also influenced by factors not present in the data (photos, reviews, description quality), which naturally limits how high the R2 score can go.
