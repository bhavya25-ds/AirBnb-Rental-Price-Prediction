# 🏠 Airbnb Rental Price Prediction

Predicting Airbnb listing prices in New York City from listing metadata. Pricing is a real pain point for hosts — set too high and you lose bookings, too low and you leave money on the table. This project focuses on regression fundamentals: outlier handling, cross-validation, and understanding where a model fails, not just where it succeeds.

## ✨ What I Did and Why
* **Outlier Removal with IQR:** Removed extreme price listings using the interquartile range method, reducing from 27,379 to 25,720 rows. Extreme outliers ($10,000/night listings) would skew the model toward the tail and hurt performance on normal-range predictions.
* **Label Encoding:** Encoded neighbourhood group and room type as numeric. Used LabelEncoder here — for tree-based models this works fine since Random Forest doesn't assume any ordinal relationship between encoded values.
* **Multi-Model Cross Validation:** Compared Linear Regression and Random Forest Regressor. Random Forest won significantly (54.5% vs 44.6% R²) — price is not a linear function of these features.
* **GridSearchCV Tuning:** Tuned `max_depth` and `n_estimators` on Random Forest. Found `max_depth=10` — deeper trees overfit on this dataset.
* **Hexbin Actual vs Predicted Plot:** Used hexbin instead of a scatter plot because 25,000 points would be completely unreadable. Shows where predictions cluster vs where they fail.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Data Analytics:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Machine Learning:** Scikit-Learn (RandomForestRegressor, LinearRegression, StandardScaler, LabelEncoder, GridSearchCV, cross_val_score)
* **Dataset:** NYC Airbnb Open Data (`data.csv`) — 48,895 listings, 8 features (neighbourhood group, latitude, longitude, room type, minimum nights, host listing count, availability)

## 🚀 Setup
1. Clone the repo.
2. Install dependencies: `pip install pandas numpy matplotlib scikit-learn`
3. Place `data.csv` in the same folder as `main.ipynb`.
4. Run `main.ipynb` cell by cell.
5. Find predictions in `Predictions.csv`.

## 📊 Results
* **Best Model:** Random Forest Regressor (`max_depth=10`, `n_estimators=300`)
* **Cross-Validation R²:** Linear Regression 44.6% · Random Forest 54.5%
* **Train R²:** ~100% · **Test R²:** 56.2% · **MAE:** $32.74
* **Key Observation:** The ~100% train R² vs 56.2% test R² gap means the model is overfitting, despite depth limiting. Airbnb pricing depends heavily on factors absent from this dataset — photos, review scores, host response rate, description quality — which creates a natural ceiling on how high R² can go regardless of model.

## ⚠️ Limitations
* Label encoding neighbourhood group could be improved with target encoding or keeping it as one-hot — the model may be inferring a false ordinal relationship.
* The 56.2% R² means the model explains just over half the variance in price. Not reliable enough for actual host pricing decisions without additional features.
* High-price listings (>$500) are underrepresented even after IQR removal, so predictions at the top end are unreliable.
