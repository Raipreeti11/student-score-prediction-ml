import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Creating dataset---------------------------------------------------------
data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Score": [35,40,50,55,65,70,80,90]
}

#dataset to dataframe conversion----------------------------------------------
df = pd.DataFrame(data)

# Define features and target----------------------------------------------------
X = df[['Hours']]
y = df['Score']

#Split dataset----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train model---------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions----------------------------------------------------------------
y_pred = model.predict(X_test)

# Model evaluation-------------------------------------------------------------
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

#  Predict new value-------------------------------------------------------------
prediction = model.predict([[9]])
print("Predicted score for 9 study hours:", prediction)

# Visualization--------------------------------------------------------------
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Score Prediction")
plt.show()