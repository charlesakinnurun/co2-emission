# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# %% [markdown]
# Data Loading

# %%
print("Loading data from 'co2.csv'....")
try:
    df = pd.read_csv("co2.csv")
    print("Data loaded successfully")
    print("First 5 rows of the dataset")
    print(df.head().to_string())
except FileNotFoundError:
    print("Error: The file 'co2.csv' was not found. Please ensure it is in the same directory.")
    exit()

# %% [markdown]
# Data Preprocessing and Feature Engineering

# %%
# We select our single feature (independent variable) and the target (dependent variable)

# Check for missing values
df_missing = df.isnull().sum()
print("Missing values")
print(df_missing)

# Check for duplicates
df_duplicates = df.duplicated().sum()
print("Duplicated values")
print(df_duplicates)

# Drop duplicate values
df = df.drop_duplicates()
df_duplicates = df.duplicated().sum()
print("Duplicates after droping")
print(df_duplicates)

# Define the features (X) and target (y). Note the column names from the CSV
# We reshape the feature to a 2D array, which is a requirement for sckit learn.
X = df[["Engine Size(L)"]].values # Independent variable (feature)
y = df[["CO2 Emissions(g/km)"]].values # Dependent variable (target)

print("Shape of features (X)",X.shape)
print("Shape of target (y)",y.shape)

# %% [markdown]
# Data Splitting

# %%
# We split the data into training set and a testing set
# The model learns from the training data and is then evaluated on the testing data
# which it has never seen before. We'll use a 70/30 split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

print("Number of sample in the training set",len(X_train))
print("Number of samples in the testing set",len(X_test))

# %% [markdown]
# Visualization before training

# %%
plt.scatter(X,y)
plt.title("Engine Size(L) vs C02 Emissions(g/km)")
plt.xlabel("Engine Size(L)")
plt.ylabel("CO2 Emissions(g/km)")
plt.grid()
plt.show()

# %% [markdown]
# Model Training

# %%
# We create an instance of the Linear Regression model and train it using the 
# training data. The "fit" method finds the best-fit line that represents the 
# relationship between the two variable

print("Training the Linear Regression.......")
model = LinearRegression()
model.fit(X_train,y_train)
print("Model training complete!")

# %% [markdown]
# Model Evaluation

# %%
# MSE measures the average squared difference predictions and actual values
# R-squared measures the proportion of variance in the target that is predictable from the features

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"Mean Squared Error (MSE):{mse:.2f}")
print(f"R-squared: {r2:.2f}")

# %% [markdown]
# Visualization of Results

# %%
# A scatter plot is an excellent way to visualize the relationship and the model's
# fit. We'll plot the actual data points and the best fit regression line

plt.figure(figsize=(10,6))
plt.scatter(X,y,color="green",alpha=0.7,label="Actual Data Points")
plt.plot(X,model.predict(X),color="red",linewidth=2,label="Linear Regression Line")
plt.title("Engine Size vs CO2 Emission")
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emission (g/km)")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# Makking a New Prediction

# %%
try:
    engine_size = float(input("Enter Engine Size (L) for prediction: "))
    new_X = np.array([[engine_size]])
    predicted_emission = model.predict(new_X)
    print(f"Predicted CO2 Emission (g/km) for Engine Size {engine_size}L: {predicted_emission[0][0]:.2f}")
except ValueError:
    print("Invalid input. Please enter a numeric value.")


