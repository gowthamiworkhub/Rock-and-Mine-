import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# -------------------------
# Load Dataset
# -------------------------
data = pd.read_csv(
    r"C:\Users\HP\Downloads\Copy of sonar data\Copy of sonar data.csv",
    header=None
)

# -------------------------
# Features and Target
# -------------------------
X = data.drop(columns=60, axis=1)
Y = data[60]

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2,
    stratify=Y
)

# -------------------------
# Create and Train Model
# -------------------------
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=10,
    random_state=2
)

model.fit(X_train, Y_train)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🚢 Rock vs Mine Prediction")

st.write("Paste 60 sonar values separated by comma, tab, or space.")

user_input = st.text_area("Enter 60 Values")

if st.button("Predict"):

    try:

        values = re.split(r'[\s,\t]+', user_input.strip())
        values = [float(x) for x in values if x != ""]

        if len(values) != 60:
            st.error(f"Please enter exactly 60 values. You entered {len(values)} values.")
        else:

            input_data = np.asarray(values).reshape(1, -1)

            prediction = model.predict(input_data)

            if prediction[0] == "R":
                st.success("🪨 The object is a ROCK")
            else:
                st.success("💣 The object is a MINE")

    except Exception as e:
        st.error(e)