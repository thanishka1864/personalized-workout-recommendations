# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:21:27 2026

@author: thani_cohvxow
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file_path = r"C:\Users\thani_cohvxow\Downloads\fitness_and_workout_dataset.csv"

df = pd.read_csv(file_path)

def clean_value(value):
    return str(value).strip("[]").replace("'", "").replace('"', "").split(",")[0].strip()

df["fitness_level"] = df["level"].apply(clean_value)
df["target_goal"] = df["goal"].apply(clean_value)

df["time_per_workout"] = pd.to_numeric(df["time_per_workout"], errors="coerce")
df["program_length"] = pd.to_numeric(df["program_length"], errors="coerce")
df["total_exercises"] = pd.to_numeric(df["total_exercises"], errors="coerce")

df = df.dropna(subset=[
    "fitness_level", "target_goal", "equipment",
    "time_per_workout", "program_length", "total_exercises"
])

X = df[[
    "fitness_level", "equipment", "program_length",
    "time_per_workout", "total_exercises"
]]

X = pd.get_dummies(X)

encoder = LabelEncoder()
y = encoder.fit_transform(df["target_goal"])

class_counts = pd.Series(y).value_counts()
valid_classes = class_counts[class_counts >= 2].index
valid_rows = pd.Series(y).isin(valid_classes).to_numpy()

df = df.loc[valid_rows].reset_index(drop=True)
X = X.loc[valid_rows].reset_index(drop=True)
y = y[valid_rows]

if len(set(y)) < 2:
    print("Not enough goal categories to train the model.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nWORKOUT RECOMMENDATION SYSTEM")
    print("--------------------------------")
    print("Model Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

    model.fit(X, y)

    print("\nAvailable Fitness Goals:")
    goals = sorted(df["target_goal"].unique())
    for i, goal in enumerate(goals, 1):
        print(i, goal)

    goal_choice = int(input("\nChoose your goal number: "))
    selected_goal = goals[goal_choice - 1]

    levels = sorted(df["fitness_level"].unique())
    print("\nAvailable Fitness Levels:")
    for i, level in enumerate(levels, 1):
        print(i, level)

    level_choice = int(input("\nChoose your fitness level number: "))
    selected_level = levels[level_choice - 1]

    equipment_list = sorted(df["equipment"].astype(str).unique())
    print("\nAvailable Equipment:")
    print("0 Any equipment")
    for i, item in enumerate(equipment_list, 1):
        print(i, item)

    equipment_choice = int(input("\nChoose equipment number: "))

    if equipment_choice == 0:
        selected_equipment = None
    else:
        selected_equipment = equipment_list[equipment_choice - 1]

    duration = float(input("\nMaximum workout duration in minutes: "))

    recommendations = df[
        (df["target_goal"] == selected_goal) &
        (df["fitness_level"] == selected_level) &
        (df["time_per_workout"] <= duration)
    ].copy()

    if selected_equipment is not None:
        recommendations = recommendations[
            recommendations["equipment"].astype(str) == selected_equipment
        ]

    recommendations = recommendations.drop_duplicates(subset=["title"])

    if recommendations.empty:
        print("\nNo matching workouts found. Try different preferences.")
    else:
        print("\nYOUR RECOMMENDED WORKOUTS")
        print("--------------------------------")

        for i, (_, workout) in enumerate(recommendations.head(5).iterrows(), 1):
            print("\nWorkout", i)
            print("Name:", workout["title"])
            print("Goal:", workout["target_goal"])
            print("Fitness Level:", workout["fitness_level"])
            print("Equipment:", workout["equipment"])
            print("Duration:", workout["time_per_workout"], "minutes")
            print("Program Length:", workout["program_length"], "weeks")
            print("Total Exercises:", workout["total_exercises"])
            print("Description:", workout["description"])

        recommendations.to_csv("workout_recommendations.csv", index=False)
        print("\nRecommendations saved to workout_recommendations.csv")
