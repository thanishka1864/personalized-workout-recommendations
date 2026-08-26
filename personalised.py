# ============================================================
#       PERSONALIZED WORKOUT RECOMMENDATION SYSTEM
#       MACHINE LEARNING - RANDOM FOREST
# ============================================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# 1. DATASET
# ------------------------------------------------------------

data = {
    "Age": [20, 22, 25, 30, 28, 35, 40, 23, 27, 32,
            21, 29, 38, 24, 31, 26, 34, 22, 36, 28,
            25, 30, 27, 23, 35],

    "Gender": [
        "Male", "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Female",
        "Male", "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Female",
        "Male", "Female", "Male", "Female", "Male"
    ],

    "Weight": [
        70, 55, 80, 65, 75, 60, 90, 52, 78, 62,
        68, 58, 85, 54, 82, 57, 88, 50, 92, 61,
        76, 59, 81, 53, 87
    ],

    "Height": [
        175, 160, 180, 165, 178, 162, 182, 158, 176, 164,
        172, 161, 185, 159, 180, 163, 183, 157, 188, 166,
        177, 162, 181, 160, 184
    ],

    "Fitness_Level": [
        "Beginner", "Beginner", "Intermediate", "Beginner",
        "Intermediate", "Beginner", "Advanced", "Beginner",
        "Intermediate", "Beginner", "Intermediate", "Beginner",
        "Advanced", "Beginner", "Advanced", "Intermediate",
        "Advanced", "Beginner", "Advanced", "Intermediate",
        "Intermediate", "Beginner", "Advanced", "Beginner",
        "Advanced"
    ],

    "Goal": [
        "Weight Loss", "Weight Loss", "Muscle Gain", "Weight Loss",
        "Muscle Gain", "Weight Loss", "Muscle Gain", "Weight Loss",
        "Muscle Gain", "Weight Loss", "Muscle Gain", "Weight Loss",
        "Muscle Gain", "Weight Loss", "Muscle Gain", "Weight Loss",
        "Muscle Gain", "Weight Loss", "Muscle Gain", "Weight Loss",
        "Muscle Gain", "Weight Loss", "Muscle Gain", "Weight Loss",
        "Muscle Gain"
    ],

    "Workout_Preference": [
        "Cardio", "Yoga", "Strength", "Cardio", "Strength",
        "Yoga", "Strength", "Cardio", "Strength", "Yoga",
        "Strength", "Cardio", "Strength", "Yoga", "Strength",
        "Cardio", "Strength", "Yoga", "Strength", "Cardio",
        "Strength", "Yoga", "Strength", "Cardio", "Strength"
    ],

    "Recommended_Workout": [
        "Running", "Yoga", "Weight Training", "Cycling",
        "Weight Training", "Yoga", "Heavy Weight Training",
        "Brisk Walking", "Weight Training", "Yoga",
        "Weight Training", "Cycling", "Heavy Weight Training",
        "Yoga", "Heavy Weight Training", "Running",
        "Heavy Weight Training", "Yoga", "Heavy Weight Training",
        "Cycling", "Weight Training", "Yoga",
        "Heavy Weight Training", "Running", "Heavy Weight Training"
    ]
}

df = pd.DataFrame(data)

# ------------------------------------------------------------
# 2. ENCODING
# ------------------------------------------------------------

categorical_columns = [
    "Gender",
    "Fitness_Level",
    "Goal",
    "Workout_Preference"
]

encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

target_encoder = LabelEncoder()

df["Recommended_Workout"] = target_encoder.fit_transform(
    df["Recommended_Workout"]
)

# ------------------------------------------------------------
# 3. FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop("Recommended_Workout", axis=1)
y = df["Recommended_Workout"]

# ------------------------------------------------------------
# 4. TRAIN MODEL
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print("Model trained successfully!")

# ------------------------------------------------------------
# 5. USER INPUT
# ------------------------------------------------------------

print("\n==========================================")
print(" PERSONALIZED FITNESS RECOMMENDATION")
print("==========================================")

age = int(input("Enter age: "))
gender = input("Enter gender (Male/Female): ")
weight = float(input("Enter weight (kg): "))
height = float(input("Enter height (cm): "))

fitness = input(
    "Fitness level (Beginner/Intermediate/Advanced): "
)

goal = input(
    "Goal (Weight Loss/Muscle Gain): "
)

preference = input(
    "Workout preference (Cardio/Yoga/Strength): "
)

# ------------------------------------------------------------
# 6. BMI CALCULATION
# ------------------------------------------------------------

height_m = height / 100

bmi = weight / (height_m ** 2)

print("\nYour BMI:", round(bmi, 2))

if bmi < 18.5:
    bmi_category = "Underweight"

elif bmi < 25:
    bmi_category = "Normal"

elif bmi < 30:
    bmi_category = "Overweight"

else:
    bmi_category = "Obese"

print("BMI Category:", bmi_category)

# ------------------------------------------------------------
# 7. ENCODE USER DATA
# ------------------------------------------------------------

gender_encoded = encoders["Gender"].transform([gender])[0]

fitness_encoded = encoders["Fitness_Level"].transform(
    [fitness]
)[0]

goal_encoded = encoders["Goal"].transform(
    [goal]
)[0]

preference_encoded = encoders["Workout_Preference"].transform(
    [preference]
)[0]

user_data = pd.DataFrame([[
    age,
    gender_encoded,
    weight,
    height,
    fitness_encoded,
    goal_encoded,
    preference_encoded
]], columns=X.columns)

# ------------------------------------------------------------
# 8. ML PREDICTION
# ------------------------------------------------------------

prediction = model.predict(user_data)

recommended_workout = target_encoder.inverse_transform(
    prediction
)[0]

# ------------------------------------------------------------
# 9. MULTIPLE WORKOUT RECOMMENDATIONS
# ------------------------------------------------------------

workout_plans = {

    "Weight Loss": {

        "Cardio": [
            "Running",
            "Cycling",
            "Jumping Rope",
            "Brisk Walking",
            "Swimming"
        ],

        "Yoga": [
            "Power Yoga",
            "Sun Salutation",
            "Vinyasa Yoga",
            "Core Yoga",
            "Flexibility Yoga"
        ],

        "Strength": [
            "Squats",
            "Lunges",
            "Push-ups",
            "Plank",
            "Bodyweight Training"
        ]
    },

    "Muscle Gain": {

        "Cardio": [
            "Incline Walking",
            "Cycling",
            "Light Jogging",
            "Rowing",
            "Stair Climbing"
        ],

        "Yoga": [
            "Power Yoga",
            "Strength Yoga",
            "Balance Yoga",
            "Core Yoga",
            "Flexibility Yoga"
        ],

        "Strength": [
            "Weight Training",
            "Bench Press",
            "Squats",
            "Deadlift",
            "Shoulder Press"
        ]
    }
}

# ------------------------------------------------------------
# 10. GET PERSONALIZED EXERCISES
# ------------------------------------------------------------

recommendations = workout_plans[goal][preference]

# ------------------------------------------------------------
# 11. FITNESS LEVEL ADJUSTMENT
# ------------------------------------------------------------

if fitness == "Beginner":

    duration = "20-30 minutes"
    frequency = "3-4 days/week"
    intensity = "Low to Moderate"
    rest = "2-3 rest days/week"

elif fitness == "Intermediate":

    duration = "30-45 minutes"
    frequency = "4-5 days/week"
    intensity = "Moderate"
    rest = "2 rest days/week"

else:

    duration = "45-60 minutes"
    frequency = "5-6 days/week"
    intensity = "Moderate to High"
    rest = "1-2 rest days/week"

# ------------------------------------------------------------
# 12. DISPLAY FINAL RECOMMENDATION
# ------------------------------------------------------------

print("\n==========================================")
print("       YOUR PERSONALIZED PLAN")
print("==========================================")

print("\nML Recommended Workout:")
print(recommended_workout)

print("\nRecommended Exercises:")

for i, exercise in enumerate(recommendations, 1):
    print(i, ".", exercise)

print("\nWorkout Duration:", duration)
print("Workout Frequency:", frequency)
print("Workout Intensity:", intensity)
print("Recommended Rest:", rest)

# ------------------------------------------------------------
# 13. GOAL-BASED ADVICE
# ------------------------------------------------------------

if goal == "Weight Loss":

    print("\nGoal:")
    print("Focus on calorie-burning exercises and regular activity.")

    print("\nSuggested Session:")
    print("Warm-up: 5-10 minutes")
    print("Main workout: 20-30 minutes")
    print("Cool-down: 5-10 minutes")

else:

    print("\nGoal:")
    print("Focus on progressive resistance and adequate recovery.")

    print("\nSuggested Session:")
    print("Warm-up: 5-10 minutes")
    print("Strength training: 30-40 minutes")
    print("Cool-down: 5-10 minutes")

print("\n==========================================")
print("       RECOMMENDATION COMPLETE")
print("==========================================")