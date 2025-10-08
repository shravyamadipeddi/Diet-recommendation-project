import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Streamlit Page Configuration
st.set_page_config(page_title="Diet Recommendation System", layout="wide")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("daily_food_nutrition_dataset.csv")
    return df

df = load_data()

# Data Preprocessing
def preprocess_data(df):
    df = df.dropna()  # Remove missing values
    df = df.drop_duplicates()  # Remove duplicates
    df.reset_index(drop=True, inplace=True)
    return df

df = preprocess_data(df)

# Load the pre-trained KNN model and scaler
knn_model = joblib.load("diet_knn_model.pkl")  # Load the trained KNN model
scaler = joblib.load("scaler.pkl")  # Load the scaler

# Function to calculate BMI
def calculate_bmi(weight, height):
    height_m = height / 100  
    bmi = weight / (height_m ** 2)
    return round(bmi, 2)

# Function to calculate daily nutritional needs
def calculate_nutrition(gender, age, weight, height, activity_level, goal):
    if gender == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_multipliers = {
        "Little/no exercise": 1.2,
        "Light exercise (1-3 days/week)": 1.375,
        "Moderate exercise (3-5 days/week)": 1.55,
        "Heavy exercise (6-7 days/week)": 1.725,
        "Very heavy exercise (twice/day, intense)": 1.9,
    }
    
    calorie_needs = bmr * activity_multipliers.get(activity_level, 1.2)

    goals = {
        "Extreme weight loss": calorie_needs - 1000,
        "Weight loss": calorie_needs - 500,
        "Mild weight loss": calorie_needs - 200,
        "Maintain weight": calorie_needs,
        "Mild weight gain": calorie_needs + 200,
        "Weight gain": calorie_needs + 500,
        "Extreme weight gain": calorie_needs + 1000,
    }
    
    calories = round(goals.get(goal, calorie_needs))
    protein = round(weight * 1.2, 2)
    fat = round(calories * 0.25 / 9, 2)
    sugars = round(calories * 0.1 / 4, 2)
    sodium = 2300
    cholesterol = 300
    carbohydrates = round((calories - (protein * 4 + fat * 9 + sugars * 4)) / 4, 2)
    fiber = round(carbohydrates * 0.1, 2)
    
    return calories, protein, carbohydrates, fat, fiber, sugars, sodium, cholesterol

# Function to recommend food based on user's nutrition
def recommend_food(calories, protein, carbohydrates, fat, fiber, sugars, sodium, cholesterol):
    input_data = np.array([[calories, protein, carbohydrates, fat, fiber, sugars, sodium, cholesterol]])
    input_scaled = scaler.transform(input_data)
    distances, indices = knn_model.kneighbors(input_scaled, n_neighbors=4)
    return df.iloc[indices[0]]

# UI Styling
st.markdown("""
    <style>
        .description {
            font-size: 18px;
            color: #555;
            text-align: center;
            font-weight: 500;
            margin-top: 20px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.7;
        }
        .title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            color: #3e8e41;
            margin-bottom: 20px;
        }
          .meal-plan {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .meal-title {
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }
        .meal-info {
            font-size: 16px;
            color: #555;
            margin-bottom: 5px;
        }
        .highlight {
            font-weight: bold;
            color: #e67e22;
    </style>
""", unsafe_allow_html=True)

# Display UI Elements
st.markdown('<div class="title">AI-Powered Diet Recommendation System</div>', unsafe_allow_html=True)

st.markdown("""
    <div class="description">
        Welcome to the AI-powered Diet Recommendation System! Get your personalized diet plan designed to help you achieve your health and fitness goals. 
        Whether you aim to lose weight, maintain a healthy weight, or gain muscle mass, we provide a tailored solution based on your unique profile.
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("Enter Your Details")
age = st.sidebar.slider("Age", min_value=1, max_value=120, value=25)
height = st.sidebar.slider("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.sidebar.slider("Weight (kg)", min_value=20, max_value=200, value=70)
gender = st.sidebar.radio("Gender", ("Male", "Female"))

activity_level = st.sidebar.selectbox("Activity Level", ["Little/no exercise", "Light exercise (1-3 days/week)", "Moderate exercise (3-5 days/week)", "Heavy exercise (6-7 days/week)", "Very heavy exercise (twice/day, intense)"])

goal = st.sidebar.selectbox("Weight Goal", ["Extreme weight loss", "Weight loss", "Mild weight loss", "Maintain weight", "Mild weight gain", "Weight gain", "Extreme weight gain"])

if st.sidebar.button("Generate Meal Plan"):
    bmi = calculate_bmi(weight, height)
    calories, protein, carbohydrates, fat, fiber, sugars, sodium, cholesterol = calculate_nutrition(gender, age, weight, height, activity_level, goal)
    food_recommendations = recommend_food(calories, protein, carbohydrates, fat, fiber, sugars, sodium, cholesterol)
    
    st.subheader("Your Personalized Diet Plan")
    st.markdown(f"**BMI:** {bmi} kg/m²")
    st.markdown(f"**Daily Calories Requirement:** {calories} Calories/day")
    if bmi < 18.5:
        st.warning("You are underweight. Consider a nutrient-rich diet.")
    elif 18.5 <= bmi < 25:
        st.success("You have a normal BMI. Maintain a balanced diet!")
    elif 25 <= bmi < 30:
        st.warning("You are overweight. Focus on reducing calorie intake.")
    else:
        st.error("You are obese. Follow a strict diet and exercise plan.")

    st.markdown("---")
    st.subheader("Full-Day Meal Plan")
    meal_categories = ["Breakfast", "Lunch", "Dinner", "Snacks"]
    food_recommendations = food_recommendations.reset_index(drop=True)
    
    for idx, meal in enumerate(meal_categories):
        if idx < len(food_recommendations):
            row = food_recommendations.iloc[idx]
            st.markdown(f"""
            <div class="meal-plan">
                <h3 class="meal-title">{meal}: {row['Food_Item']}</h3>
                <p class="meal-info"><span class="highlight">Category:</span> {row['Category']}</p>
                <p class="meal-info"><span class="highlight">Calories:</span> {row['Calories (kcal)']} kcal</p>
                <p class="meal-info"><span class="highlight">Protein:</span> {row['Protein (g)']} g | 
                <span class="highlight">Carbs:</span> {row['Carbohydrates (g)']} g | <span class="highlight">Fat:</span> {row['Fat (g)']} g</p>
                <p class="meal-info"><span class="highlight">Fiber:</span> {row['Fiber (g)']} g | 
                <span class="highlight">Sugars:</span> {row['Sugars (g)']} g</p>
                <p class="meal-info"><span class="highlight">Sodium:</span> {row['Sodium (mg)']} mg | 
                <span class="highlight">Cholesterol:</span> {row['Cholesterol (mg)']} mg</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.success("Enjoy your personalized meal plan and stay healthy! 🎯")