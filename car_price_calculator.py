import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os
import datetime

# Load the models and preprocessor
@st.cache_resource
def load_models_and_preprocessor():
    models = {
        'Linear Regression': joblib.load('linear_regression_model.pkl'),
        'Gradient Boosting': joblib.load('gradient_boosting_model.pkl')
    }
    preprocessor = joblib.load('preprocessor.pkl')
    return models, preprocessor

# Function to calculate the revised price
def calculate_revised_price(predictions, engine_condition, body_condition):
    market_sentiments = 1.1
    average_price = sum(predictions.values()) / len(predictions)
    revised_price = average_price * engine_condition * body_condition * market_sentiments
    return revised_price

def save_feedback(feedback, rating):
    feedback_file = "user_feedback.csv"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Debug: Print the current working directory
    st.write(f"Current working directory: {os.getcwd()}")
    
    # Check if the feedback file exists
    if not os.path.exists(feedback_file):
        st.write("Feedback file does not exist. Creating a new one.")
        feedback_df = pd.DataFrame(columns=["Timestamp", "Feedback", "Rating"])
    else:
        st.write("Feedback file exists. Loading it.")
        feedback_df = pd.read_csv(feedback_file)
    
    # Append new feedback
    new_feedback = pd.DataFrame({"Timestamp": [timestamp], "Feedback": [feedback], "Rating": [rating]})
    feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
    
    # Save to CSV
    feedback_df.to_csv(feedback_file, index=False)
    st.write(f"Feedback saved to: {feedback_file}")

def main():
    # Set the title of the web app with logo and site name in the header
    st.title("Car Price Predictor 🚗")

    # Add a header with a description
    st.header("Welcome to the Car Price Predictor!")
    st.write("""
    This app predicts the price of a car based on its features. 
    Fill in the details below and click **Predict Price** to see the results.
    """)

    # Add an image (replace 'car_image.jpg' with your image file)
    image = Image.open('car_image.jpg')  # Ensure you have a car image in the same directory
    st.image(image, caption='Car Image', width=300)  # Resize the car image to 500px width

    # Load the models and preprocessor
    models, preprocessor = load_models_and_preprocessor()

    # Create input fields for user input in the main page
    st.header("Enter Car Details")

    # Use columns to organize inputs (4-5 parameters in a row)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        city = st.selectbox("City", ["Lahore", "Karachi", "Islamabad", "Peshawar", "Faisalabad", "Other"])
        assembly = st.selectbox("Assembly", ["Local", "Imported"])
        body = st.selectbox("Body Type", ["Sedan", "Hatchback", "SUV", "Coupe", "Other"])

    with col2:
        make = st.selectbox("Make", ["Toyota", "Honda", "Suzuki", "Daihatsu", "Other"])
        model = st.text_input("Model", "Corolla")
        year = st.number_input("Year", min_value=1990, max_value=2023, value=2017)

    with col3:
        engine = st.number_input("Engine Capacity (cc)", min_value=500, max_value=5000, value=1800)
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Hybrid", "Electric"])

    with col4:
        color = st.selectbox("Color", ["White", "Black", "Silver", "Grey", "Red", "Blue", "Other"])
        registered = st.selectbox("Is the car registered?", ["Yes", "No"])
        mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=86000)

    # Additional parameters for revised formula
    st.header("Additional Parameters")
    col5, col6 = st.columns(2)

    with col5:
        engine_condition = st.selectbox("Engine Condition", [1.0, 0.9, 0.7, 0.6], format_func=lambda x: {
            1.0: "Excellent",
            0.9: "Good",
            0.7: "Fair",
            0.6: "Poor"
        }[x])

    with col6:
        body_condition = st.selectbox("Body Condition", [1.0, 0.9, 0.7, 0.6], format_func=lambda x: {
            1.0: "Excellent",
            0.9: "Good",
            0.7: "Fair",
            0.6: "Poor"
        }[x])

    # Create a DataFrame from user input
    user_data = pd.DataFrame([{
        'city': city,
        'assembly': assembly,
        'body': body,
        'make': make,
        'model': model,
        'year': year,
        'engine': engine,
        'transmission': transmission,
        'fuel': fuel,
        'color': color,
        'registered': registered,
        'mileage': mileage
    }])

    # Preprocess user input
    user_data_preprocessed = preprocessor.transform(user_data)

    # Predict using trained models
    if st.button("Predict Price"):
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(user_data_preprocessed)[0]

        # Calculate the revised price using the formula
        revised_price = calculate_revised_price(predictions, engine_condition, body_condition)

        # Calculate Value to Trade
        minimum_value = revised_price * 0.95
        average_value = revised_price * 1.0
        maximum_value = revised_price * 1.05

        # Display the results with custom colors
        st.subheader("Value to Trade")

        # Minimum (Blue)
        st.markdown(
            f"<div style='background-color: #ADD8E6; padding: 10px; border-radius: 5px;'>"
            f"<strong>Minimum:</strong> <span style='color: #0000FF;'>{minimum_value:.2f} PKR</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Average (Green)
        st.markdown(
            f"<div style='background-color: #90EE90; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
            f"<strong>Average:</strong> <span style='color: #008000;'>{average_value:.2f} PKR</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Maximum (Light Red)
        st.markdown(
            f"<div style='background-color: #FFCCCB; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
            f"<strong>Maximum:</strong> <span style='color: #FF0000;'>{maximum_value:.2f} PKR</span>"
            f"</div>",
            unsafe_allow_html=True
        )

 # Feedback Section
  #  st.header("Feedback")
   # feedback = st.text_area("Please share your feedback about the app:")
  #  rating = st.slider("Rate your experience (1 = Poor, 5 = Excellent):", 1, 5, 3)

   # if st.button("Submit Feedback"):
   #     if feedback.strip() == "":
   #         st.warning("Please provide feedback before submitting.")
   #     else:
   #         save_feedback(feedback, rating)
   #         st.success("Thank you for your feedback! It has been saved.")


    # Add an expander for additional information
    with st.expander("ℹ️ About this app"):
        st.write("""
        ### **About This App**
        Welcome to the **Car Price Predictor**! 🚗

        This app helps you estimate the market value of a car based on its features and condition. Whether you're buying, selling, or just curious, this tool provides a reliable price range to guide your decisions.

        ### **How to Use**
        1. Fill in the car details in the input fields.
        2. Select the condition of the **Engine** and **Body**.
        3. Click **Predict Price** to see the estimated value.
        4. Use the results to make informed decisions when buying or selling a car.

        ### **Defining Conditions**
        To get the most accurate prediction, you can specify the condition of the car's **Engine** and **Body**. Here’s what each condition means:

        #### **Engine Condition**
        - **Excellent**: The engine runs perfectly, with no issues. It has been regularly serviced, and there are no unusual noises, leaks, or performance problems.
        - **Good**: The engine runs well but may have minor issues, such as slight noise or reduced performance. It has been maintained but may need minor repairs.
        - **Fair**: The engine runs but has noticeable issues, such as reduced power, unusual noises, or minor leaks. It may require significant repairs or maintenance.
        - **Poor**: The engine has major issues, such as frequent breakdowns, poor performance, or significant leaks. It may need a complete overhaul or replacement.

        #### **Body Condition**
        - **Excellent**: The body has no scratches, dents, or touch-ups. The paint is original and in perfect condition.
        - **Good**: The body has minor scratches or small dents that are not very noticeable. There may be minor touch-ups, but the overall appearance is clean.
        - **Fair**: The body has noticeable scratches, dents, or touch-ups. The paint may have faded or chipped in some areas.
        - **Poor**: The body has significant damage, such as large dents, rust, or multiple scratches. The paint is in poor condition, and repairs are needed.

        ### **Disclaimer**
        The predictions provided by this app are based on historical data and machine learning models. Actual market prices may vary due to factors such as demand, location, and negotiation. Use this tool as a guide, not a definitive valuation.
        """)

   
# Run the web application
if __name__ == "__main__":
    main()
