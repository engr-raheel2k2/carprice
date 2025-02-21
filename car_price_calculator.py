import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load the models and preprocessor
@st.cache_resource
def load_models_and_preprocessor():
    models = {
        'Linear Regression': joblib.load('linear_regression_model.pkl'),
        #'Random Forest': joblib.load('random_forest_model.pkl'),
        'Gradient Boosting': joblib.load('gradient_boosting_model.pkl')
    }
    preprocessor = joblib.load('preprocessor.pkl')
    return models, preprocessor

# Function to calculate the revised price
def calculate_revised_price(predictions, engine_condition, body_condition):
    average_price = sum(predictions.values()) / len(predictions)
    revised_price = average_price * engine_condition * body_condition
    return revised_price

def main():
    # Set the title of the web app with logo and site name in the header
    #logo = Image.open('logo.png')  # Ensure you have a logo image in the same directory
    #st.image(logo, width=100)  # Resize the logo to 100px width
    st.title("Car Price Predictor 🚗")

    # Add a header with a description
    st.header("Welcome to the Car Price Predictor!")
    st.write("""
    This app predicts the price of a car based on its features. 
    Fill in the details below and click **Predict Price** to see the results.
    """)

    # Add an image (replace 'car_image.jpg' with your image file)
    image = Image.open('car_image.jpg')  # Ensure you have a car image in the same directory
    st.image(image, caption='Car Image', width=500)  # Resize the car image to 500px width

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

    # Add an expander for additional information
    with st.expander("ℹ️ About this app"):
        st.write("""
        This app uses machine learning models to predict car prices based on various features such as:
        - City
        - Assembly (Local/Imported)
        - Body Type
        - Make and Model
        - Year
        - Engine Capacity
        - Transmission
        - Fuel Type
        - Color
        - Registration Status
        - Mileage
        """)
        st.write("The models used are Linear Regression, Random Forest, and Gradient Boosting.")
        st.write("The hybrid model combines the results of all three models with additional parameters like engine condition and body condition.")

# Run the web application
if __name__ == "__main__":
    main()
