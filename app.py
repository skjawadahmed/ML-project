import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import base64
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# Define function to get base64 of a binary file
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Home", "Price Predict", "About", "Contact Us"],
        icons=["house-heart-fill", "car-front-fill", "envelope-heart-fill","person-fill"],
        menu_icon="list",
        default_index=0,
    )
    st.sidebar.header("Crafted with Precision and Passion by Jawad")

# Home page
if selected == "Home":
    page_bg_img = get_base64_of_bin_file('images/bg1.jpg')
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url({page_bg_img});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("""
        <h3 style='text-align: left; font-family: Arial Black; font-size: 40px; color: white; line-height: 1;'>
            <span style='font-family: "Broadway";'>VALUE WHEELS</span>
            <br>
            <span style='font-family: "Verdana"; font-size: 25px;'>Empowering Choices with</span>
            <span style='font-family: "Verdana"; font-size: 25px; color: limegreen;'>Price Insights</span>
        </h3>
        """,
        unsafe_allow_html=True
    )

# Price Predict page
if selected == "Price Predict":
    page_bg_img = get_base64_of_bin_file('images/blackGTR.jpg')
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url({page_bg_img});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    model = pk.load(open('model.pkl', 'rb'))
    st.markdown("<h1 style='color: white; font-family: Arial black; display: inline;'>USED CAR PRICE PREDICTOR</h1><br>", unsafe_allow_html=True)

    data = pd.read_csv('Cardetails.csv')

    def get_brand_name(car_name):
        car_name = car_name.split(' ')[0]
        return car_name.strip()
    data['name'] = data['name'].apply(get_brand_name)

    name = st.selectbox('Select Car Brand', data['name'].unique())
    year = st.slider('Car Manufactured Year', 1994, 2024)
    km_driven = st.slider('No of kms Driven', 11, 200000)
    fuel = st.selectbox('Fuel type', data['fuel'].unique())
    seller_type = st.selectbox('Seller type', data['seller_type'].unique())
    transmission = st.selectbox('Transmission type', data['transmission'].unique())
    owner = st.selectbox('Owner type', data['owner'].unique())
    mileage = st.slider('Car Mileage', 10, 40)
    engine = st.slider('Engine CC', 700, 5000)
    max_power = st.slider('Max Power', 0, 200)
    seats = st.slider('No of Seats', 5, 10)

    st.markdown("""
        <style>
        div.stButton > button {
            font-size: 20px;
            padding: 10px 60px;
        }
        </style>
        """, unsafe_allow_html=True)

    if st.button("PREDICT"):
        input_data_models = pd.DataFrame(
            [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
            columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
        )

        input_data_models['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
                                           [1, 2, 3, 4, 5], inplace=True)
        input_data_models['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
        input_data_models['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
        input_data_models['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
        input_data_models['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
                                           'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
                                           'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
                                           'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
                                           'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                                          inplace=True)

        car_price = model.predict(input_data_models)

        st.markdown(f"<h1 style='font-size:30px; color:white;'>Car Price is going to be ₹{car_price[0]}</h1>", unsafe_allow_html=True)

# About page
if selected == "About":
    page_bg_img = get_base64_of_bin_file('images/mustangfrontblack.jpg')
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url({page_bg_img});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("ABOUT")

    st.markdown("""
    <h6 style='text-align: left; font-size: 18px; color: white'>
        Welcome to our Used Car Price Prediction App, your reliable companion for making informed decisions in the used car market. Our application leverages advanced machine learning algorithms and comprehensive data analysis to predict the market value of used cars accurately.
    </h6>
    """, unsafe_allow_html=True)

    st.markdown("""<h1 style='text-align: left; font-family: Arial black; font-size: 25px; color: white'>Our<span style='color: limegreen'> Mission</span></h1>""", unsafe_allow_html=True)
    st.write("**Empowering Choices with Price Insights**")
    st.write("We aim to empower users by providing transparent, data-driven price insights that help buyers and sellers make informed decisions. Whether you're looking to buy a used car at a fair price or sell your vehicle with confidence, our app is designed to meet your needs.")

    st.markdown("""<h1 style='text-align: left; font-family: Arial black; font-size: 25px; color: white'>Key<span style='color: limegreen'> Features</span></h1>""", unsafe_allow_html=True)
    st.write("""
    - **Accurate Predictions:** Our machine learning models are trained on extensive datasets to ensure precise price predictions.
    - **User-Friendly Interface:** Easily input your car's details and get instant price estimates.
    - **Comprehensive Data:** Our app considers a wide range of factors, including make, model, year, mileage, condition, and market trends.
    - **Real-Time Updates:** Stay up-to-date with the latest market values and trends.
    - **Comparative Analysis:** Compare prices across different models and regions to find the best deals.
    """)

    st.subheader("How It Works")
    st.write("""
    1. **Input Vehicle Details:** Enter specific details about the car, such as make, model, year, mileage, and condition.
    2. **Data Processing:** Our algorithm processes the input data and compares it with historical sales data and current market trends.
    3. **Price Prediction:** The app provides a predicted price range, giving you a clear understanding of the car's market value.
    4. **Decision Support:** Use the predicted price to negotiate better deals, whether buying or selling.
    """)

    st.subheader("Why Choose Us?")
    st.write("""
    - **Accuracy:** We use state-of-the-art machine learning techniques to deliver reliable price estimates.
    - **Transparency:** Our predictions are backed by data, ensuring transparency and trust.
    - **Convenience:** Get price predictions anytime, anywhere, without the need for manual research.
    - **Expertise:** Our team comprises experienced data scientists and automotive industry experts dedicated to providing the best service.
    """)

    st.subheader("Contact Us")
    st.write("If you have any questions, feedback, or need assistance, please don't hesitate to reach out to us at [contact information]. We're here to help you make the best choices in the used car market.")

    st.write("Thank you for choosing our Used Car Price Prediction App. We look forward to assisting you in your journey to buy or sell used cars with confidence.")
    st.markdown("""
    <footer style='text-align: center; font-family: Arial; font-size: 14px; color: white;'>
        <hr style='border-color: white;'>
            Copyright &copy; Designed and Developed by S K Jawad Ahmed - All rights reserved.
    </footer>
    """, 
    unsafe_allow_html=True
    )
if selected == "Contact Us":
    # Title of the Contact Us page
    st.title("Contact Us")

    # Form for user input
    with st.form(key='contact_form'):
        st.subheader("We'd love to hear from you!")
        
        # Input fields
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        emailFlag = False
        # Validate the email format
        if email:
            email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
            if re.match(email_pattern, email):
                emailFlag = True
            else:
                emailFlag = False
                st.error("Invalid email address. Please try again.")

        # Submit button
        submit_button = st.form_submit_button("Send")
        
        if submit_button and emailFlag:
            # Display a thank you message or perform further actions
            st.success("Thank you for your message! We will get back to you soon.")
            #send email
            # Sender and Receiver details
            receiver_email = "skjawadahmed07@gmail.com"
            sender_email = "skjawadahmed07@gmail.com"
            password = "rcgu fruc bemw lfeb"  # Replace with the app password for your Gmail account

            # Email content
            subject = "Test Email from Python"
            body = f"This is from Value Wheels \n Name: {name} \n Email: {email} \n Message: {message}"
            # Create the email message
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))

            # Send the email
            try:
                smtp_server = "smtp.gmail.com"
                smtp_port = 587  # Port for TLS
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()  # Upgrade the connection to secure
                    server.login(sender_email, password)  # Login to the SMTP server
                    server.send_message(message)  # Send the email
                st.success("Email sent successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")