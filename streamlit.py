import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# Set the Streamlit app layout and page title
st.set_page_config(page_title="Animal Shelter Outcomes", layout="wide")

# Load the cleaned dataset
df = pd.read_csv(r"C:\\Users\\Administrator\\Desktop\\cleaned_animal_outcomes.csv")

# Add a sidebar for navigation
st.sidebar.title("Animal Shelter Outcomes")
st.sidebar.markdown("Navigate through the app using the options below:")
page = st.sidebar.radio("Sections", ["Home", "Visualizations", "Model Insights", "About"])

# 1. Home Section
if page == "Home":
    st.title("Welcome to the Animal Shelter Outcomes Dashboard")
    st.markdown("""
    This dashboard provides insights into the outcomes of animals in shelters based on various factors. 
    Use the navigation menu to explore:
    - **Visualizations** for trends and distributions.
    - **Model Insights** to check predictive accuracy.
    - **About** to learn more about this project.
    """)
    st.image("https://media.licdn.com/dms/image/v2/C4E12AQFxcL8hYjxb3Q/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1582581371175?e=2147483647&v=beta&t=n6z8RyjZYkYQ9cFleGwWwhzOyxUM1TWzU_9VIDAmWhA", use_container_width=True)

    st.header("Dataset Overview")
    st.dataframe(df.head())  # Show the first few rows of the dataset

# 2. Visualizations Section
elif page == "Visualizations":
    st.title("Visualizations")
    st.markdown("Explore key insights from the data below:")

    # Outcome Type Distribution
    st.subheader("Outcome Type Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='outcome_type', data=df, ax=ax, palette="viridis")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Animal Type Distribution
    st.subheader("Animal Type Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='animal_type', hue='outcome_type', data=df, ax=ax, palette="viridis")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Age Distribution
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Age Numeric'], kde=True, ax=ax, color="teal")  # Replace 'Age Numeric' with your actual age column name
    st.pyplot(fig)
    # Boxplot for Age by Outcome Type
    st.subheader("Age by Outcome Type")
    fig, ax = plt.subplots()
    sns.boxplot(x='outcome_type', y='Age Numeric', data=df, ax=ax)
    ax.set_title("Age by Outcome Type")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    
    

    # Correlation Matrix
    st.subheader("Correlation Matrix of Numeric Features")
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.title("Correlation Matrix")
    st.pyplot(fig)

    # Adoption Outcomes Over Time
    st.subheader("Adoption Outcomes Over Time")
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    monthly_outcomes = df.groupby([df['datetime'].dt.to_period('M'), 'outcome_type']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_outcomes.plot(kind='line', ax=ax, colormap="tab10")
    plt.title("Monthly Adoption Trends")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 3. Model Insights Section
elif page == "Model Insights":
    st.title("Model Insights")
    st.markdown("""
    Here we evaluate the performance of a predictive model that classifies animal outcomes.
    """)
     # Load the trained model
    import joblib
    model = joblib.load(r"C:\Users\Administrator\Desktop\project\random_forest_model.pkl")


    # Model Performance
    st.subheader("Model Accuracy")
    accuracy = 0.8687707641196013  # Replace with your model's accuracy
    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = [[5990,2,0,1,0,0,639,1,0],
          [5,134,5,5,0,0,2,0,5],
          [5,3,31,14,0,0,4,0,0],
          [9,7,12,1214,0,0,1,0,1],
          [4,3,0,1,1,0,1,0,0],
          [1,0,0,2,0,0,0,0,0],
          [1293,0,2,2,0,0,1588,0,0],
          [20,0,0,0,0,0,4,0,0],
          [0,0,0,0,0,0,0,0,4640]]  # Replace with your confusion matrix
    st.dataframe(pd.DataFrame(cm))
    st.subheader("Make a Prediction")
    st.markdown("Provide the necessary details below to predict the outcome for an animal:")

    # Collect user inputs
    animal_type = st.selectbox("Animal Type", ['Dog', 'Cat', 'Other', 'Bird', 'Livestock'])
    age = st.slider("Age in Years", 0, 25, 1)
    sex = st.selectbox("Sex Upon Outcome", ['Neutered Male', 'Spayed Female', 'Intact Male', 'Intact Female', 'Unknown'])
    breed = st.text_input("Breed", "Domestic Short Hair Mix")
    color = st.text_input("Color", "Black/White")
    outcome_subtype = st.selectbox("Outcome Subtype", ['Unknown', 'Aggressive', 'Behavior', 'Medical', 'Rabies Risk', 'Other'])

    # Encode inputs into a DataFrame
    input_data = {
        'animal_type': animal_type,
        'Age Numeric': age,
        'sex_upon_outcome': sex,
        'breed': breed,
        'color': color,
        'outcome_subtype': outcome_subtype
     }

    input_df = pd.DataFrame([input_data])  # Convert to DataFrame

    # One-hot encode user inputs (ensure alignment with training features)
    input_df = pd.get_dummies(input_df)

    # Ensure the input DataFrame matches the model's training columns by adding missing columns as zeros
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make prediction when the button is clicked
    if st.button("Predict Outcome"):
        prediction = model.predict(input_df)[0]
        st.write(f"### Predicted Outcome: **{prediction}**")


# 4. About Section
elif page == "About":
    st.title("About This Project")
    st.markdown("""
    **Animal Shelter Outcomes Dashboard** is a data-driven project aimed at analyzing and predicting 
    outcomes of shelter animals. It provides insights into:
    - The types of outcomes for animals in shelters.
    - The distribution and trends of animal types and ages.
    - Predictive modeling to aid in decision-making.
    
    Built with 💻 by [Khalood Sami].
    """)

# Footer Section
st.markdown("---")
st.markdown("<p style='text-align: center;'>Developed by <b>Khalood Sami</b> | Powered by Streamlit</p>", unsafe_allow_html=True)
