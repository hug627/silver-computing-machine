import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page title and icon
st.set_page_config(page_title="Population Predictor", page_icon="🌍")

# Main title
st.title("🌍 World Bank Population Predictor")
st.write("This app predicts population based on country and year data!")

# Load the saved model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("machine_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please make sure 'machine_model.pkl' is in the same folder.")
        return None

model = load_model()

if model is not None:
    st.success("✅ Model loaded successfully!")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📋 Enter Information")
        
        # Input fields for user
        country_iso = st.text_input("Country Code (3 letters)", "KEN", help="Example: USA, KEN, CHN")
        year = st.number_input("Year", min_value=1960, max_value=2030, value=2020)
        country_id = st.text_input("Country ID (2 letters)", "KE", help="Example: US, KE, CN")
        country_name = st.text_input("Country Name", "Kenya", help="Full country name")
    
    with col2:
        st.header("🔍 Your Input")
        
        # Show what the user entered
        input_data = {
            "Country Code": country_iso,
            "Year": year,
            "Country ID": country_id,
            "Country Name": country_name
        }
        
        for key, value in input_data.items():
            st.write(f"**{key}:** {value}")
    
    # Big predict button
    if st.button("🚀 Predict Population", type="primary"):
        
        # Create input dataframe (same format as training data)
        input_df = pd.DataFrame({
            'countryiso3code': [country_iso],
            'date': [year],
            'country.id': [country_id],
            'country.value': [country_name]
        })
        
        # Apply one-hot encoding (same as training)
        input_encoded = pd.get_dummies(input_df, columns=['countryiso3code', 'country.id', 'country.value'])
        
        # Get all the columns that were used in training
        # (This is a simplified approach - in real projects, you'd save the column names)
        training_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
        
        # Add missing columns with zeros
        for col in training_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Remove extra columns and reorder to match training
        if len(training_columns) > 0:
            input_encoded = input_encoded[training_columns]
        
        try:
            # Make prediction
            prediction = model.predict(input_encoded)[0]
            
            # Display result
            st.success("🎉 Prediction Complete!")
            
            # Create a nice display box
            st.markdown("---")
            col_result1, col_result2, col_result3 = st.columns([1,2,1])
            
            with col_result2:
                st.metric(
                    label="🏘️ Predicted Population", 
                    value=f"{int(prediction):,}",
                    help="This is the estimated population for the given country and year"
                )
            
            st.markdown("---")
            
            # Additional information
            st.info(f"💡 The model predicts that **{country_name}** will have approximately **{int(prediction):,}** people in **{year}**.")
            
        except Exception as e:
            st.error(f"❌ Something went wrong with the prediction: {e}")
            
            # Help the user debug
            st.write("**Debug Info:**")
            st.write(f"- Input shape: {input_encoded.shape}")
            st.write(f"- Model expects: {len(training_columns) if training_columns else 'Unknown'} features")

# Sidebar with instructions
st.sidebar.header("📖 How to Use")
st.sidebar.write("""
1. **Enter Country Code**: 3-letter code (like USA, KEN)
2. **Choose Year**: Any year between 1960-2030
3. **Enter Country ID**: 2-letter code (like US, KE)
4. **Type Country Name**: Full name of the country
5. **Click Predict**: Get the population prediction!
""")

st.sidebar.markdown("---")
st.sidebar.header("🤔 Example Countries")
examples = {
    "🇰🇪 Kenya": {"Code": "KEN", "ID": "KE"},
    "🇺🇸 USA": {"Code": "USA", "ID": "US"},
    "🇨🇳 China": {"Code": "CHN", "ID": "CN"},
    "🇮🇳 India": {"Code": "IND", "ID": "IN"},
    "🇧🇷 Brazil": {"Code": "BRA", "ID": "BR"}
}

for country, codes in examples.items():
    st.sidebar.write(f"**{country}**")
    st.sidebar.write(f"Code: {codes['Code']}, ID: {codes['ID']}")

st.sidebar.markdown("---")
st.sidebar.write("💻 **Need Help?**")
st.sidebar.write("Make sure your model file 'machine_model.pkl' is in the same folder as this app!")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit 🚀 | World Bank Population Data 📊*")