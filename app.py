import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("model.pkl")

st.title("🌍 Earthquake Magnitude Prediction")
st.write("Enter the feature values to predict earthquake magnitude")

st.markdown("---")

# Geographic Location Section
st.subheader("📍 Geographic Location")
col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input(
        "🌐 Latitude", 
        min_value=-90.0, 
        max_value=90.0, 
        value=0.0, 
        step=0.001,
        help="Geographic latitude (-90 to 90 degrees)"
    )

with col2:
    longitude = st.number_input(
        "🌐 Longitude", 
        min_value=-180.0, 
        max_value=180.0, 
        value=0.0, 
        step=0.001,
        help="Geographic longitude (-180 to 180 degrees)"
    )

st.markdown("---")

# Earthquake Properties Section
st.subheader("🏔️ Earthquake Properties")

depth = st.number_input(
    "⬇️ Depth (km)", 
    min_value=0.0, 
    value=10.0, 
    step=0.1,
    help="Depth of the earthquake in kilometers"
)

magNst = st.number_input(
    "🎛️ Magnitude NST", 
    min_value=0, 
    value=10, 
    step=1,
    help="Number of stations used to determine magnitude"
)

st.markdown("---")

# Classification Features Section
st.subheader("📋 Classification Features")

magType = st.selectbox(
    "📏 Magnitude Type",
    options=["ml", "md", "mw", "mb", "ms"],
    help="Type of magnitude calculation used"
)

# Encode magType to numeric (same mapping as training)
mag_mapping = {"ml": 0, "md": 1, "mw": 2, "mb": 3, "ms": 4}
magType_encoded = mag_mapping[magType]

st.markdown("---")

# Create prediction section
st.subheader("🔮 Prediction")

# Prepare input data
inputs = [latitude, longitude, depth, magType_encoded, magNst]

# Display current input values
with st.expander("📋 Current Input Values"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Latitude:** {latitude}")
        st.write(f"**Longitude:** {longitude}")
        st.write(f"**Depth:** {depth} km")
    with col_b:
        st.write(f"**Magnitude NST:** {magNst}")
        st.write(f"**Magnitude Type (encoded):** {magType_encoded}")

# Prediction button
if st.button("🎯 Predict Earthquake Magnitude", type="primary"):
    try:
        # Prepare data for prediction
        data = np.array([inputs])
        
        # Make prediction
        prediction = model.predict(data)
        predicted_magnitude = prediction[0]
        
        # Display result with color coding based on magnitude
        st.success("✅ Prediction Complete!")
        
        if predicted_magnitude < 3.0:
            st.info(f"🟢 **Predicted Magnitude: {predicted_magnitude:.2f}**")
            st.info("Minor earthquake - Usually not felt")
        elif predicted_magnitude < 4.0:
            st.info(f"🟡 **Predicted Magnitude: {predicted_magnitude:.2f}**")
            st.info("Light earthquake - Often felt but rarely causes damage")
        elif predicted_magnitude < 5.0:
            st.warning(f"🟠 **Predicted Magnitude: {predicted_magnitude:.2f}**")
            st.warning("Moderate earthquake - Can cause minor damage")
        elif predicted_magnitude < 6.0:
            st.warning(f"🔴 **Predicted Magnitude: {predicted_magnitude:.2f}**")
            st.warning("Strong earthquake - Can cause damage")
        else:
            st.error(f"🆘 **Predicted Magnitude: {predicted_magnitude:.2f}**")
            st.error("Major earthquake - Can cause serious damage")
            
        # Additional information
        st.markdown("---")
        st.subheader("📊 Magnitude Scale Reference")
        
        scale_info = """
        - **< 3.0**: Micro - Not felt
        - **3.0 - 3.9**: Minor - Often felt, rarely causes damage
        - **4.0 - 4.9**: Light - Noticeable shaking, minor damage
        - **5.0 - 5.9**: Moderate - Can cause damage to buildings
        - **6.0 - 6.9**: Strong - Can cause damage over wide areas
        - **7.0+**: Major - Can cause serious damage over large areas
        """
        st.markdown(scale_info)
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.error("Please check your model file and input values")

# Footer
st.markdown("---")
st.markdown("*🔬 This prediction is based on machine learning model trained on historical earthquake data*")
