import streamlit as st
import pandas as pd
import joblib
import json

# Configuración de la página
st.set_page_config(page_title="Predictor Model", layout="wide")

# Cargar el modelo y el scaler
@st.cache_resource
def load_model():
    model = joblib.load('models/best_regresor_model.joblib')
    # Intentar cargar el scaler si existe, si no, retornar None
    try:
        scaler = joblib.load('models/scaler.joblib')
    except:
        scaler = None
    
    with open('models/best_regresor_info.json', 'r') as f:
        model_info = json.load(f)
    return model, scaler, model_info

def main():
    st.title("Predictor Model")
    
    try:
        model, scaler, model_info = load_model()
        st.success("Modelo cargado exitosamente!")
        
        # Mostrar información del modelo
        with st.expander("Ver información del modelo"):
            st.json(model_info)
        
        # Crear el formulario de entrada
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                gender = st.selectbox('Género:', ['Male', 'Female'])
                contract = st.selectbox('Tipo de Contrato:', 
                                      ['Month-to-month', 'One year', 'Two year'])
                tenure = st.number_input('Tiempo con la empresa (meses):', 
                                       min_value=0, max_value=100, value=12)
                partner = st.selectbox('¿Tiene pareja?:', ['Yes', 'No'])
                
            with col2:
                payment_method = st.selectbox('Método de Pago:', 
                                            ['Electronic check', 'Mailed check', 
                                             'Bank transfer (automatic)', 
                                             'Credit card (automatic)'])
                multiple_lines = st.selectbox('¿Múltiples líneas?:', 
                                            ['Yes', 'No', 'No phone service'])
                streaming_tv = st.selectbox('¿TV Streaming?:', 
                                          ['Yes', 'No', 'No internet service'])
            
            submit_button = st.form_submit_button("Realizar Predicción")
        
        if submit_button:
            # Crear DataFrame con los datos de entrada
            input_data = pd.DataFrame({
                'gender': [gender],
                'Contract': [contract],
                'tenure': [tenure],
                'Partner': [partner],
                'PaymentMethod': [payment_method],
                'MultipleLines': [multiple_lines],
                'StreamingTV': [streaming_tv]
            })
            
            # Realizar one-hot encoding
            categorical_cols = ['gender', 'Contract', 'Partner', 
                              'PaymentMethod', 'MultipleLines', 'StreamingTV']
            input_encoded = pd.get_dummies(input_data, columns=categorical_cols)
            
            # Asegurar que todas las columnas del training estén presentes
            for col in model_info['feature_names']:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            # Reordenar columnas para coincidir con el training
            input_encoded = input_encoded[model_info['feature_names']]
            
            # Escalar datos solo si el scaler existe
            if scaler is not None:
                input_scaled = scaler.transform(input_encoded)
            else:
                input_scaled = input_encoded.values  # Convertir a numpy array
            
            # Realizar predicción
            prediction = model.predict(input_scaled)
            
            # Mostrar resultado
            st.subheader("Resultado de la Predicción:")
            if 'metrics' in model_info and 'accuracy' in model_info['metrics']:
                # Es un modelo de clasificación
                st.write(f"Predicción: {prediction[0]}")
            else:
                # Es un modelo de regresión
                st.write(f"Valor predicho: {prediction[0]:.2f}")
            
            # Mostrar datos de entrada procesados
            with st.expander("Ver datos de entrada procesados"):
                st.write(input_encoded)
    
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.write("Asegúrate de que los archivos del modelo estén en la carpeta 'models/'")

if __name__ == "__main__":
    main()