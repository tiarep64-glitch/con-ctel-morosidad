import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="ConecTel - Predictor de Morosidad", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('model.joblib')
    imputer_num = joblib.load('imputer_num.joblib')
    imputer_cat = joblib.load('imputer_cat.joblib')
    encoder = joblib.load('encoder.joblib')
    return model, imputer_num, imputer_cat, encoder

try:
    model, imputer_num, imputer_cat, encoder = load_assets()
except Exception as e:
    st.error(f"❌ Error al cargar los archivos del modelo: {e}")

st.title("🔔 Sistema de Alerta Temprana de Morosidad — ConecTel S.A.")
st.markdown("Use este formulario para evaluar proactivamente el riesgo financiero de un cliente antes de los próximos 6 meses.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Datos Demográficos y de Cuenta")
    edad = st.slider("Edad", 18, 100, 35)
    genero = st.selectbox("Género", ["Masculino", "Femenino"])
    region = st.selectbox("Región", ["Metropolitana", "Valparaíso", "Antofagasta", "Araucanía", "Biobío"])
    tipo_contrato = st.selectbox("Tipo de Contrato", ["Mensual", "Anual", "Bianual"])
    plan = st.selectbox("Tipo de Plan", ["Básico", "Estándar", "Premium"])
    antiguedad_meses = st.number_input("Antigüedad del cliente (meses)", min_value=0, value=12)
    meses_sin_reajuste = st.number_input("Meses sin reajuste tarifario", min_value=0, value=6)

with col2:
    st.subheader("💸 Datos Financieros y de Comportamiento")
    factura_mensual_clp = st.number_input("Monto Factura Mensual (CLP)", min_value=0, value=35000)
    ingreso_estimado_clp = st.number_input("Ingreso Estimado Mensual (CLP)", min_value=0, value=650000)
    dias_mora_hist = st.number_input("Días de mora históricos", min_value=0, value=0)
    reclamos_12m = st.number_input("Cantidad de reclamos (últimos 12 meses)", min_value=0, value=0)
    llamadas_soporte_6m = st.number_input("Llamadas a soporte técnico (últimos 6 meses)", min_value=0, value=1)
    nps = st.slider("Nota NPS dada por el cliente (Satisfacción 1-10)", 1, 10, 7)
    descuento_activo = st.selectbox("¿Tiene descuento activo actualmente?", ["Sí", "No"])
    cambios_plan_12m = st.number_input("Cambios de plan realizados en el año", min_value=0, value=0)

tiene_internet, velocidad_mbps, tiene_tv, tiene_linea_movil, num_servicios = 1, 300.0, 1, 1, 3
metodo_pago = "WebPay"

if st.button("Evaluar Riesgo de Cliente", type="primary"):
    try:
        input_dict = {
            'region': region, 'edad': float(edad), 'genero': genero, 'tipo_contrato': tipo_contrato,
            'antiguedad_meses': float(antiguedad_meses), 'plan': plan, 'tiene_internet': float(tiene_internet),
            'velocidad_mbps': float(velocidad_mbps), 'tiene_tv': float(tiene_tv), 'tiene_linea_movil': float(tiene_linea_movil),
            'num_servicios': float(num_servicios), 'factura_mensual_clp': float(factura_mensual_clp),
            'metodo_pago': metodo_pago, 'dias_mora_hist': float(dias_mora_hist), 'reclamos_12m': float(reclamos_12m),
            'llamadas_soporte_6m': float(llamadas_soporte_6m), 'nps': float(nps), 'descuento_activo': descuento_activo,
            'meses_sin_reajuste': float(meses_sin_reajuste), 'ingreso_estimado_clp': float(ingreso_estimado_clp),
            'cambios_plan_12m': float(cambios_plan_12m)
        }
        X_input = pd.DataFrame([input_dict])

        num_cols_originales = imputer_num.feature_names_in_.tolist()
        cat_cols_originales = imputer_cat.feature_names_in_.tolist()

        X_num_imp = imputer_num.transform(X_input[num_cols_originales])
        X_cat_imp = imputer_cat.transform(X_input[cat_cols_originales])
        X_cat_enc = encoder.transform(X_cat_imp)

        ratio_factura_ingreso = float(factura_mensual_clp) / (float(ingreso_estimado_clp) + 1.0)
        indice_conflictividad = float(reclamos_12m) + float(llamadas_soporte_6m)
        
        derivadas = np.array([[ratio_factura_ingreso, indice_conflictividad]])
        X_num_total = np.hstack((X_num_imp, derivadas))
        features_procesadas = np.hstack((X_num_total, X_cat_enc))

        probabilidad = model.predict_proba(features_procesadas)[0][1]
        prob_porcentaje = probabilidad * 100

        st.write("---")
        st.subheader("📈 Resultado del Análisis de Riesgo")
        
        if probabilidad >= 0.60:
            st.error(f"⚠️ **Riesgo CRÍTICO / ALTO:** El cliente presenta un **{prob_porcentaje:.2f}%** de probabilidad de caer en mora severa.")
            st.markdown("**Acción recomendada:** Gatillar campaña de cobranza preventiva inmediata.")
        elif probabilidad >= 0.30:
            st.warning(f"🟡 **Riesgo MEDIO:** El cliente presenta un **{prob_porcentaje:.2f}%** de probabilidad de retraso.")
            st.markdown("**Acción recomendada:** Monitorear comportamiento de pago.")
        else:
            st.success(f"🟢 **Riesgo BAJO:** El cliente se mantiene estable con un **{prob_porcentaje:.2f}%** de probabilidad de mora.")
    except Exception as error_ejecucion:
        st.error(f"🚨 Error en procesamiento: {error_ejecucion}")
