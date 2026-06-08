import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Cargar los objetos entrenados de forma segura
try:
    model = joblib.load('model.joblib')
    imputer_num = joblib.load('imputer_num.joblib')
    imputer_cat = joblib.load('imputer_cat.joblib')
    encoder = joblib.load('encoder.joblib')
except Exception as e:
    st.error(f"Error al cargar los archivos .joblib: {e}")

st.title("📊 Sistema Predictivo de Morosidad — ConecTel")
st.markdown("Ingrese los datos del cliente para evaluar el riesgo de morosidad severa (90+ días).")

# 2. Formulario de entrada de datos basados en el dataset original
with st.form("form_prediccion"):
    col1, col2 = st.columns(2)
    
    with col1:
        edad = st.number_input("Edad", min_value=18, max_value=100, value=40)
        antiguedad_meses = st.number_input("Antigüedad (Meses)", min_value=0, value=24)
        factura_mensual_clp = st.number_input("Factura Mensual (CLP)", min_value=0, value=25000)
        ingreso_estimado_clp = st.number_input("Ingreso Estimado (CLP)", min_value=0, value=600000)
        dias_mora_hist = st.number_input("Días de Mora Históricos", min_value=0, value=5)
        reclamos_12m = st.number_input("Reclamos (Últimos 12 meses)", min_value=0, value=1)
        llamadas_soporte_6m = st.number_input("Llamadas a Soporte (Últimos 6 meses)", min_value=0, value=2)
        nps = st.slider("Nota NPS (Satisfacción)", min_value=1, max_value=10, value=8)
        meses_sin_reajuste = st.number_input("Meses sin Reajuste de Plan", min_value=0, value=6)
        cambios_plan_12m = st.number_input("Cambios de Plan (Últimos 12 meses)", min_value=0, value=0)

    with col2:
        region = st.selectbox("Región", ['Metropolitana', 'Valparaíso', 'Biobío', 'Antofagasta', 'Otros'])
        genero = st.selectbox("Género", ['Masculino', 'Femenino', 'Otro'])
        tipo_contrato = st.selectbox("Tipo de Contrato", ['Mensual', 'Anual', 'Dos Años'])
        plan = st.selectbox("Plan Contratado", ['Básico', 'Intermedio', 'Premium'])
        metodo_pago = st.selectbox("Método de Pago", ['Transferencia', 'Tarjeta de Crédito', 'PAC', 'Efectivo'])
        descuento_activo = st.selectbox("¿Tiene Descuento Activo?", ['Sí', 'No'])
        
        tiene_internet = st.selectbox("¿Tiene Internet?", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
        velocidad_mbps = st.number_input("Velocidad de Internet (Mbps)", min_value=0, value=300)
        tiene_tv = st.selectbox("¿Tiene TV por Cable?", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
        tiene_linea_movil = st.selectbox("¿Tiene Línea Móvil?", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
        num_servicios = st.number_input("Número de Servicios Contratados", min_value=1, max_value=5, value=2)

    btn_predecir = st.form_submit_button("Evaluar Riesgo de Cliente")

# 3. Procesamiento y Predicción al hacer clic
if btn_predecir:
    # Agrupamos las columnas respetando estrictamente el orden que espera el imputer numérico
    num_cols = ['edad', 'antiguedad_meses', 'tiene_internet', 'velocidad_mbps', 'tiene_tv', 
                'tiene_linea_movil', 'num_servicios', 'factura_mensual_clp', 'dias_mora_hist', 
                'reclamos_12m', 'llamadas_soporte_6m', 'nps', 'meses_sin_reajuste', 
                'ingreso_estimado_clp', 'cambios_plan_12m']
    
    cat_cols = ['region', 'genero', 'tipo_contrato', 'plan', 'metodo_pago', 'descuento_activo']

    # Crear diccionarios con los valores capturados
    dict_num = {
        'edad': edad, 'antiguedad_meses': antiguedad_meses, 'tiene_internet': tiene_internet,
        'velocidad_mbps': velocidad_mbps, 'tiene_tv': tiene_tv, 'tiene_linea_movil': tiene_linea_movil,
        'num_servicios': num_servicios, 'factura_mensual_clp': factura_mensual_clp, 'dias_mora_hist': dias_mora_hist,
        'reclamos_12m': reclamos_12m, 'llamadas_soporte_6m': llamadas_soporte_6m, 'nps': nps,
        'meses_sin_reajuste': meses_sin_reajuste, 'ingreso_estimado_clp': ingreso_estimado_clp, 'cambios_plan_12m': cambios_plan_12m
    }
    
    dict_cat = {
        'region': region, 'genero': genero, 'tipo_contrato': tipo_contrato,
        'plan': plan, 'metodo_pago': metodo_pago, 'descuento_activo': descuento_activo
    }

    # Convertir a DataFrames de una sola fila
    df_num = pd.DataFrame([dict_num], columns=num_cols)
    df_cat = pd.DataFrame([dict_cat], columns=cat_cols)

    # Aplicar transformaciones usando los objetos cargados (.joblib)
    X_num_imp = imputer_num.transform(df_num)
    X_cat_imp = imputer_cat.transform(df_cat)
    X_cat_enc = encoder.transform(X_cat_imp)

    # Calcular las variables derivadas de forma posicional e idéntica al entrenamiento
    idx_factura = num_cols.index('factura_mensual_clp')
    idx_ingreso = num_cols.index('ingreso_estimado_clp')
    idx_reclamos = num_cols.index('reclamos_12m')
    idx_soporte = num_cols.index('llamadas_soporte_6m')

    ratio_factura_ingreso = X_num_imp[:, idx_factura] / (X_num_imp[:, idx_ingreso] + 1)
    indice_conflictividad = X_num_imp[:, idx_reclamos] + X_num_imp[:, idx_soporte]

    # Unir el bloque numérico con sus derivadas
    X_num_final = np.hstack((X_num_imp, ratio_factura_ingreso.reshape(-1, 1), indice_conflictividad.reshape(-1, 1)))

    # Concatenar todo para ingresar al modelo
    X_final = np.hstack((X_num_final, X_cat_enc))

    # Asegurar dimensionalidad recortando o adaptando si sobra algún elemento fantasma
    if X_final.shape[1] > model.n_features_in_:
        X_final = X_final[:, :model.n_features_in_]

    # Ejecutar la predicción
    prediccion = model.predict(X_final)[0]
    probabilidad = model.predict_proba(X_final)[0][1]

    # Mostrar resultados en la interfaz
    st.subheader("Resultado del Análisis:")
    if prediccion == 1:
        st.error(f"🚨 **ALTO RIESGO DE MOROSIDAD**. Probabilidad: {probabilidad:.1%}")
        st.markdown("**Recomendación Institucional:** Activar protocolo preventivo de cobranza, ofrecer opciones de flexibilización de planes o canales prioritarios de atención para reducir el índice de conflictividad.")
    else:
        st.success(f"✅ **CLIENTE DE BAJO RIESGO**. Probabilidad de mora: {probabilidad:.1%}")
        st.markdown("**Recomendación Comercial:** Cliente apto para campañas de fidelización, ofertas de up-selling o renovación anticipada de contratos.")
