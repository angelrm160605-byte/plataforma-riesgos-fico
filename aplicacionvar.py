import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Risk Management Pro", layout="wide")
st.title("🏦 Plataforma Avanzada de Riesgos FICO")

# --- PESTAÑAS PARA SEPARAR ACTIVOS ---
tab1, tab2, tab3 = st.tabs(["📈 Cartera General (Acciones/Divisas)", "🏛️ Módulo Bonos", "📉 Módulo Opciones"])

# ==========================================
# PESTAÑA 1: CARTERA GENERAL Y MONTECARLO
# ==========================================
with tab1:
    st.write("Cálculo Paramétrico y Montecarlo usando todo el histórico disponible.")
    
    col1, col2 = st.columns(2)
    with col1:
        tickers_input = st.text_input("Tickers (ej. SAN.MC, AAPL, EURUSD=X)", "SAN.MC, IBE.MC")
    with col2:
        pesos_input = st.text_input("Pesos (ej. 0.5, 0.5)", "0.5, 0.5")
        
    inversion_inicial = st.number_input("Inversión Inicial (€)", value=100000)

    if st.button("Ejecutar Motor de Riesgos"):
        try:
            tickers = [t.strip() for t in tickers_input.split(',')]
            pesos = np.array([float(p.strip()) for p in pesos_input.split(',')])
            
            st.info("Descargando TODO el histórico disponible...")
            # Descargamos periodo MÁXIMO
            datos = yf.download(tickers, period="max")['Close']
            
            # Asegurarnos de que el orden de columnas es correcto
            datos = datos[tickers]
            rendimientos = datos.pct_change().dropna()
            
            # Rendimiento de la cartera combinada
            rend_cartera = rendimientos.dot(pesos)
            
            # 1. VAR PARAMÉTRICO (Asumiendo Normalidad)
            media = np.mean(rend_cartera)
            desviacion = np.std(rend_cartera)
            z_95 = norm.ppf(0.05) # Nivel de confianza 95% (-1.645)
            
            var_param_1d_pct = abs(media + z_95 * desviacion)
            var_param_1d_eur = inversion_inicial * var_param_1d_pct
            
            # 2. SIMULACIÓN MONTECARLO (10,000 caminos)
            simulaciones = 10000
            escenarios_1d = np.random.normal(media, desviacion, simulaciones)
            var_mc_1d_pct = abs(np.percentile(escenarios_1d, 5))
            var_mc_1d_eur = inversion_inicial * var_mc_1d_pct

            st.subheader("Resultados del Riesgo (95% Confianza)")
            
            # Tabla comparativa de horizontes temporales
            horizontes = [1, 7, 30]
            resultados = []
            
            for t in horizontes:
                # Escalamos por la raíz del tiempo (T^0.5)
                escalar = np.sqrt(t)
                resultados.append({
                    "Horizonte": f"{t} Día(s)",
                    "VaR Paramétrico (€)": f"€ {var_param_1d_eur * escalar:,.2f}",
                    "VaR Montecarlo (€)": f"€ {var_mc_1d_eur * escalar:,.2f}"
                })
                
            st.table(pd.DataFrame(resultados))
        except Exception as e:
            st.error(f"Error: Comprueba que has puesto el mismo número de tickers y pesos. ({e})")

# ==========================================
# PESTAÑA 2: MÓDULO DE BONOS (Duración Modificada)
# ==========================================
with tab2:
    st.write("Cálculo de VaR para renta fija basado en la sensibilidad a tipos de interés.")
    
    col3, col4 = st.columns(2)
    with col3:
        # Pongo por defecto los datos de tu PDF (Página 8)
        posicion_bono = st.number_input("Posición en el Bono (€)", value=6000000)
        duracion_mod = st.number_input("Duración Modificada (Dm)", value=5.2)
    with col4:
        tir_y = st.number_input("Nivel actual de TIR (y) en decimal (ej. 0.0483)", value=0.0483)
        vol_tir = st.number_input("Volatilidad de la TIR (Δy/y) diaria", value=0.008)
        
    if st.button("Calcular VaR del Bono"):
        # Fórmula: Vol_precio = Dm * TIR * Vol(TIR)
        vol_precio_bono = duracion_mod * tir_y * vol_tir
        
        # Z al 99% para el ejercicio de tu PDF es 2.33
        z_99 = 2.33 
        var_bono_1d = posicion_bono * vol_precio_bono * z_99
        
        st.success(f"VaR del Bono (1 día, 99%): € {var_bono_1d:,.2f}")
        st.info("Fórmula aplicada: Posición * Dm * y * σ(Δy/y) * Z")

# ==========================================
# PESTAÑA 3: MÓDULO DE OPCIONES (Delta-Gamma)
# ==========================================
with tab3:
    st.write("Aproximación no lineal Delta-Gamma para derivados.")
    
    col5, col6 = st.columns(2)
    with col5:
        # Datos por defecto del PDF (Página 19)
        var_subyacente = st.number_input("VaR del Activo Subyacente (€)", value=6.2423)
        delta = st.number_input("Delta de la opción (Δ)", value=0.618)
    with col6:
        gamma = st.number_input("Gamma de la opción (Γ)", value=0.0095)
        
    if st.button("Calcular VaR de la Opción"):
        # Fórmula Delta-Gamma
        var_delta = abs(delta) * var_subyacente
        ajuste_gamma = 0.5 * gamma * (var_subyacente ** 2)
        var_delta_gamma = var_delta - ajuste_gamma
        
        st.metric(label="VaR Lineal (Solo Delta)", value=f"€ {var_delta:,.4f}")
        st.metric(label="VaR Real (Delta-Gamma)", value=f"€ {var_delta_gamma:,.4f}")
        st.info("Las posiciones largas (Gamma > 0) disminuyen el VaR respecto al modelo lineal.")
