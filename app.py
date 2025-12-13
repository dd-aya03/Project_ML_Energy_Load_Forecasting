import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import xgboost as xgb
from sklearn.tree import plot_tree

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Détection Consommation Électrique", layout="centered")

st.title("⚡ Prédiction de Consommation Électrique (Maroc)")
st.markdown("### Dashboard de gestion énergétique intelligent")
st.markdown("Ce système utilise l'IA pour anticiper les pics de consommation et protéger le réseau.")

# --- 1. FONCTIONS UTILITAIRES ---

def load_model_data(model_path, col_path):
    try:
        model = joblib.load(model_path)
        cols = joblib.load(col_path)
        return model, cols
    except FileNotFoundError:
        return None, None

def get_decision(value):
    # Seuils basés sur l'analyse statistique (Médiane et Top 10%)
    SEUIL_ECO = 69788.79
    SEUIL_CRITIQUE = 94912.14

    if value < SEUIL_ECO:
        return f"🌱 **VERTE (< {SEUIL_ECO:,.0f} MW)**\n\nSituation excellente."
    elif SEUIL_ECO <= value < SEUIL_CRITIQUE:
        return f"⚡ **NORMALE**\n\nLe réseau est stable."
    else:
        return f"⚠️ **ALERTE PIC (> {SEUIL_CRITIQUE:,.0f} MW)**\n\nRisque de saturation !"

def clean_data_for_graph(data):
    # Cette fonction prépare les données historiques pour le graphique
    df = data.copy()
    df.columns = df.columns.str.strip()
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed', dayfirst=True)
    
    df['Total_Consumption'] = df['Zone 1'] + df['Zone 2'] + df['Zone 3']
    
    # Extraction de base
    df['Hour'] = df['DateTime'].dt.hour
    df['Month'] = df['DateTime'].dt.month
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['DayOfMonth'] = df['DateTime'].dt.day
    df['Year'] = df['DateTime'].dt.year 
    
    # --- ENGINEERING POUR LA RÉGRESSION LINÉAIRE ---
    # On recrée les features mathématiques pour le graphique
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Temp_Squared'] = df['Temperature'] ** 2
    
    # Pour le graphique, le "Lag" est la vraie consommation de l'heure d'avant
    df['Total_Lag1'] = df['Total_Consumption'].shift(1)
    
    # Nettoyage
    cols_to_numeric = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True) # Pour le premier Lag qui est NaN
    return df

def plot_feature_importance(model, feature_names, model_name, color_palette):
    st.subheader(f"📊 Analyse des Variables - {model_name}")
    
    if hasattr(model, 'feature_importances_'):
        # Arbres (XGB, RF)
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette=color_palette, ax=ax)
        st.pyplot(fig)
        
    elif hasattr(model, 'coef_'):
        # Régression Linéaire
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.coef_})
        fi_df['Abs_Importance'] = fi_df['Importance'].abs()
        fi_df = fi_df.sort_values(by='Abs_Importance', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette="Oranges_r", ax=ax)
        st.pyplot(fig)

# --- 2. SIDEBAR : PARAMÈTRES ---
st.sidebar.header("1. Paramètres Météo & Temps")

date_input = st.sidebar.date_input("Date", datetime.now())
time_input = st.sidebar.time_input("Heure", datetime.now())
temp = st.sidebar.slider("Température (°C)", -5.0, 50.0, 20.0, 0.1)
humidity = st.sidebar.slider("Humidité (%)", 0, 100, 60, 1)
wind = st.sidebar.number_input("Vitesse du vent", 0.0, 10.0, 0.08, 0.01)
gen_diffuse = st.sidebar.number_input("Flux Diffus Général", value=0.05, format="%.3f")
diffuse = st.sidebar.number_input("Flux Diffus", value=0.10, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.header("2. Choix du Modèle")
choix_modele = st.sidebar.radio(
    "Modèle actif :",
    ("XGBoost", "Random Forest", "Régression Linéaire (avec Lag)", "Comparaison (Tous)")
)

# --- GESTION INTELLIGENTE DU LAG ---
# On affiche la case "Lag" seulement si la Régression est concernée
lag_value = 0.0
if choix_modele in ["Régression Linéaire (avec Lag)", "Comparaison (Tous)"]:
    st.sidebar.markdown("---")
    st.sidebar.warning("ℹ️ La Régression Linéaire a besoin du passé.")
    lag_value = st.sidebar.number_input("Consommation Heure Précédente (MW)", value=65000.0, step=100.0)

# --- 3. PRÉDICTION PRINCIPALE ---
st.info("💡 Modifiez les paramètres à gauche pour simuler une situation.")

launch = st.button("Lancer la Prédiction 🚀", use_container_width=True)

if launch:
    full_date = datetime.combine(date_input, time_input)
    
    # 1. Création du DataFrame de base
    input_data = {
        'Temperature': [temp], 'Humidity': [humidity], 'Wind Speed': [wind],
        'general diffuse flows': [gen_diffuse], 'diffuse flows': [diffuse],
        'Hour': [full_date.hour], 'Month': [full_date.month],
        'DayOfWeek': [full_date.weekday()], 'DayOfMonth': [full_date.day], 'Year': [full_date.year],
        
        # Features Mathématiques (Calculés à la volée pour la Régression)
        'Hour_Sin': [np.sin(2 * np.pi * full_date.hour / 24)],
        'Hour_Cos': [np.cos(2 * np.pi * full_date.hour / 24)],
        'Temp_Squared': [temp ** 2],
        'Total_Lag1': [lag_value] # La valeur entrée par l'utilisateur
    }
    df_input = pd.DataFrame(input_data)

    # Préparation des colonnes d'affichage
    cols = st.columns(3) if choix_modele == "Comparaison (Tous)" else st.columns(1)

    # --- MODÈLE 1 : XGBOOST ---
    if choix_modele in ["XGBoost", "Comparaison (Tous)"]:
        model_xgb, cols_xgb = load_model_data('models\energy_xgboost_model.pkl', 'models\model_columns_XGB.pkl')
        if model_xgb:
            try:
                # On filtre df_input pour ne garder que les colonnes connues par XGBoost
                # (XGBoost ne connait pas le Lag ni Temp_Squared si on ne lui a pas appris)
                valid_cols = [c for c in cols_xgb if c in df_input.columns]
                res_xgb = model_xgb.predict(df_input[valid_cols])[0]
                
                with cols[0] if choix_modele == "Comparaison (Tous)" else cols[0]:
                    st.success(f"### XGBoost\n{res_xgb:,.2f} MW")
                    st.markdown(get_decision(res_xgb))
            except Exception as e:
                st.error(f"Erreur XGB: {e}")

    # --- MODÈLE 2 : RANDOM FOREST ---
    if choix_modele in ["Random Forest", "Comparaison (Tous)"]:
        model_rf, cols_rf = load_model_data('models\model_consommation_maroc.pkl', 'models\model_columns_forest.pkl')
        if model_rf:
            try:
                valid_cols = [c for c in cols_rf if c in df_input.columns]
                res_rf = model_rf.predict(df_input[valid_cols])[0]
                
                idx = 1 if choix_modele == "Comparaison (Tous)" else 0
                with cols[idx]:
                    st.info(f"### Random Forest\n{res_rf:,.2f} MW")
                    st.markdown(get_decision(res_rf))
            except Exception as e:
                st.error(f"Erreur RF: {e}")

    # --- MODÈLE 3 : RÉGRESSION LINÉAIRE (AVEC LAG) ---
    if choix_modele in ["Régression Linéaire (avec Lag)", "Comparaison (Tous)"]:
        model_lr, cols_lr = load_model_data('models\modele_regression_lineaire.pkl', 'models\model_columns_regression.pkl')
        if model_lr:
            try:
                # Ici, on a besoin de toutes les features mathématiques et du Lag
                res_lr = model_lr.predict(df_input[cols_lr])[0]
                
                idx = 2 if choix_modele == "Comparaison (Tous)" else 0
                with cols[idx]:
                    st.warning(f"### Régression Linéaire\n{res_lr:,.2f} MW")
                    st.markdown(get_decision(res_lr))
                    st.caption(f"(Avec Lag: {lag_value:,.0f} MW)")
            except Exception as e:
                st.error(f"Erreur LR. Vérifiez que 'model_columns_regression.pkl' correspond au modèle.")

# --- 4. VISUALISATIONS AVANCÉES ---
st.markdown("---")
st.header("📈 Analyse et Transparence")

tab1, tab2, tab3 = st.tabs(["Comparaison Courbes", "Importance Variables", "Structure Arbres"])

# --- TAB 1 : GRAPHIQUES ---
with tab1:
    st.caption("Comparaison sur les données historiques (morocco.csv).")
    if st.checkbox("Charger les courbes", key="graph_btn"):
        try:
            df_raw = pd.read_csv('morocco.csv')
            sample_size = st.slider("Nombre d'heures à afficher", 50, 500, 150)
            df_sample = df_raw.tail(sample_size).copy()
            
            # Nettoyage et calcul des Lags automatiques pour le graphe
            df_clean = clean_data_for_graph(df_sample)
            
            y_true = df_clean['Total_Consumption']
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(y_true.values, label='Réel', color='#1f77b4', linewidth=2)
            
            # Tracé XGBoost
            if choix_modele in ["XGBoost", "Comparaison (Tous)"]:
                 m, c = load_model_data('models\energy_xgboost_model.pkl', 'models\model_columns_XGB.pkl')
                 if m: ax.plot(m.predict(df_clean[[col for col in c if col in df_clean.columns]]), label='XGBoost', color='red', linestyle='--')

            # Tracé Random Forest
            if choix_modele in ["Random Forest", "Comparaison (Tous)"]:
                 m, c = load_model_data('models\model_consommation_maroc.pkl', 'models\model_columns_forest.pkl')
                 if m: ax.plot(m.predict(df_clean[[col for col in c if col in df_clean.columns]]), label='Random Forest', color='green', linestyle='-.')

            # Tracé Régression Linéaire
            if choix_modele in ["Régression Linéaire (avec Lag)", "Comparaison (Tous)"]:
                 m, c = load_model_data('models\modele_regression_lineaire.pkl', 'models\model_columns_regression.pkl')
                 if m: ax.plot(m.predict(df_clean[c]), label='Régression (Lag)', color='orange', linestyle=':')

            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur graphique : {e}")

# --- TAB 2 : IMPORTANCE ---
with tab2:
    if choix_modele == "Comparaison (Tous)":
        c1, c2, c3 = st.columns(3)
        with c1:
            m, c = load_model_data('models\energy_xgboost_model.pkl', 'models\model_columns_XGB.pkl')
            if m: plot_feature_importance(m, c, "XGBoost", "Reds_r")
        with c2:
            m, c = load_model_data('models\model_consommation_maroc.pkl', 'models\model_columns_forest.pkl')
            if m: plot_feature_importance(m, c, "Random Forest", "Greens_r")
        with c3:
            m, c = load_model_data('models\modele_regression_lineaire.pkl', 'models\model_columns_regression.pkl')
            if m: plot_feature_importance(m, c, "Régression", "Oranges_r")
    else:
        # Affichage individuel
        if choix_modele == "XGBoost":
             m, c = load_model_data('models\energy_xgboost_model.pkl', 'models\model_columns_XGB.pkl')
             if m: plot_feature_importance(m, c, "XGBoost", "Reds_r")
        elif choix_modele == "Régression Linéaire (avec Lag)":
             m, c = load_model_data('models\modele_regression_lineaire.pkl', 'models\model_columns_regression.pkl')
             if m: plot_feature_importance(m, c, "Régression", "Oranges_r")
        # ... Random Forest ...

# --- TAB 3 : STRUCTURE ---
with tab3:
    if choix_modele == "XGBoost" or choix_modele == "Comparaison (Tous)":
        m, _ = load_model_data('models\energy_xgboost_model.pkl', 'models\model_columns_XGB.pkl')
        if m:
            st.subheader("🌳 Arbre XGBoost")
            fig, ax = plt.subplots(figsize=(15, 8))
            xgb.plot_tree(m, num_trees=0, max_depth=3, ax=ax)
            st.pyplot(fig)
            
    if choix_modele == "Régression Linéaire (avec Lag)":
        st.info("La régression linéaire est une équation mathématique, pas un arbre.")

st.markdown("---")
st.caption("Projet Data Science - ENSIA")