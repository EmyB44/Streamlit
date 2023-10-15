# streamlit_app.py


import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go


df = pd.read_csv('X_train.csv')

# Titre information sur les clients
st.title("Informations sur les clients")

# Sélectionnez un numéro de client à l'aide d'un widget de sélection
selected_client = st.selectbox("Sélectionnez un numéro de client :", df['SK_ID_CURR'])

if st.button("Prédire"):
    if selected_client:
        # Effectuer une requête à l'API Flask pour obtenir les prédictions
        response = requests.get(f"https://api-projet-7-emilie-brosseau.onrender.com/predict/130972")

        if response.status_code == 200:
            predictions = response.json()
            st.write("Prédictions :")
            st.json(predictions)
        else:
            st.error("Erreur lors de la récupération des prédictions.")
    else:
        st.warning("Sélectionnez un client avant de prédire.")





# Visualisation remboursement pret
st.title("Prédictions remboursements sur les clients")



selected_client = st.selectbox("Sélectionnez un numéro de client :", df['SK_ID_CURR'])

if st.button("Prédire"):
    if selected_client:
        # Effectuer une requête à l'API Flask pour obtenir les prédictions de probabilité
        response = requests.get(f"https://api-projet-7-emilie-brosseau.onrender.com/predict_proba/{selected_client}")

        if response.status_code == 200:
            predictions_proba = response.json()
            pourcentage_credit = predictions_proba[0][1] * 100  # Obtenez la probabilité de crédit

            # Création de la figure de la jauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pourcentage_credit,
                title={'text': "Clients avec Crédit"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': 'darkblue'},
                    'steps': [
                        {'range': [0, 50], 'color': 'red'},
                        {'range': [50, 100], 'color': 'green'}
                    ],
                }
            ))

            # Affichage de la figure dans Streamlit
            st.plotly_chart(fig)

            st.write("Probabilité de crédit : {:.2f}%".format(pourcentage_credit))
        else:
            st.error("Erreur lors de la récupération des prédictions de probabilité.")
    else:
        st.warning("Sélectionnez un client avant de prédire.")








