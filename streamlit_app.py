# streamlit_app.py


import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import shap


df = pd.read_csv('X_train.csv')


# Sélectionnez un numéro de client à l'aide d'un widget de sélection
selected_client = st.selectbox("Sélectionnez un numéro de client :", df['SK_ID_CURR'])


# Titre information sur les clients
st.title("Informations sur les clients")



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


if st.button("Prédire"):
    if selected_client:
        # Effectuer une requête à l'API Flask pour obtenir les prédictions de probabilité
        response = requests.get(f"https://api-projet-7-emilie-brosseau.onrender.com/predict_proba/130972")

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





#Features les plus importantes
st.title("Features importances sur les clients")


if st.button("Afficher le graphique SHAP"):
    if selected_client:
        # Effectuer une requête à l'API pour obtenir les valeurs SHAP
        response = requests.get(f"https://api-projet-7-emilie-brosseau.onrender.com/client-info/130972")

        if response.status_code == 200:
            # Récupérer le graphique SHAP au format JSON depuis l'API
            shap_chart_json = response.json()

            # Afficher le graphique dans Streamlit en utilisant st.pyplot
            st.pyplot(generate_shap_chart(shap_chart_json))

        else:
            st.error("Erreur lors de la récupération du graphique SHAP depuis l'API.")
    else:
        st.warning("Sélectionnez un client avant d'afficher le graphique SHAP.")
