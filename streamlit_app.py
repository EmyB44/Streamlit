import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import json

# Charger les données des clients depuis un fichier CSV
df = pd.read_csv('X_train.csv')

# Sélectionnez un numéro de client à l'aide d'un widget de sélection
df_sorted = df.sort_values(by='SK_ID_CURR')

# Sélectionnez un numéro de client à l'aide d'un widget de sélection
selected_client = st.selectbox("Sélectionnez un numéro de client :", df_sorted['SK_ID_CURR'])


# Titre information sur les clients
st.title("Informations sur les clients")

# Bouton pour prédire
if st.button("Prédire"):
    if selected_client:
        # Effectuer une requête à l'API Flask pour obtenir les prédictions
        response = requests.get(f"https://api-projet-7-emilie-brosseau.onrender.com/predict/{selected_client}")

        if response.status_code == 200:
            predictions = response.json()

            # Vérifiez la prédiction
            if predictions == [0]:
                predictions = "Aucun risque de défaut"
            elif predictions == [1]:
                predictions = "Risque de défaut"

            # Affichez les prédictions
            st.write("Prédictions :")
            st.write(predictions)
        else:
            st.error("Erreur lors de la récupération des prédictions.")
    else:
        st.warning("Sélectionnez un client avant de prédire.")

# Visualisation remboursement prêt
st.title("Prédictions risque de défaut de remboursement")

# Bouton pour prédire la probabilité de crédit
if st.button("Prédire le risque de non remboursement de crédit"):
    if selected_client:
        # Effectuer une requête à l'API Flask pour obtenir les prédictions de probabilité
        response = requests.get(f"https://api-projet-7-emilie-brosseau.onrender.com/predict_proba/{selected_client}")

        if response.status_code == 200:
            predictions_proba = response.json()
            pourcentage_credit = predictions_proba[0][1] * 100  # Obtenir la probabilité de crédit

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

            st.write("Probabilité de défaut de remboursement : {:.2f}%".format(pourcentage_credit))
        else:
            st.error("Erreur lors de la récupération des prédictions de probabilité de défaut de remboursement.")
    else:
        st.warning("Sélectionnez un client avant de prédire la probabilité de défaut de remboursement.")

## Titre
st.title("Features importances sur les clients")



# Bouton pour afficher le graphique SHAP
if st.button("Afficher le graphique SHAP") and selected_client:
    # Effectuer une requête à l'API pour obtenir les valeurs SHAP
    response = requests.get(f"https://api-projet-7-emilie-brosseau.onrender.com/prediction_feature_importance/{selected_client}")

    if response.status_code == 200:
        # Récupérer le graphique SHAP au format JSON depuis l'API
        shap_chart_json = response.json()

        # Créer un graphique à partir des données JSON
        if shap_chart_json:
            shap_values = {k: v for k, v in shap_chart_json.items()}
            shap_df = pd.DataFrame(shap_values.items(), columns=['Feature', 'SHAP Value'])

            # Trier le DataFrame par SHAP values
            shap_df = shap_df.sort_values(by='SHAP Value', ascending=False)

            # Limiter aux 20 fonctionnalités les plus importantes
            shap_df_top20 = shap_df.head(20)

            # Créer un graphique à barres
            plt.figure(figsize=(10, 6))
            plt.barh(shap_df_top20['Feature'], shap_df_top20['SHAP Value'])
            plt.xlabel('SHAP Value')
            plt.ylabel('Feature')
            plt.title('Graphique SHAP des 20 fonctionnalités les plus importantes')

            # Afficher le graphique dans Streamlit
            st.pyplot(plt)
        else:
            st.error("Les données SHAP reçues de l'API sont vides ou mal formatées.")
    else:
        st.error("Erreur lors de la récupération du graphique SHAP depuis l'API.")
elif not selected_client:
    st.warning("Sélectionnez un client avant d'afficher le graphique SHAP.")