# streamlit_app.py

import streamlit as st
import requests

# Fonction pour obtenir des informations sur un client à partir de l'API
def get_client_info(selected_client):
    api_url = f"http://localhost:80/client_info/{selected_client}"  # Assurez-vous que l'URL est correcte
    response = requests.get(api_url)

    if response.status_code == 200:
        client_info = response.json()
        return client_info
    else:
        return None

st.title("Informations sur les clients")

# Chargement du fichier CSV et affichage de la liste des clients disponibles
df = pd.read_csv('X_train.csv')
client_list = df['SK_ID_CURR'].tolist()

# Sélectionnez un numéro de client à l'aide d'un widget de sélection
selected_client = st.selectbox("Sélectionnez un numéro de client :", client_list)

# Affichez les informations du client sélectionné
if selected_client is not None:
    client_data = get_client_info(selected_client)
    if client_data is not None:
        st.write("Informations du client sélectionné :")
        st.write(client_data)
    else:
        st.warning("Aucune information trouvée pour ce client.")
else:
    st.info("Sélectionnez un numéro de client dans la liste ci-dessus.")
