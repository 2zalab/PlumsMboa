import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import requests
from streamlit_image_comparison import image_comparison
from src.data_utils import load_data, get_class_distribution
from src.visualization import plot_confusion_matrix, plot_training_history
from src.predict import preprocess_image, predict_image
import streamlit.components.v1 as components

# Import des pages depuis le dossier pages
from modules.home import render_homepage
from modules.data import render_data_page
from modules.model import render_model_page
from modules.classification import render_classification_page
from modules.about import render_about_page

# Configuration de la page
st.set_page_config(
    page_title="PlumsMboa - Tri Automatique des Prunes",
    page_icon= "assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement du fichier CSS personnalisé
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("assets/css/style.css")
except:
    st.warning("Fichier CSS non trouvé. L'interface utilisera le style par défaut.")

# Fonction pour charger des animations Lottie
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Fonction pour charger le modèle
@st.cache_resource
def load_classification_model():
    try:
        # Vérifier plusieurs chemins possibles
        possible_model_paths = [
           # "models/plum_classifier_final.h5",
            "models/plum_classifier_best.h5",
            "models/last_run/final_weights.h5",
            "models/last_run/final_model",
            # Ajoutez d'autres chemins possibles ici
        ]
        
        # Parcourir les chemins possibles
        for model_path in possible_model_paths:
            if os.path.exists(model_path):
                st.success(f"Modèle trouvé à: {model_path}")
                
                # Si c'est un dossier SavedModel
                if os.path.isdir(model_path):
                    model = tf.keras.models.load_model(model_path)
                    return model
                    
                # Si ce sont des poids (.h5)
                elif model_path.endswith(".h5"):
                    # S'il s'agit d'un fichier de poids uniquement
                    if "weights" in model_path:
                        # Recréer le modèle et charger les poids
                        from src.model import create_efficientnet_model
                        model, _ = create_efficientnet_model(num_classes=6)
                        model.load_weights(model_path)
                        return model
                    else:
                        # Charger le modèle complet
                        model = tf.keras.models.load_model(model_path)
                        return model
                        
        # Si aucun modèle n'est trouvé, utiliser un modèle simulé
        st.warning("Aucun modèle trouvé, utilisation de prédictions simulées.")
        return "SIMULATION"
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        st.warning("Utilisation de prédictions simulées à la place.")
        return "SIMULATION"

# Fonction pour charger les données
@st.cache_data
def load_csv_data():
    try:
        return pd.read_csv("data/plums_data.csv")
    except:
        st.error("Fichier CSV non trouvé. Veuillez exécuter le script de téléchargement des données.")
        return None

# Menu de navigation
with st.sidebar:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
            st.image("assets/logo.png", width=200)
            #st.image("https://img.icons8.com/color/240/000000/plum.png", width=150)
            st.markdown("## PlumsMboa")
    st.markdown("Tri Automatique des Prunes avec Intelligence Artificielle")
    
   # Méthode de création du menu avec style personnalisé
    # Style CSS personnalisé pour le menu
    st.markdown("""
    <style>
    .stRadio > div {
        display: flex;
        flex-direction: column;
    }
    
    .stRadio > div > label {
        margin-bottom: 10px;
        padding: 10px 15px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stRadio > div > label:hover {
        background-color: rgba(93, 63, 211, 0.1);
    }
    
    .stRadio > div > label[data-selected="true"] {
        background-color: #903e97;
        color: white !important;
        font-weight: bold;
    }
    
    .stRadio > div > label > div > div > div {
        color: inherit !important;
    }
    
    /* Icône */
    .stRadio > div > label > div > div > div > div {
        color: inherit !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Menu de navigation avec style personnalisé
    selected = option_menu(
        menu_title=None,
        options=["Accueil", "Données", "Modèle", "Classification", "À propos"],
        icons=["house", "database", "cpu", "image", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "inherit", "font-size": "20px"}, 
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "rgba(93, 63, 211, 0.1)",
            },
            "nav-link-selected": {
                "background-color": "#903e97",
                "color": "white",
                "font-weight": "bold"
            },
        }
    )
    
    st.markdown("---")
    st.markdown("### Équipe de MIT")
    st.markdown("Hackathon JCIA 2025")

# Rendu de la page sélectionnée
if selected == "Accueil":
    render_homepage()
elif selected == "Données":
    render_data_page(load_csv_data())
elif selected == "Modèle":
    render_model_page()
elif selected == "Classification":
    render_classification_page(load_classification_model())
else:  # À propos
    render_about_page()