"""
Configuration de l'application PlumsMboa
"""

import os

# Chemins des dossiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Sous-dossiers de données
DATASET_DIR = os.path.join(DATA_DIR, "african_plums_dataset")
CSV_PATH = os.path.join(DATA_DIR, "plums_data.csv")

# Chemins des modèles
MODEL_PATH = os.path.join(MODELS_DIR, "plum_classifier.h5")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")

# Configuration du modèle
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6
LEARNING_RATE = 0.0001
EPOCHS = 50

# Catégories et descriptions
CATEGORIES = {
    'unaffected': 'Bonne qualité',
    'unripe': 'Non mûre',
    'spotted': 'Tachetée',
    'cracked': 'Fissurée',
    'bruised': 'Meurtrie',
    'rotten': 'Pourrie'
}

CATEGORY_COLORS = {
    'unaffected': '#3CB371',  # Vert
    'unripe': '#98FB98',      # Vert clair
    'spotted': '#FFD700',     # Jaune
    'cracked': '#FFA500',     # Orange
    'bruised': '#FF6347',     # Rouge-orangé
    'rotten': '#8B0000'       # Rouge foncé
}

DEFECT_DESCRIPTIONS = {
    'unaffected': "Prune de bonne qualité, prête pour la commercialisation.",
    'unripe': "Prune non mûre, besoin de plus de temps avant récolte.",
    'spotted': "Prune tachetée, peut être utilisée pour la transformation.",
    'cracked': "Prune fissurée, qualité réduite, à traiter rapidement.",
    'bruised': "Prune meurtrie, peut être utilisée pour certaines transformations.",
    'rotten': "Prune pourrie, à écarter de la chaîne de production."
}

# Informations sur le projet
PROJECT_INFO = {
    'name': 'PlumsMboa',
    'description': 'Tri Automatique des Prunes avec Intelligence Artificielle',
    'version': '1.0.0',
    'team': 'Équipe PlumsMboa',
    'hackathon': 'JCIA 2025',
    'github': 'https://github.com/plumsmboa/plumsmboa',
    'email': 'plumsmboa@example.com'
} 
