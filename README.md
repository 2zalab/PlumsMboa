# PlumsMboa - Tri Automatique des Prunes

![Logo PlumsMboa](assets/logo.png)

## 🌟 Présentation

PlumsMboa est une application de tri automatique des prunes africaines développée dans le cadre du Hackathon de la Journée Internationale de l'Intelligence Artificielle (JCIA) 2025. Cette application utilise des techniques avancées de vision par ordinateur et d'apprentissage profond pour classifier les prunes en six catégories selon leur qualité et leurs défauts.

## 🎯 Objectif du projet

Notre objectif est de développer un système de vision par ordinateur capable de classifier automatiquement les prunes africaines en six catégories, afin d'améliorer l'efficacité du tri, réduire les pertes post-récolte et augmenter la valeur ajoutée de la production fruitière au Cameroun.

## 🧠 Fonctionnalités

- **Classification d'images** : Prédiction de la catégorie d'une prune à partir de son image
- **Visualisation des données** : Exploration et visualisation du jeu de données
- **Analyse des performances** : Visualisation des métriques de performance du modèle
- **Interface utilisateur intuitive** : Application web développée avec Streamlit pour une utilisation facile

## 📊 Catégories de prunes

L'application peut classifier les prunes dans les catégories suivantes :

- ✅ **Bonne qualité** (Unaffected)
- 🟢 **Non mûre** (Unripe)
- 🟡 **Tachetée** (Spotted)
- 🟠 **Fissurée** (Cracked)
- 🔴 **Meurtrie** (Bruised)
- ⚫ **Pourrie** (Rotten)

## 🛠️ Technologies utilisées

- **Frontend** : Streamlit, HTML/CSS
- **Backend** : Python, TensorFlow, Keras
- **Traitement d'images** : OpenCV, Pillow
- **Visualisation** : Matplotlib, Plotly, Seaborn
- **Analyse de données** : Pandas, NumPy

## 🚀 Installation et exécution

### Prérequis

- Python 3.8+ installé
- Pip (gestionnaire de paquets Python)
- Git (pour cloner le dépôt)

### Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/plumsmboa/plumsmboa.git
cd plumsmboa
```

2. Créer un environnement virtuel :
```bash
# Windows
python -m venv plumsmboa-env
plumsmboa-env\Scripts\activate

# Linux/MacOS
python -m venv plumsmboa-env
source plumsmboa-env/bin/activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Télécharger les données :
```bash
python scripts/download_data.py
```

### Exécution

1. Lancer l'application Streamlit :
```bash
streamlit run app.py
```

2. Ouvrir votre navigateur à l'adresse indiquée (généralement http://localhost:8501)

## 📂 Structure du projet

```
PlumsMboa/
├── README.md                     # Documentation du projet
├── requirements.txt              # Dépendances du projet
├── .gitignore                    # Fichiers à ignorer pour Git
├── app.py                        # Application principale Streamlit
├── config.py                     # Configuration de l'application
├── data/                         # Dossier pour les données
│   └── plums_data.csv            # Fichier CSV avec les métadonnées
├── models/                       # Dossier pour les modèles entraînés
│   └── plum_classifier.h5        # Modèle sauvegardé
├── notebooks/                    # Notebooks pour l'exploration et l'entraînement
│   └── model_training.ipynb      # Notebook d'entraînement
├── src/                          # Code source
│   ├── __init__.py
│   ├── data_utils.py             # Utilitaires pour les données
│   ├── model.py                  # Définition du modèle
│   ├── train.py                  # Script d'entraînement
│   ├── predict.py                # Script de prédiction
│   └── visualization.py          # Fonctions de visualisation
├── scripts/                      # Scripts utilitaires
│   ├── download_data.py          # Script pour télécharger les données
│   └── preprocess.py             # Script de prétraitement
└── assets/                       # Ressources pour l'interface
    ├── logo.png                  # Logo du projet
    └── css/                      # Styles CSS personnalisés
        └── style.css             # Fichier de style
```

## 📈 Entraînement du modèle

Pour entraîner le modèle à partir de zéro, exécutez :

```bash
python src/train.py --model_type efficientnet --epochs 50 --batch_size 32
```

Options disponibles :
- `--model_type` : Type de modèle à utiliser ('efficientnet' ou 'custom_cnn')
- `--epochs` : Nombre d'époques d'entraînement
- `--batch_size` : Taille du batch
- `--img_size` : Dimension des images (par défaut : 224)
- `--learning_rate` : Taux d'apprentissage (par défaut : 0.0001)
- `--output_dir` : Répertoire de sortie pour sauvegarder le modèle

## 🧪 Classification d'images

Pour classifier une image avec le modèle entraîné, utilisez :

```bash
python src/predict.py --image_path chemin/vers/image.jpg
```

## 👥 Équipe

- Chef d'équipe et Développeur IA : Isaac Touza
- Data Scientist : Sali Emmanuel
- Developpeur Backend : Mana Tchindebe Etienne
- Développeur Frontend : Mohamed El Bachir
- Expert en Agroalimentaire : Massama Barnabas

## 📊 Jeu de données

Le projet utilise le jeu de données African Plums Dataset, disponible sur Kaggle : [African Plums Dataset](https://www.kaggle.com/datasets/arnaudfadja/african-plums-quality-and-defect-assessment-data)

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

Nous remercions les organisateurs du JCIA 2025 pour cette opportunité, ainsi que Dr. Arnaud Nguembang Fadja pour la mise à disposition du jeu de données African Plums Dataset.

---

© 2025 Équipe PlumsMboa - Tous droits réservés 
