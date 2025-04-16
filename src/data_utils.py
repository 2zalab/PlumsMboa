import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(csv_path="data/plums_data.csv", img_dir="data/african_plums_dataset"):
    """
    Charge les données à partir du fichier CSV et du répertoire d'images
    
    Args:
        csv_path: Chemin vers le fichier CSV contenant les métadonnées
        img_dir: Répertoire contenant les images organisées par sous-dossiers
        
    Returns:
        df: DataFrame pandas contenant les métadonnées
    """
    # Vérifier si le fichier CSV existe
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier CSV {csv_path} n'existe pas.")
    
    # Charger le fichier CSV
    df = pd.read_csv(csv_path)
    
    # Vérifier si le répertoire d'images existe
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Le répertoire d'images {img_dir} n'existe pas.")
    
    return df

def get_class_distribution(df, defect_type_col="defect_type"):
    """
    Calcule la distribution des classes dans le jeu de données
    
    Args:
        df: DataFrame pandas contenant les métadonnées
        defect_type_col: Nom de la colonne contenant les types de défauts
        
    Returns:
        dict: Dictionnaire contenant la distribution des classes
    """
    class_counts = df[defect_type_col].value_counts()
    total = len(df)
    
    distribution = {
        "counts": class_counts.to_dict(),
        "percentages": (class_counts / total * 100).round(2).to_dict(),
        "total": total
    }
    
    return distribution

def create_data_generators(img_dir, batch_size=32, img_size=(224, 224), validation_split=0.2):
    """
    Crée des générateurs de données pour l'entraînement et la validation
    
    Args:
        img_dir: Répertoire contenant les images organisées par sous-dossiers
        batch_size: Taille du batch
        img_size: Dimensions cibles des images
        validation_split: Proportion des données à utiliser pour la validation
        
    Returns:
        train_generator: Générateur pour les données d'entraînement
        validation_generator: Générateur pour les données de validation
        class_indices: Dictionnaire mappant les noms de classes aux indices
    """
    # Augmentation des données pour l'entraînement
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Générateur pour les données d'entraînement
    train_generator = train_datagen.flow_from_directory(
        img_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Générateur pour les données de validation
    validation_generator = train_datagen.flow_from_directory(
        img_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator, train_generator.class_indices

def prepare_test_data(test_dir, img_size=(224, 224), batch_size=32):
    """
    Prépare les données de test
    
    Args:
        test_dir: Répertoire contenant les images de test
        img_size: Dimensions cibles des images
        batch_size: Taille du batch
        
    Returns:
        test_generator: Générateur pour les données de test
        class_indices: Dictionnaire mappant les noms de classes aux indices
    """
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator, test_generator.class_indices

def split_data(df, img_dir, test_size=0.2, random_state=42):
    """
    Divise les données en ensembles d'entraînement et de test
    
    Args:
        df: DataFrame pandas contenant les métadonnées
        img_dir: Répertoire contenant les images
        test_size: Proportion des données à utiliser pour le test
        random_state: Graine aléatoire pour la reproductibilité
        
    Returns:
        train_df: DataFrame pour les données d'entraînement
        test_df: DataFrame pour les données de test
    """
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['defect_type']  # Stratification pour maintenir la distribution des classes
    )
    
    return train_df, test_df 
