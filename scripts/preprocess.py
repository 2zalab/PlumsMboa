#!/usr/bin/env python3
"""
Script de prétraitement des données pour le projet PlumsMboa
"""

import os
import argparse
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import shutil
import random
import sys

# Ajouter le répertoire parent au chemin de recherche pour importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import load_data

def resize_images(input_dir, output_dir, target_size=(224, 224), overwrite=False):
    """
    Redimensionne toutes les images dans le répertoire d'entrée et les enregistre dans le répertoire de sortie
    
    Args:
        input_dir: Répertoire contenant les images originales
        output_dir: Répertoire pour enregistrer les images redimensionnées
        target_size: Taille cible pour les images
        overwrite: Si True, écrase les fichiers existants
    """
    # Vérifier si le répertoire de sortie existe, sinon le créer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Récupérer la liste des sous-dossiers (catégories)
    categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    total_processed = 0
    
    # Parcourir chaque catégorie
    for category in categories:
        category_input_dir = os.path.join(input_dir, category)
        category_output_dir = os.path.join(output_dir, category)
        
        # Créer le sous-dossier de sortie s'il n'existe pas
        if not os.path.exists(category_output_dir):
            os.makedirs(category_output_dir)
        
        # Récupérer la liste des images dans cette catégorie
        images = [f for f in os.listdir(category_input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Afficher la progression
        print(f"Traitement de la catégorie '{category}' ({len(images)} images)...")
        
        # Parcourir et traiter chaque image
        for img_file in tqdm(images, desc=category):
            input_path = os.path.join(category_input_dir, img_file)
            output_path = os.path.join(category_output_dir, img_file)
            
            # Vérifier si le fichier existe déjà et si nous ne voulons pas l'écraser
            if os.path.exists(output_path) and not overwrite:
                continue
            
            try:
                # Ouvrir l'image avec PIL
                img = Image.open(input_path)
                
                # Redimensionner l'image
                img_resized = img.resize(target_size, Image.LANCZOS)
                
                # Sauvegarder l'image redimensionnée
                img_resized.save(output_path)
                
                total_processed += 1
            except Exception as e:
                print(f"Erreur lors du traitement de {input_path}: {e}")
    
    print(f"Prétraitement terminé. {total_processed} images ont été traitées.")

def create_train_val_test_split(data_dir, csv_path, output_dir, val_size=0.15, test_size=0.15, random_state=42):
    """
    Divise le jeu de données en ensembles d'entraînement, de validation et de test
    
    Args:
        data_dir: Répertoire contenant les images organisées par sous-dossiers
        csv_path: Chemin vers le fichier CSV contenant les métadonnées
        output_dir: Répertoire pour enregistrer les ensembles divisés
        val_size: Proportion des données à utiliser pour la validation
        test_size: Proportion des données à utiliser pour le test
        random_state: Graine aléatoire pour la reproductibilité
    """
    # Charger les données
    df = load_data(csv_path, data_dir)
    
    # Vérifier si le répertoire de sortie existe, sinon le créer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Créer les sous-répertoires pour les ensembles d'entraînement, de validation et de test
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Récupérer la liste des catégories
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Créer les sous-répertoires pour chaque catégorie dans chaque ensemble
    for category in categories:
        for directory in [train_dir, val_dir, test_dir]:
            category_dir = os.path.join(directory, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
    
    # Diviser les données pour chaque catégorie
    for category in categories:
        print(f"Traitement de la catégorie '{category}'...")
        
        # Filtrer les données pour cette catégorie
        category_df = df[df['defect_type'] == category]
        
        # Obtenir la liste des noms de fichiers pour cette catégorie
        files = []
        category_dir = os.path.join(data_dir, category)
        for f in os.listdir(category_dir):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                files.append(f)
        
        # Mélanger les fichiers
        random.seed(random_state)
        random.shuffle(files)
        
        # Calculer les indices de séparation
        val_split = int(len(files) * val_size)
        test_split = int(len(files) * test_size)
        
        # Diviser les fichiers
        test_files = files[:test_split]
        val_files = files[test_split:test_split + val_split]
        train_files = files[test_split + val_split:]
        
        # Copier les fichiers dans les répertoires appropriés
        for file_list, target_dir in [(train_files, train_dir), (val_files, val_dir), (test_files, test_dir)]:
            for file in file_list:
                src = os.path.join(data_dir, category, file)
                dst = os.path.join(target_dir, category, file)
                shutil.copy2(src, dst)
        
        print(f"  Entraînement: {len(train_files)} images")
        print(f"  Validation: {len(val_files)} images")
        print(f"  Test: {len(test_files)} images")
    
    print("Division des données terminée.")

def augment_data(input_dir, output_dir, augmentation_factor=2, overwrite=False):
    """
    Augmente les données en appliquant des transformations aléatoires aux images
    
    Args:
        input_dir: Répertoire contenant les images originales
        output_dir: Répertoire pour enregistrer les images augmentées
        augmentation_factor: Nombre d'images augmentées à générer pour chaque image originale
        overwrite: Si True, écrase les fichiers existants
    """
    # Vérifier si le répertoire de sortie existe, sinon le créer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Récupérer la liste des sous-dossiers (catégories)
    categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Parcourir chaque catégorie
    for category in categories:
        category_input_dir = os.path.join(input_dir, category)
        category_output_dir = os.path.join(output_dir, category)
        
        # Créer le sous-dossier de sortie s'il n'existe pas
        if not os.path.exists(category_output_dir):
            os.makedirs(category_output_dir)
        
        # Récupérer la liste des images dans cette catégorie
        images = [f for f in os.listdir(category_input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Afficher la progression
        print(f"Augmentation de la catégorie '{category}' ({len(images)} images)...")
        
        # Parcourir et traiter chaque image
        for img_file in tqdm(images, desc=category):
            input_path = os.path.join(category_input_dir, img_file)
            
            # Créer le nom de base pour les fichiers augmentés
            base_name = os.path.splitext(img_file)[0]
            extension = os.path.splitext(img_file)[1]
            
            # Copier l'original dans le répertoire de sortie
            output_path = os.path.join(category_output_dir, img_file)
            if not os.path.exists(output_path) or overwrite:
                shutil.copy2(input_path, output_path)
            
            # Ouvrir l'image avec PIL
            try:
                img = Image.open(input_path)
                
                # Générer des images augmentées
                for i in range(augmentation_factor):
                    # Créer un nom pour le fichier augmenté
                    aug_file = f"{base_name}_aug_{i+1}{extension}"
                    aug_path = os.path.join(category_output_dir, aug_file)
                    
                    # Vérifier si le fichier existe déjà et si nous ne voulons pas l'écraser
                    if os.path.exists(aug_path) and not overwrite:
                        continue
                    
                    # Appliquer des transformations aléatoires
                    augmented_img = img.copy()
                    
                    # Rotation aléatoire (-20° à +20°)
                    if random.random() > 0.5:
                        angle = random.uniform(-20, 20)
                        augmented_img = augmented_img.rotate(angle, Image.BICUBIC, expand=False)
                    
                    # Flip horizontal
                    if random.random() > 0.5:
                        augmented_img = augmented_img.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    # Ajustement de la luminosité
                    if random.random() > 0.5:
                        factor = random.uniform(0.8, 1.2)
                        enhancer = ImageEnhance.Brightness(augmented_img)
                        augmented_img = enhancer.enhance(factor)
                    
                    # Ajustement du contraste
                    if random.random() > 0.5:
                        factor = random.uniform(0.8, 1.2)
                        enhancer = ImageEnhance.Contrast(augmented_img)
                        augmented_img = enhancer.enhance(factor)
                    
                    # Ajustement de la couleur
                    if random.random() > 0.5:
                        factor = random.uniform(0.8, 1.2)
                        enhancer = ImageEnhance.Color(augmented_img)
                        augmented_img = enhancer.enhance(factor)
                    
                    # Sauvegarder l'image augmentée
                    augmented_img.save(aug_path)
            
            except Exception as e:
                print(f"Erreur lors de l'augmentation de {input_path}: {e}")
    
    print("Augmentation des données terminée.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de prétraitement des données pour le projet PlumsMboa")
    
    parser.add_argument('--data_dir', type=str, default="data/african_plums_dataset",
                        help='Répertoire contenant les images organisées par sous-dossiers')
    parser.add_argument('--csv_path', type=str, default="data/plums_data.csv",
                        help='Chemin vers le fichier CSV contenant les métadonnées')
    parser.add_argument('--output_dir', type=str, default="data/processed",
                        help="Répertoire de sortie pour les images traitées")
    parser.add_argument('--resize', action='store_true',
                        help="Redimensionner les images")
    parser.add_argument('--target_size', type=int, default=224,
                        help="Taille cible pour les images (carré)")
    parser.add_argument('--split', action='store_true',
                        help="Diviser les données en ensembles d'entraînement, de validation et de test")
    parser.add_argument('--augment', action='store_true',
                        help="Augmenter les données")
    parser.add_argument('--check', action='store_true',
                        help="Vérifier l'intégrité du jeu de données")
    parser.add_argument('--all', action='store_true',
                        help="Exécuter toutes les étapes de prétraitement")
    
    args = parser.parse_args()
    
    # Si aucune option n'est spécifiée, afficher l'aide
    if not (args.resize or args.split or args.augment or args.check or args.all):
        parser.print_help()
        sys.exit(1)
    
    # Exécuter les étapes demandées
    if args.check or args.all:
        check_dataset_integrity(args.data_dir, args.csv_path)
    
    if args.resize or args.all:
        resize_images(args.data_dir, args.output_dir, target_size=(args.target_size, args.target_size))
    
    if args.split or args.all:
        # Utiliser le répertoire des images redimensionnées si le redimensionnement a été effectué
        src_dir = args.output_dir if (args.resize or args.all) else args.data_dir
        split_dir = os.path.join(args.output_dir, "split")
        create_train_val_test_split(src_dir, args.csv_path, split_dir)
    
    if args.augment or args.all:
        # Utiliser le répertoire d'entraînement si la division a été effectuée
        src_dir = os.path.join(args.output_dir, "split", "train") if (args.split or args.all) else args.data_dir
        aug_dir = os.path.join(args.output_dir, "augmented")
        augment_data(src_dir, aug_dir)

def check_dataset_integrity(data_dir, csv_path):
    """
    Vérifie l'intégrité du jeu de données
    
    Args:
        data_dir: Répertoire contenant les images organisées par sous-dossiers
        csv_path: Chemin vers le fichier CSV contenant les métadonnées
    """
    print("Vérification de l'intégrité du jeu de données...")
    
    # Charger les données
    df = pd.read_csv(csv_path)
    
    # Vérifier si le répertoire principal existe
    if not os.path.exists(data_dir):
        print(f"Erreur: Le répertoire {data_dir} n'existe pas.")
        return
    
    # Récupérer la liste des sous-dossiers (catégories)
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Vérifier si toutes les catégories attendues existent
    expected_categories = df['defect_type'].unique()
    missing_categories = set(expected_categories) - set(categories)
    
    if missing_categories:
        print(f"Avertissement: Catégories manquantes dans le répertoire des données: {missing_categories}")
    
    # Vérifier le nombre d'images dans chaque catégorie
    total_images = 0
    corrupt_images = 0
    
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        images = [f for f in os.listdir(category_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        total_images += len(images)
        
        # Vérifier si les images peuvent être ouvertes (non corrompues)
        for img_file in images:
            img_path = os.path.join(category_dir, img_file)
            try:
                img = Image.open(img_path)
                img.verify()  # Vérifier si l'image est valide
            except Exception as e:
                corrupt_images += 1
                print(f"Image corrompue: {img_path}")
        
        # Comparer avec le nombre attendu d'après le CSV
        expected_count = len(df[df['defect_type'] == category])
        actual_count = len(images)
        
        print(f"Catégorie '{category}': {actual_count} images trouvées, {expected_count} attendues")
        
        if actual_count != expected_count:
            print(f"  Avertissement: Le nombre d'images ne correspond pas au CSV pour la catégorie '{category}'") 
