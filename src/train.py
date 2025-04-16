import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import json
import time
from datetime import datetime

from data_utils import load_data, create_data_generators
from model import create_efficientnet_model, create_custom_cnn_model, create_callbacks
from visualization import plot_training_history

def train_model(
    data_dir='../data/african_plums_dataset',
    csv_path='../data/plums_data.csv',
    model_type='efficientnet',
    batch_size=32,
    img_size=(224, 224),
    epochs=50,
    learning_rate=0.0001,
    output_dir='models',
    feature_extraction_epochs=10,
    fine_tuning_epochs=40,
):
    """
    Entraîne un modèle de classification des prunes
    
    Args:
        data_dir: Répertoire contenant les images organisées par sous-dossiers
        csv_path: Chemin vers le fichier CSV contenant les métadonnées
        model_type: Type de modèle à utiliser ('efficientnet' ou 'custom_cnn')
        batch_size: Taille du batch
        img_size: Dimensions cibles des images
        epochs: Nombre total d'époques d'entraînement
        learning_rate: Taux d'apprentissage
        output_dir: Répertoire de sortie pour sauvegarder le modèle et les résultats
        feature_extraction_epochs: Nombre d'époques pour la phase 1 (extraction de caractéristiques)
        fine_tuning_epochs: Nombre d'époques pour la phase 2 (fine-tuning)
    """
    try:
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Créer un sous-dossier avec horodatage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Logger les paramètres d'entraînement
        params = {
            "data_dir": data_dir,
            "csv_path": csv_path,
            "model_type": model_type,
            "batch_size": batch_size,
            "img_size": img_size,
            "epochs": epochs,
            "feature_extraction_epochs": feature_extraction_epochs,
            "fine_tuning_epochs": fine_tuning_epochs,
            "learning_rate": learning_rate,
            "timestamp": timestamp
        }
        
        with open(os.path.join(run_dir, 'training_params.json'), 'w') as f:
            json.dump(params, f, indent=4)
        
        # Charger ou créer les données
        print("Chargement des données...")
        try:
            # Essayer de charger les données si le CSV existe
            if os.path.exists(csv_path):
                df = load_data(csv_path, data_dir)
                print(f"Données chargées depuis {csv_path}")
            else:
                print(f"Fichier CSV {csv_path} non trouvé, utilisation directe des dossiers")
                # Utiliser directement les dossiers sans CSV
                df = None
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            print("Utilisation directe des dossiers sans CSV")
            df = None
        
        # Obtenir les catégories depuis les dossiers
        categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        num_classes = len(categories)
        class_indices = {category: i for i, category in enumerate(categories)}
        
        print(f"Nombre de classes: {num_classes}")
        print(f"Catégories détectées: {categories}")
        print(f"Indices des classes: {class_indices}")
        
        # Sauvegarder les indices des classes pour la prédiction
        with open(os.path.join(run_dir, 'class_indices.json'), 'w') as f:
            json.dump(class_indices, f, indent=4)
        
        # Créer les générateurs de données
        print("Création des générateurs de données...")
        try:
            train_generator, validation_generator, _ = create_data_generators(
                data_dir, batch_size, img_size
            )
        except Exception as e:
            print(f"Erreur lors de la création des générateurs de données: {e}")
            # Créer des générateurs manuellement
            print("Création manuelle des générateurs de données...")
            
            # Générateur d'entraînement avec augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2
            )
            
            # Générateur pour les données d'entraînement
            train_generator = train_datagen.flow_from_directory(
                data_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )
            
            # Générateur pour les données de validation
            validation_generator = train_datagen.flow_from_directory(
                data_dir,
                target_size=img_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )
            
            # Mettre à jour les indices de classe
            class_indices = train_generator.class_indices
            with open(os.path.join(run_dir, 'class_indices.json'), 'w') as f:
                json.dump(class_indices, f, indent=4)
        
        # Créer le modèle
        print(f"Création du modèle {model_type}...")
        if model_type == 'efficientnet':
            model, base_model = create_efficientnet_model(
                input_shape=img_size + (3,),
                num_classes=num_classes,
                learning_rate=learning_rate
            )
        else:
            model = create_custom_cnn_model(
                input_shape=img_size + (3,),
                num_classes=num_classes
            )
            base_model = None
        
        # Résumé du modèle
        model.summary()
        
        # Créer un fichier pour sauvegarder le résumé du modèle
        with open(os.path.join(run_dir, 'model_summary.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Créer les callbacks avec save_weights_only=True pour éviter les problèmes de sérialisation
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(run_dir, 'best_weights.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=True  # Sauvegarder uniquement les poids
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(run_dir, 'training_log.csv'),
                separator=',',
                append=False
            )
        ]
        
        # Phase 1: Feature Extraction (modèle de base gelé)
        print("\nPhase 1: Feature Extraction")
        
        # Enregistrer le temps de début
        start_time = time.time()
        
        # Entraîner le modèle (feature extraction)
        history1 = model.fit(
            train_generator,
            epochs=feature_extraction_epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning (dégeler certaines couches)
        if model_type == 'efficientnet' and base_model is not None:
            print("\nPhase 2: Fine-tuning")
            
            # Dégeler certaines couches pour le fine-tuning
            base_model.trainable = True
            
            # Geler les premières couches et dégeler les dernières
            for layer in base_model.layers[:-30]:  # Geler toutes les couches sauf les 30 dernières
                layer.trainable = False
            
            # Afficher le nombre de couches entraînables
            trainable_count = sum(1 for layer in model.layers if layer.trainable)
            total_count = len(model.layers)
            print(f"Nombre de couches entraînables: {trainable_count}/{total_count}")
                
            # Recompiler le modèle avec un taux d'apprentissage plus faible
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate / 10),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Fine-tuning du modèle
            history2 = model.fit(
                train_generator,
                epochs=fine_tuning_epochs,
                validation_data=validation_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combiner les historiques
            combined_history = {}
            for key in history1.history.keys():
                combined_history[key] = history1.history[key] + history2.history[key]
            
            # Sauvegarder l'historique combiné
            pd.DataFrame(combined_history).to_csv(os.path.join(run_dir, 'training_history.csv'), index=False)
            
            # Visualiser l'historique combiné
            try:
                plot_training_history(history1, history2, save_path=os.path.join(run_dir, 'training_history.png'))
            except Exception as e:
                print(f"Erreur lors de la visualisation de l'historique: {e}")
                
                # Sauvegarde manuelle des graphiques
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(combined_history['accuracy'], label='Entraînement')
                plt.plot(combined_history['val_accuracy'], label='Validation')
                plt.title('Précision du modèle')
                plt.xlabel('Époque')
                plt.ylabel('Précision')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(combined_history['loss'], label='Entraînement')
                plt.plot(combined_history['val_loss'], label='Validation')
                plt.title('Perte du modèle')
                plt.xlabel('Époque')
                plt.ylabel('Perte')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, 'training_history_manual.png'))
        else:
            # Sauvegarder l'historique d'entraînement
            pd.DataFrame(history1.history).to_csv(os.path.join(run_dir, 'training_history.csv'), index=False)
            
            # Visualiser l'historique d'entraînement
            try:
                plot_training_history(history1, save_path=os.path.join(run_dir, 'training_history.png'))
            except Exception as e:
                print(f"Erreur lors de la visualisation de l'historique: {e}")
        
        # Calculer le temps d'entraînement
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Temps d'entraînement total: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Sauvegarder les poids du modèle final
        try:
            # Essayer de sauvegarder le modèle complet au format SavedModel
            model.save(os.path.join(run_dir, 'final_model'), save_format='tf')
            print(f"Modèle complet sauvegardé dans {os.path.join(run_dir, 'final_model')}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du modèle complet: {e}")
            
            # Alternative: sauvegarder uniquement les poids
            model.save_weights(os.path.join(run_dir, 'final_weights.h5'))
            print(f"Poids du modèle sauvegardés dans {os.path.join(run_dir, 'final_weights.h5')}")
            
            # Sauvegarder la configuration du modèle si possible
            try:
                model_json = model.to_json()
                with open(os.path.join(run_dir, 'model_config.json'), 'w') as f:
                    f.write(model_json)
                print(f"Configuration du modèle sauvegardée dans {os.path.join(run_dir, 'model_config.json')}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde de la configuration du modèle: {e}")
        
        # Sauvegarder un lien symbolique vers le dernier entraînement réussi
        last_run_link = os.path.join(output_dir, 'last_run')
        try:
            # Supprimer le lien existant s'il existe
            if os.path.exists(last_run_link):
                if os.path.islink(last_run_link):
                    os.unlink(last_run_link)
                else:
                    os.rename(last_run_link, f"{last_run_link}_{int(time.time())}")
            
            # Créer un nouveau lien ou dossier avec le nom du run actuel
            try:
                os.symlink(run_dir, last_run_link, target_is_directory=True)
            except:
                # Si les liens symboliques ne sont pas supportés (Windows sans droits admin)
                # Créer un fichier texte contenant le chemin
                with open(f"{last_run_link}.txt", 'w') as f:
                    f.write(run_dir)
        except Exception as e:
            print(f"Erreur lors de la création du lien vers le dernier run: {e}")
        
        print(f"Entraînement terminé! Tous les résultats sont sauvegardés dans {run_dir}")
        return run_dir
        
    except Exception as e:
        print(f"Erreur lors de l'entraînement du modèle: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'entraînement pour le modèle de classification des prunes")
    
    parser.add_argument('--data_dir', type=str, default='data/african_plums_dataset',
                        help='Répertoire contenant les images organisées par sous-dossiers')
    parser.add_argument('--csv_path', type=str, default='data/plums_data.csv',
                        help='Chemin vers le fichier CSV contenant les métadonnées')
    parser.add_argument('--model_type', type=str, default='efficientnet', choices=['efficientnet', 'custom_cnn'],
                        help='Type de modèle à utiliser')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Taille du batch')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Dimension des images (carrées)')
    parser.add_argument('--feature_extraction_epochs', type=int, default=10,
                        help="Nombre d'époques pour la phase d'extraction de caractéristiques")
    parser.add_argument('--fine_tuning_epochs', type=int, default=40,
                        help="Nombre d'époques pour la phase de fine-tuning")
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help="Taux d'apprentissage")
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Répertoire de sortie pour sauvegarder le modèle et les résultats')
    
    args = parser.parse_args()
    
    # Exécuter l'entraînement avec les arguments fournis
    train_model(
        data_dir=args.data_dir,
        csv_path=args.csv_path,
        model_type=args.model_type,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        epochs=args.feature_extraction_epochs + args.fine_tuning_epochs,
        feature_extraction_epochs=args.feature_extraction_epochs,
        fine_tuning_epochs=args.fine_tuning_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )