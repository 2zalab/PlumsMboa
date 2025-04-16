import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import cv2

def load_class_indices(model_dir='models'):
    """
    Charge les indices des classes à partir du fichier JSON
    
    Args:
        model_dir: Répertoire contenant le fichier d'indices des classes
        
    Returns:
        dict: Dictionnaire d'indices des classes (inversé pour mapper les indices aux noms)
    """
    try:
        with open(os.path.join(model_dir, 'class_indices.json'), 'r') as f:
            class_indices = json.load(f)
        
        # Inverser le dictionnaire pour mapper les indices aux noms
        class_indices_inv = {v: k for k, v in class_indices.items()}
        return class_indices_inv
    except FileNotFoundError:
        print(f"Fichier d'indices de classe non trouvé: {os.path.join(model_dir, 'class_indices.json')}")
        return None

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Prétraite une image pour la prédiction
    
    Args:
        image_path: Chemin vers l'image ou objet PIL.Image
        target_size: Dimensions cibles de l'image (hauteur, largeur)
        
    Returns:
        numpy.ndarray: Image prétraitée prête pour la prédiction
    """
    # Charger l'image depuis le chemin si un chemin est fourni
    if isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        img = image_path  # Utiliser l'objet image directement
    
    # Redimensionner l'image
    img = img.resize(target_size)
    
    # Convertir en tableau numpy
    img_array = img_to_array(img)
    
    # Ajouter la dimension du batch (1 image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normaliser les valeurs de pixels
    img_array = img_array / 255.0
    
    return img_array

def predict_image(model, image_path, class_indices_inv=None, target_size=(224, 224)):
    """
    Prédit la classe d'une image
    
    Args:
        model: Modèle Keras chargé
        image_path: Chemin vers l'image ou objet PIL.Image
        class_indices_inv: Dictionnaire inversé d'indices des classes
        target_size: Dimensions cibles de l'image (hauteur, largeur)
        
    Returns:
        tuple: (classe prédite, probabilité, toutes les probabilités)
    """
    # Prétraiter l'image
    img_array = preprocess_image(image_path, target_size)
    
    # Prédire la classe
    predictions = model.predict(img_array)
    
    # Obtenir l'indice de la classe prédite
    pred_index = np.argmax(predictions[0])
    
    # Obtenir la probabilité de la classe prédite
    pred_prob = predictions[0][pred_index]
    
    # Obtenir le nom de la classe prédite si les indices sont fournis
    if class_indices_inv is not None:
        pred_class = class_indices_inv[pred_index]
    else:
        pred_class = f"classe_{pred_index}"
    
    return pred_class, pred_prob, predictions[0]

def predict_batch(model, image_paths, class_indices_inv=None, target_size=(224, 224), batch_size=32):
    """
    Prédit les classes pour un lot d'images
    
    Args:
        model: Modèle Keras chargé
        image_paths: Liste de chemins vers les images
        class_indices_inv: Dictionnaire inversé d'indices des classes
        target_size: Dimensions cibles des images (hauteur, largeur)
        batch_size: Taille du lot pour la prédiction
        
    Returns:
        list: Liste de tuples (chemin de l'image, classe prédite, probabilité)
    """
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            img_array = preprocess_image(img_path, target_size)
            batch_images.append(img_array[0])
        
        batch_images = np.array(batch_images)
        batch_predictions = model.predict(batch_images)
        
        for j, pred in enumerate(batch_predictions):
            pred_index = np.argmax(pred)
            pred_prob = pred[pred_index]
            
            if class_indices_inv is not None:
                pred_class = class_indices_inv[pred_index]
            else:
                pred_class = f"classe_{pred_index}"
            
            results.append((batch_paths[j], pred_class, pred_prob, pred))
    
    return results

def visualize_prediction(image_path, pred_class, pred_prob, defect_descriptions=None):
    """
    Visualise une image avec sa prédiction
    
    Args:
        image_path: Chemin vers l'image
        pred_class: Classe prédite
        pred_prob: Probabilité de la prédiction
        defect_descriptions: Dictionnaire de descriptions des défauts
        
    Returns:
        numpy.ndarray: Image avec l'annotation
    """
    # Charger l'image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Définir les couleurs pour chaque classe
    colors = {
        'unaffected': (60, 179, 113),    # Vert
        'unripe': (152, 251, 152),       # Vert clair
        'spotted': (255, 215, 0),        # Jaune
        'cracked': (255, 165, 0),        # Orange
        'bruised': (255, 99, 71),        # Rouge-orangé
        'rotten': (139, 0, 0)            # Rouge foncé
    }
    
    # Couleur par défaut si la classe n'est pas dans le dictionnaire
    color = colors.get(pred_class, (255, 255, 255))
    
    # Obtenir la description si disponible
    description = ""
    if defect_descriptions is not None and pred_class in defect_descriptions:
        description = defect_descriptions[pred_class]
    
    # Dessiner un rectangle en haut de l'image pour le texte
    h, w, _ = img.shape
    cv2.rectangle(img, (0, 0), (w, 60), color, -1)
    
    # Ajouter le texte
    cv2.putText(img, f"Classe: {pred_class}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Prob: {pred_prob:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Ajouter la description si disponible
    if description:
        cv2.rectangle(img, (0, h-40), (w, h), (0, 0, 0), -1)
        cv2.putText(img, description, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return img

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="Script de prédiction pour le modèle de classification des prunes")
    
    parser.add_argument('--model_path', type=str, default='models/plum_classifier.h5',
                        help='Chemin vers le modèle entraîné')
    parser.add_argument('--image_path', type=str, required=True,
                        help="Chemin vers l'image à classifier")
    parser.add_argument('--model_dir', type=str, default='models',
                        help="Répertoire contenant le fichier d'indices des classes")
    
    args = parser.parse_args()
    
    # Charger le modèle
    model = load_model(args.model_path)
    
    # Charger les indices des classes
    class_indices_inv = load_class_indices(args.model_dir)
    
    # Descriptions des défauts
    defect_descriptions = {
        'unaffected': "Prune de bonne qualité, prête pour la commercialisation.",
        'unripe': "Prune non mûre, besoin de plus de temps avant récolte.",
        'spotted': "Prune tachetée, peut être utilisée pour la transformation.",
        'cracked': "Prune fissurée, qualité réduite, à traiter rapidement.",
        'bruised': "Prune meurtrie, peut être utilisée pour certaines transformations.",
        'rotten': "Prune pourrie, à écarter de la chaîne de production."
    }
    
    # Prédire la classe de l'image
    pred_class, pred_prob, _ = predict_image(model, args.image_path, class_indices_inv)
    
    # Afficher les résultats
    print(f"Classe prédite: {pred_class}")
    print(f"Probabilité: {pred_prob:.4f}")
    
    # Visualiser la prédiction
    img_result = visualize_prediction(args.image_path, pred_class, pred_prob, defect_descriptions)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_result)
    plt.axis('off')
    plt.title(f"Prédiction: {pred_class} ({pred_prob:.4f})")
    plt.show() 
