import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import cv2
import tensorflow as tf
import io

def plot_training_history(history, figsize=(12, 5), save_path=None):
    """
    Visualise l'historique d'entraînement (précision et perte)
    
    Args:
        history: Historique d'entraînement Keras
        figsize: Taille de la figure
        save_path: Chemin pour sauvegarder la figure
        
    Returns:
        matplotlib.Figure: Figure matplotlib
    """
    # Créer la figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Tracé de la précision
    ax1.plot(history.history['accuracy'], label='Entraînement')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Précision du modèle')
    ax1.set_ylabel('Précision')
    ax1.set_xlabel('Époque')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Tracé de la perte
    ax2.plot(history.history['loss'], label='Entraînement')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Perte du modèle')
    ax2.set_ylabel('Perte')
    ax2.set_xlabel('Époque')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder la figure si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(10, 8), save_path=None):
    """
    Visualise la matrice de confusion
    
    Args:
        y_true: Étiquettes réelles
        y_pred: Étiquettes prédites
        class_names: Noms des classes
        figsize: Taille de la figure
        save_path: Chemin pour sauvegarder la figure
        
    Returns:
        matplotlib.Figure: Figure matplotlib
    """
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Normaliser la matrice de confusion
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Tracer la matrice de confusion avec seaborn
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    # Configurer les étiquettes
    ax.set_xlabel('Classe prédite')
    ax.set_ylabel('Classe réelle')
    ax.set_title('Matrice de confusion normalisée')
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder la figure si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_class_distribution(df, defect_type_col='defect_type', figsize=(10, 6), save_path=None):
    """
    Visualise la distribution des classes
    
    Args:
        df: DataFrame pandas contenant les données
        defect_type_col: Nom de la colonne contenant les types de défauts
        figsize: Taille de la figure
        save_path: Chemin pour sauvegarder la figure
        
    Returns:
        matplotlib.Figure: Figure matplotlib
    """
    # Calculer les comptes par classe
    class_counts = df[defect_type_col].value_counts().sort_values(ascending=False)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Définir les couleurs pour chaque classe
    colors = {
        'unaffected': '#3CB371',  # Vert
        'unripe': '#98FB98',      # Vert clair
        'spotted': '#FFD700',     # Jaune
        'cracked': '#FFA500',     # Orange
        'bruised': '#FF6347',     # Rouge-orangé
        'rotten': '#8B0000'       # Rouge foncé
    }
    
    # Créer la liste des couleurs dans l'ordre des classes
    bar_colors = [colors.get(cls, '#1f77b4') for cls in class_counts.index]
    
    # Tracer le graphique à barres
    bars = ax.bar(class_counts.index, class_counts.values, color=bar_colors)
    
    # Ajouter les annotations sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 20,
            f'{height}',
            ha='center',
            va='bottom'
        )
    
    # Configurer les étiquettes
    ax.set_title('Distribution des classes de prunes')
    ax.set_xlabel('Type de défaut')
    ax.set_ylabel('Nombre d\'images')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotation des étiquettes pour une meilleure lisibilité
    plt.xticks(rotation=45)
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder la figure si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_sample_images(data_dir, num_samples=3, class_names=None, figsize=(15, 10), save_path=None):
    """
    Visualise des exemples d'images de chaque classe
    
    Args:
        data_dir: Répertoire contenant les images organisées par sous-dossiers
        num_samples: Nombre d'exemples à afficher par classe
        class_names: Noms des classes (sous-dossiers) à visualiser
        figsize: Taille de la figure
        save_path: Chemin pour sauvegarder la figure
        
    Returns:
        matplotlib.Figure: Figure matplotlib
    """
    # Si les noms des classes ne sont pas fournis, utiliser tous les sous-dossiers
    if class_names is None:
        class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Nombre de classes
    num_classes = len(class_names)
    
    # Créer la figure
    fig, axes = plt.subplots(num_classes, num_samples, figsize=figsize)
    
    # Parcourir chaque classe
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        
        # Vérifier si le répertoire existe
        if not os.path.isdir(class_dir):
            print(f"Répertoire non trouvé: {class_dir}")
            continue
        
        # Obtenir la liste des fichiers image
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sélectionner des exemples aléatoires
        selected_samples = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
        
        # Afficher chaque exemple
        for j, sample in enumerate(selected_samples):
            if j < num_samples:
                img_path = os.path.join(class_dir, sample)
                img = Image.open(img_path)
                
                # Si num_classes == 1, axes est un tableau 1D
                if num_classes == 1:
                    axes[j].imshow(img)
                    axes[j].set_title(f"{class_name}")
                    axes[j].axis('off')
                else:
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f"{class_name}")
                    axes[i, j].axis('off')
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder la figure si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_plotly_confusion_matrix(y_true, y_pred, class_names):
    """
    Crée une matrice de confusion interactive avec Plotly
    
    Args:
        y_true: Étiquettes réelles
        y_pred: Étiquettes prédites
        class_names: Noms des classes
        
    Returns:
        plotly.graph_objects.Figure: Figure Plotly
    """
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Normaliser la matrice de confusion
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Créer la figure avec Plotly
    fig = px.imshow(
        cm_norm,
        x=class_names,
        y=class_names,
        color_continuous_scale='Blues',
        labels=dict(x="Classe prédite", y="Classe réelle", color="Proportion")
    )
    
    # Ajouter les annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            fig.add_annotation(
                x=j, y=i,
                text=f"{cm[i, j]} ({cm_norm[i, j]:.2f})",
                showarrow=False,
                font=dict(color="white" if cm_norm[i, j] > 0.5 else "black")
            )
    
    fig.update_layout(
        title="Matrice de confusion",
        xaxis_title="Classe prédite",
        yaxis_title="Classe réelle",
        width=800,
        height=800
    )
    
    return fig

def visualize_augmented_images(original_img, augmentation_func, num_examples=4, figsize=(15, 8)):
    """
    Visualise plusieurs exemples d'augmentation d'une image
    
    Args:
        original_img: Image originale (chemin ou tableau numpy)
        augmentation_func: Fonction d'augmentation qui prend une image et retourne une image augmentée
        num_examples: Nombre d'exemples augmentés à afficher
        figsize: Taille de la figure
        
    Returns:
        matplotlib.Figure: Figure matplotlib
    """
    # Charger l'image si un chemin est fourni
    if isinstance(original_img, str):
        img = Image.open(original_img)
        img_array = np.array(img)
    else:
        img_array = original_img
    
    # Créer la figure
    fig, axes = plt.subplots(1, num_examples + 1, figsize=figsize)
    
    # Afficher l'image originale
    axes[0].imshow(img_array)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Afficher les exemples augmentés
    for i in range(num_examples):
        # Appliquer l'augmentation
        augmented = augmentation_func(img_array.copy())
        
        # Afficher l'image augmentée
        axes[i + 1].imshow(augmented)
        axes[i + 1].set_title(f"Augmentation {i+1}")
        axes[i + 1].axis('off')
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    return fig

def plot_attention_heatmap(model, image_path, layer_name=None, figsize=(12, 5)):
    """
    Visualise une carte de chaleur d'attention sur une image
    
    Args:
        model: Modèle Keras
        image_path: Chemin vers l'image
        layer_name: Nom de la couche de convolution à visualiser
        figsize: Taille de la figure
        
    Returns:
        matplotlib.Figure: Figure matplotlib
    """
    # Charger et prétraiter l'image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    
    # Obtenir les prédictions
    preds = model.predict(x)
    pred_class = np.argmax(preds[0])
    
    # Si le nom de la couche n'est pas spécifié, prendre la dernière couche de convolution
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
                layer_name = layer.name
                break
    
    # Créer un modèle pour extraire les activations
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Enregistrer les gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, pred_class]
    
    # Obtenir les gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Moyenne des gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Pondérer les canaux de sortie de la couche de convolution avec les gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normaliser la carte de chaleur
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    # Redimensionner la carte de chaleur à la taille de l'image
    img_orig = cv2.imread(image_path)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    heatmap = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superposer la carte de chaleur sur l'image
    superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)
    
    # Créer la figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Afficher l'image originale
    ax1.imshow(img_orig)
    ax1.set_title("Image originale")
    ax1.axis('off')
    
    # Afficher l'image avec la carte de chaleur
    ax2.imshow(superimposed_img)
    ax2.set_title("Carte d'attention")
    ax2.axis('off')
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    return fig
