import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def create_custom_cnn_model(input_shape=(224, 224, 3), num_classes=6, learning_rate=0.0001):
    """
    Crée un modèle CNN personnalisé
    
    Args:
        input_shape: Dimensions de l'image d'entrée
        num_classes: Nombre de classes
        learning_rate: Taux d'apprentissage pour l'optimiseur
        
    Returns:
        model: Modèle Keras compilé
        None: Pas de modèle de base (pour compatibilité avec l'interface de create_efficientnet_model)
    """
    model = Sequential([
        # Première couche de convolution
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Deuxième couche de convolution
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Troisième couche de convolution
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Quatrième couche de convolution
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Aplatir les caractéristiques
        Flatten(),
        
        # Couches denses
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Couche de sortie
        Dense(num_classes, activation='softmax')
    ])
    
    # Compiler le modèle
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, None

def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=6, learning_rate=0.0001, fine_tune=False):
    """
    Crée un modèle basé sur EfficientNetB0 avec transfert d'apprentissage
    
    Args:
        input_shape: Dimensions de l'image d'entrée
        num_classes: Nombre de classes
        learning_rate: Taux d'apprentissage pour l'optimiseur
        fine_tune: Si True, les couches du modèle de base sont débloquées pour l'apprentissage
        
    Returns:
        tuple: (modèle Keras compilé, modèle de base)
    """
    try:
        # Charger le modèle EfficientNetB0 pré-entraîné
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Geler les couches du modèle de base (pour la phase d'extraction de caractéristiques)
        base_model.trainable = False
        
        # Créer le modèle complet
        inputs = Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        # Compiler le modèle
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model, base_model
        
    except Exception as e:
        print(f"Erreur lors de la création du modèle EfficientNet: {e}")
        print("Création d'un modèle CNN personnalisé comme solution de secours...")
        return create_custom_cnn_model(input_shape, num_classes, learning_rate)

def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=6, learning_rate=0.0001):
    """
    Crée un modèle basé sur MobileNetV2 avec transfert d'apprentissage
    
    Args:
        input_shape: Dimensions de l'image d'entrée
        num_classes: Nombre de classes
        learning_rate: Taux d'apprentissage pour l'optimiseur
        
    Returns:
        tuple: (modèle Keras compilé, modèle de base)
    """
    try:
        # Charger le modèle MobileNetV2 pré-entraîné
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Geler les couches du modèle de base
        base_model.trainable = False
        
        # Créer le modèle complet
        inputs = Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        # Compiler le modèle
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model, base_model
        
    except Exception as e:
        print(f"Erreur lors de la création du modèle MobileNet: {e}")
        print("Création d'un modèle CNN personnalisé comme solution de secours...")
        return create_custom_cnn_model(input_shape, num_classes, learning_rate)

def create_resnet_model(input_shape=(224, 224, 3), num_classes=6, learning_rate=0.0001):
    """
    Crée un modèle basé sur ResNet50 avec transfert d'apprentissage
    
    Args:
        input_shape: Dimensions de l'image d'entrée
        num_classes: Nombre de classes
        learning_rate: Taux d'apprentissage pour l'optimiseur
        
    Returns:
        tuple: (modèle Keras compilé, modèle de base)
    """
    try:
        # Charger le modèle ResNet50 pré-entraîné
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Geler les couches du modèle de base
        base_model.trainable = False
        
        # Créer le modèle complet
        inputs = Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        # Compiler le modèle
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model, base_model
        
    except Exception as e:
        print(f"Erreur lors de la création du modèle ResNet: {e}")
        print("Création d'un modèle CNN personnalisé comme solution de secours...")
        return create_custom_cnn_model(input_shape, num_classes, learning_rate)

def create_callbacks(checkpoint_path="models/best_model.h5", save_weights_only=True):
    """
    Crée des callbacks pour l'entraînement du modèle
    
    Args:
        checkpoint_path: Chemin pour sauvegarder le meilleur modèle
        save_weights_only: Si True, sauvegarde uniquement les poids (évite les problèmes de sérialisation)
        
    Returns:
        list: Liste de callbacks
    """
    # Callback pour sauvegarder le meilleur modèle
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=save_weights_only  # Sauvegarde uniquement les poids pour éviter les problèmes
    )
    
    # Callback pour arrêter l'entraînement si aucune amélioration
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Callback pour réduire le taux d'apprentissage si aucune amélioration
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Callback pour journaliser l'entraînement
    csv_logger = tf.keras.callbacks.CSVLogger(
        'training_log.csv', 
        separator=',', 
        append=False
    )
    
    return [model_checkpoint, early_stopping, reduce_lr, csv_logger]

def get_available_models():
    """
    Retourne la liste des modèles disponibles
    
    Returns:
        dict: Dictionnaire des modèles disponibles
    """
    return {
        'efficientnet': create_efficientnet_model,
        'mobilenet': create_mobilenet_model,
        'resnet': create_resnet_model,
        'custom_cnn': create_custom_cnn_model
    }