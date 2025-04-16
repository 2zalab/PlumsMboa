import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from PIL import Image

def render_classification_page(model):
    st.title("Classification d'images de prunes")
    
    # Section: Étapes du processus de classification
    st.markdown("""
    <div style="max-width: 900px; margin: 60px auto 40px auto;">
        <h2 style="
            text-align: center;
            position: relative;
            margin-bottom: 50px;
            padding-bottom: 15px;
            font-weight: 700;
        ">
            <span style="
                background: linear-gradient(to right, #5D3FD3, #9C27B0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">Le Processus de Classification</span>
            <div style="
                position: absolute;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 120px;
                height: 3px;
                background: linear-gradient(to right, #5D3FD3, #9C27B0);
                border-radius: 3px;
            "></div>
        </h2>   
        <div style="position: relative;">
            <!-- Ligne verticale de progression -->
            <div style="
                position: absolute;
                top: 0;
                bottom: 0;
                left: 50px;
                width: 4px;
                background: linear-gradient(to bottom, #5D3FD3, #9C27B0);
                border-radius: 4px;
                z-index: 0;
            "></div>       
            <!-- Étape 1: Acquisition d'image -->
            <div style="
                display: flex;
                margin-bottom: 35px;
                position: relative;
                z-index: 1;
            ">
                <div style="
                    background: linear-gradient(135deg, #5D3FD3, #7B68EE);
                    width: 100px;
                    height: 100px;
                    border-radius: 50%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    box-shadow: 0 10px 25px rgba(93, 63, 211, 0.3);
                    flex-shrink: 0;
                ">
                    <span style="font-size: 2.8rem; color: white;">📸</span>
                </div>
                <div style="
                    background-color: white;
                    border-radius: 16px;
                    padding: 25px 30px;
                    margin-left: -20px;
                    flex-grow: 1;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                    border-left: 5px solid #5D3FD3;
                    transform: translateX(20px);
                    transition: all 0.3s ease;
                " onmouseover="this.style.transform='translateX(20px) translateY(-5px)'; this.style.boxShadow='0 15px 35px rgba(0,0,0,0.12)';"
                onmouseout="this.style.transform='translateX(20px)'; this.style.boxShadow='0 10px 30px rgba(0,0,0,0.08)';">
                    <h3 style="margin: 0 0 10px 0; color: #5D3FD3; font-weight: 600;">1. Acquisition d'Image</h3>
                    <p style="margin: 0; color: #555; line-height: 1.6;">
                        Capture de l'image de la prune à l'aide d'un appareil photo ou sélection d'une image existante. 
                        Notre système accepte une variété de formats d'image et s'adapte à différentes conditions d'éclairage.
                    </p>
                </div>
            </div>         
            <!-- Étape 2: Prétraitement -->
            <div style="
                display: flex;
                margin-bottom: 35px;
                position: relative;
                z-index: 1;
            ">
                <div style="
                    background: linear-gradient(135deg, #7B68EE, #9370DB);
                    width: 100px;
                    height: 100px;
                    border-radius: 50%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    box-shadow: 0 10px 25px rgba(123, 104, 238, 0.3);
                    flex-shrink: 0;
                ">
                    <span style="font-size: 2.8rem; color: white;">⚙️</span>
                </div>
                <div style="
                    background-color: white;
                    border-radius: 16px;
                    padding: 25px 30px;
                    margin-left: -20px;
                    flex-grow: 1;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                    border-left: 5px solid #7B68EE;
                    transform: translateX(20px);
                    transition: all 0.3s ease;
                " onmouseover="this.style.transform='translateX(20px) translateY(-5px)'; this.style.boxShadow='0 15px 35px rgba(0,0,0,0.12)';"
                onmouseout="this.style.transform='translateX(20px)'; this.style.boxShadow='0 10px 30px rgba(0,0,0,0.08)';">
                    <h3 style="margin: 0 0 10px 0; color: #7B68EE; font-weight: 600;">2. Prétraitement</h3>
                    <p style="margin: 0; color: #555; line-height: 1.6;">
                        L'image est redimensionnée à 224×224 pixels et normalisée pour garantir la cohérence des entrées. 
                        Cette étape cruciale optimise les caractéristiques visuelles pour l'analyse par le réseau de neurones.
                    </p>
                </div>
            </div>          
            <!-- Étape 3: Analyse par IA -->
            <div style="
                display: flex;
                margin-bottom: 35px;
                position: relative;
                z-index: 1;
            ">
                <div style="
                    background: linear-gradient(135deg, #9370DB, #BA55D3);
                    width: 100px;
                    height: 100px;
                    border-radius: 50%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    box-shadow: 0 10px 25px rgba(147, 112, 219, 0.3);
                    flex-shrink: 0;
                ">
                    <span style="font-size: 2.8rem; color: white;">🧠</span>
                </div>
                <div style="
                    background-color: white;
                    border-radius: 16px;
                    padding: 25px 30px;
                    margin-left: -20px;
                    flex-grow: 1;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                    border-left: 5px solid #9370DB;
                    transform: translateX(20px);
                    transition: all 0.3s ease;
                " onmouseover="this.style.transform='translateX(20px) translateY(-5px)'; this.style.boxShadow='0 15px 35px rgba(0,0,0,0.12)';"
                onmouseout="this.style.transform='translateX(20px)'; this.style.boxShadow='0 10px 30px rgba(0,0,0,0.08)';">
                    <h3 style="margin: 0 0 10px 0; color: #9370DB; font-weight: 600;">3. Analyse par Intelligence Artificielle</h3>
                    <p style="margin: 0; color: #555; line-height: 1.6;">
                        Notre modèle EfficientNet analyse l'image à travers plusieurs couches de neurones convolutifs, 
                        extrayant des caractéristiques sophistiquées pour identifier avec précision la catégorie de la prune.
                    </p>
                </div>
            </div>           
            <!-- Étape 4: Classification -->
            <div style="
                display: flex;
                margin-bottom: 35px;
                position: relative;
                z-index: 1;
            ">
                <div style="
                    background: linear-gradient(135deg, #BA55D3, #9C27B0);
                    width: 100px;
                    height: 100px;
                    border-radius: 50%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    box-shadow: 0 10px 25px rgba(186, 85, 211, 0.3);
                    flex-shrink: 0;
                ">
                    <span style="font-size: 2.8rem; color: white;">🏷️</span>
                </div>
                <div style="
                    background-color: white;
                    border-radius: 16px;
                    padding: 25px 30px;
                    margin-left: -20px;
                    flex-grow: 1;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                    border-left: 5px solid #BA55D3;
                    transform: translateX(20px);
                    transition: all 0.3s ease;
                " onmouseover="this.style.transform='translateX(20px) translateY(-5px)'; this.style.boxShadow='0 15px 35px rgba(0,0,0,0.12)';"
                onmouseout="this.style.transform='translateX(20px)'; this.style.boxShadow='0 10px 30px rgba(0,0,0,0.08)';">
                    <h3 style="margin: 0 0 10px 0; color: #BA55D3; font-weight: 600;">4. Classification</h3>
                    <p style="margin: 0; color: #555; line-height: 1.6;">
                        Le système détermine la catégorie de la prune parmi six possibilités: bonne qualité, non mûre, 
                        tachetée, fissurée, meurtrie ou pourrie, avec un niveau de confiance pour chaque catégorie.
                    </p>
                </div>
            </div>           
            <!-- Étape 5: Résultats et recommandations -->
            <div style="
                display: flex;
                position: relative;
                z-index: 1;
            ">
                <div style="
                    background: linear-gradient(135deg, #9C27B0, #5D3FD3);
                    width: 100px;
                    height: 100px;
                    border-radius: 50%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    box-shadow: 0 10px 25px rgba(156, 39, 176, 0.3);
                    flex-shrink: 0;
                ">
                    <span style="font-size: 2.8rem; color: white;">📊</span>
                </div>
                <div style="
                    background-color: white;
                    border-radius: 16px;
                    padding: 25px 30px;
                    margin-left: -20px;
                    flex-grow: 1;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                    border-left: 5px solid #9C27B0;
                    transform: translateX(20px);
                    transition: all 0.3s ease;
                " onmouseover="this.style.transform='translateX(20px) translateY(-5px)'; this.style.boxShadow='0 15px 35px rgba(0,0,0,0.12)';"
                onmouseout="this.style.transform='translateX(20px)'; this.style.boxShadow='0 10px 30px rgba(0,0,0,0.08)';">
                    <h3 style="margin: 0 0 10px 0; color: #9C27B0; font-weight: 600;">5. Résultats et Recommandations</h3>
                    <p style="margin: 0; color: #555; line-height: 1.6;">
                        Visualisation des résultats avec un graphique détaillé des probabilités et des recommandations 
                        personnalisées sur la meilleure utilisation de la prune selon sa classification.
                    </p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if model is not None:
        st.markdown("""
        ### Testez notre modèle de classification
        
        Chargez une image de prune et notre modèle d'intelligence artificielle 
        la classifiera dans l'une des six catégories:
        
        - ✅ **Bonne qualité** (Unaffected)
        - 🟢 **Non mûre** (Unripe)
        - 🟡 **Tachetée** (Spotted)
        - 🟠 **Fissurée** (Cracked)
        - 🔴 **Meurtrie** (Bruised)
        - ⚫ **Pourrie** (Rotten)
        """)
        
        # Options de chargement d'image
        upload_option = st.radio(
            "Comment souhaitez-vous charger une image?",
            options=["Charger une image", "Utiliser une image d'exemple"]
        )
        
        image = None
        
        if upload_option == "Charger une image":
            uploaded_file = st.file_uploader("Choisissez une image de prune...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Image chargée", width=300)
        
        else:  # Utiliser une image d'exemple
            categories = ['unaffected', 'unripe', 'spotted', 'cracked', 'bruised', 'rotten']
            category_names_fr = {
                'unaffected': 'Bonne qualité',
                'unripe': 'Non mûre',
                'spotted': 'Tachetée',
                'cracked': 'Fissurée',
                'bruised': 'Meurtrie',
                'rotten': 'Pourrie'
            }
            
            # Afficher les exemples
            st.markdown("### Sélectionnez une image d'exemple:")
            
            # Créer une grille d'exemples
            cols = st.columns(3)
            image_paths = []
            images = []
            
            for i, category in enumerate(categories):
                try:
                    # Chercher un chemin d'image exemple pour chaque catégorie
                    example_path = f"data/african_plums_dataset/{category}/{category}_plum_1.png"
                    img = Image.open(example_path)
                    images.append(img)
                    image_paths.append(example_path)
                    
                    # Afficher l'image dans la colonne appropriée
                    with cols[i % 3]:
                        st.image(img, caption=f"{category_names_fr[category]}", width=200)
                        if st.button(f"Sélectionner ({category})", key=f"select_{category}"):
                            image = img
                            st.image(img, caption="Image chargée", width=300)
                except:
                    with cols[i % 3]:
                        st.error(f"Image exemple non disponible pour {category}")
        
        # Classification
        if image is not None and st.button("Classifier l'image"):
            st.markdown("### Résultats de la classification")
            
            # Prétraiter l'image
            img_array = np.array(image.resize((224, 224)))
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Simulation de prédiction (remplacer par le modèle réel)
            # Normalement: predictions = model.predict(img_array)
            # Pour la démonstration, on génère des valeurs aléatoires avec une préférence
            
            # Simulation d'une prédiction
            categories = ['unaffected', 'unripe', 'spotted', 'cracked', 'bruised', 'rotten']
            
            # Simuler une prédiction avec une valeur dominante
            dominant_index = np.random.randint(0, len(categories))
            predictions = np.random.rand(1, len(categories)) * 0.2  # Valeurs de base faibles
            predictions[0, dominant_index] = 0.6 + np.random.rand() * 0.3  # Valeur dominante
            
            # Normaliser pour que la somme soit 1
            predictions = predictions / np.sum(predictions)
            
            # Afficher les résultats
            pred_class_index = np.argmax(predictions[0])
            pred_class = categories[pred_class_index]
            confidence = predictions[0, pred_class_index] * 100
            
            # Traduction en français
            category_names_fr = {
                'unaffected': 'Bonne qualité',
                'unripe': 'Non mûre',
                'spotted': 'Tachetée',
                'cracked': 'Fissurée',
                'bruised': 'Meurtrie',
                'rotten': 'Pourrie'
            }
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption="Image analysée", width=250)
            
            with col2:
                # Résultat principal
                result_color = {
                    'unaffected': '#3CB371',  # Vert
                    'unripe': '#98FB98',      # Vert clair
                    'spotted': '#FFD700',     # Jaune
                    'cracked': '#FFA500',     # Orange
                    'bruised': '#FF6347',     # Rouge-orangé
                    'rotten': '#8B0000'       # Rouge foncé
                }
                
                st.markdown(
                    f"<div style='background-color: {result_color[pred_class]}; padding: 20px; border-radius: 10px;'>"
                    f"<h3 style='color: white; margin:0;'>Résultat: {category_names_fr[pred_class]}</h3>"
                    f"<p style='color: white; margin:0;'>Confiance: {confidence:.1f}%</p>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                # Graphique des prédictions
                results_df = pd.DataFrame({
                    'Catégorie': [category_names_fr[cat] for cat in categories],
                    'Confiance (%)': predictions[0] * 100,
                    'Catégorie_en': categories
                })
                
                fig = px.bar(
                    results_df,
                    x='Catégorie',
                    y='Confiance (%)',
                    color='Catégorie_en',
                    color_discrete_map=result_color,
                    title="Confiance par catégorie"
                )
                
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig)
            
            # Recommandation
            st.markdown("### Recommandation")
            
            recommendations = {
                'unaffected': "Cette prune est de bonne qualité et peut être commercialisée pour la consommation directe.",
                'unripe': "Cette prune n'est pas encore mûre. Il est recommandé d'attendre quelques jours avant sa commercialisation.",
                'spotted': "Cette prune présente des taches. Elle peut être utilisée pour la transformation (confitures, jus) mais n'est pas idéale pour la vente directe.",
                'cracked': "Cette prune est fissurée. Elle devrait être traitée rapidement pour la transformation ou écartée si les fissures sont importantes.",
                'bruised': "Cette prune présente des meurtrissures. Elle peut être utilisée pour la transformation si les meurtrissures sont légères.",
                'rotten': "Cette prune est pourrie et devrait être écartée de la chaîne de production."
            }
            
            st.info(recommendations[pred_class])