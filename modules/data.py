import streamlit as st
import pandas as pd
import plotly.express as px
import os
from PIL import Image

def render_data_page(data):
    st.title("Exploration des données")
    
    # Nouvelle section: Description du dataset
    st.markdown("""
    <div style="background-color: #f8f9fb; padding: 20px; border-radius: 10px; border-left: 5px solid #903e97; margin-bottom: 25px;">
        <h3 style="color: #903e97; margin-top: 0;">Présentation du dataset</h3>
        <p style="font-size: 1.05rem; line-height: 1.6;">
            Le dataset <strong>African Plums</strong> est une collection de <strong>4507 images annotées</strong> de prunes africaines 
            (Dacryodes edulis, également appelées "Safou") collectées à travers diverses régions du Cameroun. 
            Il s'agit du premier dataset spécifiquement conçu pour l'évaluation de la qualité de ce fruit par intelligence artificielle.
        </p>
        <h4 style="color: #505050;">Caractéristiques principales</h4>
        <ul style="font-size: 1.05rem;">
            <li>Catégorisé en <strong>six niveaux de qualité</strong>: non affectée (bonne qualité), meurtrie, fissurée, pourrie, tachetée et non mûre</li>
            <li>Capturé sous lumière naturelle à l'aide d'un smartphone Tecno Camon 12</li>
            <li>Images soigneusement étiquetées par des experts agricoles</li>
            <li>Collectées dans 8 villes différentes du Cameroun, couvrant 3 régions agro-écologiques</li>
        </ul>     
        <h4 style="color: #505050;">Applications potentielles</h4>
        <p style="font-size: 1.05rem; line-height: 1.6;">
            Ce dataset est précieux pour développer et tester des systèmes de vision par ordinateur, des modèles de deep learning 
            et de détection d'objets en agriculture, permettant une évaluation automatisée de la qualité des prunes pour leur 
            commercialisation.
        </p>        
        <div style="font-size: 0.9rem; font-style: italic; margin-top: 15px; color: #666;">
            Source: Fadja, A.N., Tagni, A.G.F., Che, S.R., Atemkeng, M. (2025). 
            A dataset of annotated African plum images from Cameroon for AI-based quality assessment. 
            <em>Data in Brief, 59</em>, 111351.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if data is not None:
        st.markdown("### Aperçu du jeu de données")
        st.dataframe(data.head(10))
        
        st.markdown("### Informations sur le jeu de données")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Nombre total d'images", len(data))
            
            # Distribution des classes
            label_counts = data['Defect Type'].value_counts()
            fig = px.pie(
                names=label_counts.index,
                values=label_counts.values,
                title="Distribution des catégories de prunes",
                color=label_counts.index,
                color_discrete_map={
                    'unaffected': '#3CB371',  # Vert
                    'unripe': '#98FB98',      # Vert clair
                    'spotted': '#FFD700',     # Jaune
                    'cracked': '#FFA500',     # Orange
                    'bruised': '#FF6347',     # Rouge-orangé
                    'rotten': '#8B0000'       # Rouge foncé
                }
            )
            st.plotly_chart(fig)
        
        with col2:
            st.markdown("### Statistiques par catégorie")
            stats_df = pd.DataFrame({
                'Catégorie': label_counts.index,
                'Nombre d\'images': label_counts.values,
                'Pourcentage': (label_counts.values / len(data) * 100).round(2)
            })
            st.dataframe(stats_df)
            
            # Visualisation en barres
            fig = px.bar(
                stats_df,
                x='Catégorie',
                y='Nombre d\'images',
                color='Catégorie',
                title="Nombre d'images par catégorie",
                color_discrete_map={
                    'unaffected': '#3CB371',
                    'unripe': '#98FB98',
                    'spotted': '#FFD700',
                    'cracked': '#FFA500',
                    'bruised': '#FF6347',
                    'rotten': '#8B0000'
                }
            )
            st.plotly_chart(fig)
        
        # Information sur les lieux de collecte
        st.markdown("### Lieux de collecte des données")
        
        locations_data = {
            'Ville': ['Limbe', 'Buea', 'Douala', 'Yaoundé', 'Ayos', 'Nguiwa Yangamo', 'Bafoussam', 'Ngaoundéré'],
            'Région': ['Sud-Ouest', 'Sud-Ouest', 'Littoral', 'Centre', 'Centre', 'Est', 'Ouest', 'Adamaoua'],
            'Nombre d\'images': [260, 900, 2255, 210, 210, 202, 170, 300]
        }
        
        locations_df = pd.DataFrame(locations_data)
        
        # Visualisation des lieux de collecte
        fig = px.bar(
            locations_df, 
            x='Ville', 
            y='Nombre d\'images',
            color='Région',
            title="Distribution des images par lieu de collecte",
            labels={'Nombre d\'images': 'Nombre d\'images', 'Ville': 'Ville de collecte'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        st.plotly_chart(fig)
        
        # Exemples d'images
        st.markdown("### Exemples d'images par catégorie")
        categories = ['unaffected', 'unripe', 'spotted', 'cracked', 'bruised', 'rotten']
        category_names_fr = {
            'unaffected': 'Bonne qualité',
            'unripe': 'Non mûre',
            'spotted': 'Tachetée',
            'cracked': 'Fissurée',
            'bruised': 'Meurtrie',
            'rotten': 'Pourrie'
        }
        
        cols = st.columns(len(categories))
        for i, category in enumerate(categories):
            try:
                # Construire le chemin correctement avec os.path.join
                category_path = os.path.join("data", "african_plums_dataset", category)
                
                # Vérifier si le dossier existe
                if not os.path.exists(category_path):
                    cols[i].error(f"Dossier introuvable: {category_path}")
                    continue
                    
                # Lister les images dans ce dossier
                images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if images:
                    # Prendre la première image
                    example_image = images[0]
                    example_path = os.path.join(category_path, example_image)
                    
                    img = Image.open(example_path)
                    cols[i].image(img, caption=f"{category_names_fr.get(category, category)} ({category})", use_container_width=True)
                else:
                    cols[i].error(f"Aucune image trouvée pour {category}")
            except Exception as e:
                cols[i].error(f"Erreur pour {category}: {str(e)}")
        
         # Descriptif des catégories
        st.markdown("### Description détaillée des catégories")

        # Style moderne avec cartes en grille
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 20px;">
            <!-- Bonne qualité -->
            <div style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow: hidden; transition: transform 0.3s;">
                <div style="background-color: #3CB371; padding: 15px; color: white;">
                    <h4 style="margin: 0; display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 10px;">✅</span>
                        Bonne qualité (Unaffected)
                    </h4>
                </div>
                <div style="padding: 15px; background-color: white;">
                    <p style="margin: 0; color: #333;">Prunes en condition optimale sans défauts visibles. Cette catégorie est idéale pour la vente sur le marché car elle représente des produits de haute qualité avec la meilleure apparence et durée de conservation.</p>
                </div>
            </div>         
            <!-- Non mûre -->
            <div style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow: hidden; transition: transform 0.3s;">
                <div style="background-color: #98FB98; padding: 15px; color: #333;">
                    <h4 style="margin: 0; display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 10px;">🟢</span>
                        Non mûre (Unripe)
                    </h4>
                </div>
                <div style="padding: 15px; background-color: white;">
                    <p style="margin: 0; color: #333;">Prunes qui n'ont pas encore atteint leur maturité complète, généralement plus fermes et moins savoureuses. Bien que pas encore prêtes pour une consommation immédiate, ces prunes peuvent être vendues aux acheteurs qui préfèrent faire mûrir les fruits chez eux.</p>
                </div>
            </div>          
            <!-- Tachetée -->
            <div style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow: hidden; transition: transform 0.3s;">
                <div style="background-color: #FFD700; padding: 15px; color: #333;">
                    <h4 style="margin: 0; display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 10px;">🟡</span>
                        Tachetée (Spotted)
                    </h4>
                </div>
                <div style="padding: 15px; background-color: white;">
                    <p style="margin: 0; color: #333;">Prunes avec des taches visibles qui peuvent être causées par des infections fongiques, des dommages d'insectes ou d'autres défauts mineurs de surface. Bien que les taches n'affectent pas toujours le goût, elles peuvent réduire l'attrait pour les consommateurs.</p>
                </div>
            </div>       
            <!-- Fissurée -->
            <div style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow: hidden; transition: transform 0.3s;">
                <div style="background-color: #FFA500; padding: 15px; color: white;">
                    <h4 style="margin: 0; display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 10px;">🟠</span>
                        Fissurée (Cracked)
                    </h4>
                </div>
                <div style="padding: 15px; background-color: white;">
                    <p style="margin: 0; color: #333;">Prunes avec des fentes ou des fissures dans la peau. Les fissures résultent généralement de facteurs environnementaux comme une humidité excessive ou une croissance inégale, ce qui peut rendre le fruit plus susceptible à une détérioration rapide.</p>
                </div>
            </div>      
            <!-- Meurtrie -->
            <div style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow: hidden; transition: transform 0.3s;">
                <div style="background-color: #FF6347; padding: 15px; color: white;">
                    <h4 style="margin: 0; display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 10px;">🔴</span>
                        Meurtrie (Bruised)
                    </h4>
                </div>
                <div style="padding: 15px; background-color: white;">
                    <p style="margin: 0; color: #333;">Prunes qui montrent des signes de dommages physiques, souvent caractérisés par des zones sombres et ramollies. Les meurtrissures peuvent survenir pendant la manipulation ou le transport, affectant à la fois l'attrait visuel et la durée de conservation.</p>
                </div>
            </div>
            <!-- Pourrie -->
            <div style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow: hidden; transition: transform 0.3s;">
                <div style="background-color: #8B0000; padding: 15px; color: white;">
                    <h4 style="margin: 0; display: flex; align-items: center;">
                        <span style="font-size: 1.5rem; margin-right: 10px;">⚫</span>
                        Pourrie (Rotten)
                    </h4>
                </div>
                <div style="padding: 15px; background-color: white;">
                    <p style="margin: 0; color: #333;">Prunes qui présentent une décomposition visible, souvent accompagnée de changements de couleur, de texture et d'odeur. Les prunes pourries sont généralement impropres à la vente en raison de problèmes de qualité et de santé.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)