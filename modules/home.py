import streamlit as st

def render_homepage():
    
    # Titre principal avec dégradé et centré
    st.markdown(f"""
    <div style="text-align: center;">
        <h1 style='
            color: #5D3FD3; 
            background: linear-gradient(to right, #5D3FD3, #9C27B0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
            margin-bottom: 30px;
            font-size: 2.5rem;
            letter-spacing: 0.5px;
            text-shadow: 0px 4px 8px rgba(93, 63, 211, 0.2);
        '>
            PlumsMboa: Tri Automatique des Prunes
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Section de bienvenue avec fond et style moderne
    st.markdown("""
        <div 
            style='
            text-align: center; 
            max-width: 900px; 
            margin: 0 auto 30px auto; 
            padding: 25px; 
        '>
            <h3 style='
                margin-top: 0; 
                color: #333; 
                font-weight: 600;
                letter-spacing: 0.5px;
            '>🌟 Bienvenue sur PlumsMboa!</h3>
          <p style="font-size: 1.15em; margin: 15px 0 25px 0;  color: #444;  line-height: 1.6;">
                Notre application révolutionnaire utilise <strong>l'intelligence artificielle</strong> pour classifier 
                automatiquement les prunes africaines avec une précision remarquable.
            </p>
           
    </div>
    
    """, unsafe_allow_html=True)
    
    # Titre des catégories
    st.markdown("<h3 style='text-align:center; margin-bottom:25px;'>Nos Catégories de Classification</h3>", unsafe_allow_html=True)

    # Définition des catégories
    categories = [
        {"icon": "✅", "title": "Bonne qualité", "desc": "Prunes parfaites", "color": "#3CB371"},
        {"icon": "🟢", "title": "Non mûre", "desc": "Prunes pas encore à point", "color": "#98FB98"},
        {"icon": "🟡", "title": "Tachetée", "desc": "Prunes avec des marques légères", "color": "#FFD700"},
        {"icon": "🟠", "title": "Fissurée", "desc": "Prunes présentant des fissures", "color": "#FFA500"},
        {"icon": "🔴", "title": "Meurtrie", "desc": "Prunes endommagées", "color": "#FF6347"},
        {"icon": "⚫", "title": "Pourrie", "desc": "Prunes non comestibles", "color": "#8B0000"}
    ]

    # Créer des colonnes pour les cartes (3 cartes par ligne)
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    # Parcourir les catégories et afficher dans les colonnes
    for i, cat in enumerate(categories):
        col_index = i % 3  # Détermine quelle colonne utiliser (0, 1 ou 2)
        
        # Utiliser la colonne appropriée
        with cols[col_index]:
            # Créer une carte HTML simple pour chaque catégorie
            st.markdown(f"""
            <div style="
                background-color: white;
                border-radius: 10px;
                border-top: 3px solid {cat['color']};
                padding: 15px;
                text-align: center;
                box-shadow: 0 4px 10px rgba(0,0,0,0.05);
                margin-bottom: 15px;
            ">
                <div style="font-size: 24px; margin-bottom: 10px;">{cat['icon']}</div>
                <div style="font-weight: bold; color: {cat['color']}; margin-bottom: 8px;">{cat['title']}</div>
                <div style="font-size: 0.9em; color: #666;">{cat['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # NOUVELLE SECTION: Comment utiliser cette plateforme
    st.markdown("""
    <div style='
        max-width: 900px;
        margin: 40px auto;
        padding: 0;
    '>
        <h2 style='
            text-align: center;
            color: #333;
            margin-bottom: 25px;
            font-weight: 600;
            position: relative;
        '>
            <span style='
                background: linear-gradient(to right, #5D3FD3, #9C27B0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            '>Comment utiliser cette plateforme?</span>
        </h2>   
        <div style='
            display: flex;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        '>
            <!-- Étape 1 -->
            <div style='
                background-color: white;
                border-radius: 16px;
                box-shadow: 0 8px 24px rgba(149, 157, 165, 0.15);
                padding: 25px;
                width: 100%;
                max-width: 900px;
                display: flex;
                align-items: center;
                gap: 20px;
                margin-bottom: 15px;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            ' onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 12px 30px rgba(149, 157, 165, 0.2)';"
              onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 8px 24px rgba(149, 157, 165, 0.15)';">
                <div style='
                    background-color: #E8F0FE;
                    color: #2196F3;
                    width: 50px;
                    height: 50px;
                    border-radius: 50%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    font-size: 1.5rem;
                    font-weight: bold;
                    flex-shrink: 0;
                '>1</div>
                <div>
                    <h3 style='margin: 0 0 8px 0; color: #333;'>Explorer les données</h3>
                    <p style='margin: 0; color: #555; line-height: 1.5;'>
                        Commencez par explorer la page <strong>Données</strong> pour comprendre la distribution des différentes catégories de prunes et visualiser des exemples de chaque type.
                    </p>
                </div>
            </div>         
            <!-- Étape 2 -->
            <div style='
                background-color: white;
                border-radius: 16px;
                box-shadow: 0 8px 24px rgba(149, 157, 165, 0.15);
                padding: 25px;
                width: 100%;
                max-width: 900px;
                display: flex;
                align-items: center;
                gap: 20px;
                margin-bottom: 15px;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            ' onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 12px 30px rgba(149, 157, 165, 0.2)';"
              onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 8px 24px rgba(149, 157, 165, 0.15)';">
                <div style='
                    background-color: #F3E5F5;
                    color: #9C27B0;
                    width: 50px;
                    height: 50px;
                    border-radius: 50%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    font-size: 1.5rem;
                    font-weight: bold;
                    flex-shrink: 0;
                '>2</div>
                <div>
                    <h3 style='margin: 0 0 8px 0; color: #333;'>Découvrir le modèle</h3>
                    <p style='margin: 0; color: #555; line-height: 1.5;'>
                        Visitez la page <strong>Modèle</strong> pour comprendre l'architecture et les performances du système d'intelligence artificielle que nous utilisons pour classifier les prunes africaines.
                    </p>
                </div>
            </div>           
            <!-- Étape 3 -->
            <div style='
                background-color: white;
                border-radius: 16px;
                box-shadow: 0 8px 24px rgba(149, 157, 165, 0.15);
                padding: 25px;
                width: 100%;
                max-width: 900px;
                display: flex;
                align-items: center;
                gap: 20px;
                margin-bottom: 15px;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            ' onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 12px 30px rgba(149, 157, 165, 0.2)';"
              onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 8px 24px rgba(149, 157, 165, 0.15)';">
                <div style='
                    background-color: #E8F5E9;
                    color: #4CAF50;
                    width: 50px;
                    height: 50px;
                    border-radius: 50%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    font-size: 1.5rem;
                    font-weight: bold;
                    flex-shrink: 0;
                '>3</div>
                <div>
                    <h3 style='margin: 0 0 8px 0; color: #333;'>Tester la classification</h3>
                    <p style='margin: 0; color: #555; line-height: 1.5;'>
                        Rendez-vous sur la page <strong>Classification</strong> pour tester notre modèle en temps réel. Vous pouvez soit charger vos propres images de prunes, soit utiliser nos exemples préchargés pour voir le système en action.
                    </p>
                </div>
            </div>          
            <!-- Étape 4 -->
            <div style='
                background-color: white;
                border-radius: 16px;
                box-shadow: 0 8px 24px rgba(149, 157, 165, 0.15);
                padding: 25px;
                width: 100%;
                max-width: 900px;
                display: flex;
                align-items: center;
                gap: 20px;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            ' onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 12px 30px rgba(149, 157, 165, 0.2)';"
              onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 8px 24px rgba(149, 157, 165, 0.15)';">
                <div style='
                    background-color: #FFF3E0;
                    color: #FF9800;
                    width: 50px;
                    height: 50px;
                    border-radius: 50%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    font-size: 1.5rem;
                    font-weight: bold;
                    flex-shrink: 0;
                '>4</div>
                <div>
                    <h3 style='margin: 0 0 8px 0; color: #333;'>En savoir plus</h3>
                    <p style='margin: 0; color: #555; line-height: 1.5;'>
                        Pour plus d'informations sur notre projet et notre équipe, consultez la page <strong>À propos</strong>. Vous y trouverez des détails sur nos objectifs, la technologie utilisée et les perspectives futures.
                    </p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
  
    # Message de conseil avec style moderne
    st.markdown("""
    <div style='
        background-color: #E6F2FF; 
        border-left: 4px solid #2196F3; 
        padding: 15px; 
        border-radius: 8px;
        margin: 20px auto 30px auto;
        max-width: 900px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    '>
        <div style="display: flex; align-items: center;">
            <div style="font-size: 1.5em; margin-right: 10px;">💡</div>
            <div>
                <strong style="color: #0D47A1;">Conseil</strong>: 
                <span style="color: #333;">Utilisez le menu de navigation à gauche pour explorer l'application.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Séparateur avec dégradé
    st.markdown("""
    <hr style='
        border: 0; 
        height: 2px; 
        background: linear-gradient(to right, #5D3FD3, #9C27B0);
        margin: 30px auto;
        max-width: 900px;
    '>
    """, unsafe_allow_html=True)
    
    # Titre de la section fonctionnalités
    st.markdown("""
    <h2 style='
        text-align: center;
        color: #333;
        margin-bottom: 30px;
        font-weight: 600;
    '>
        🚀 Aperçu des Fonctionnalités
    </h2>
    """, unsafe_allow_html=True)
    
    # Style personnalisé pour les boutons
    st.markdown("""
    <style>
        /* Personnalisation des boutons dans les cartes */
        div[data-testid="stButton"] > button {
            border-radius: 5px !important;
            padding: 0.4rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Couleurs personnalisées pour les boutons */
        .explore-data-button button {
            background-color: #2196F3 !important;
            color: white !important;
        }
        
        .view-model-button button {
            background-color: #9C27B0 !important;
            color: white !important;
        }
        
        .test-classification-button button {
            background-color: #4CAF50 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Cartes de fonctionnalités en colonnes
    col1, col2, col3 = st.columns(3)
    
    # Carte 1: Analyse des données
    with col1:
        st.markdown("""
        <div style='
            background-color: white;
            border-radius: 12px;
            padding: 20px 20px 10px 20px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            height: 100%;
            border-top: 4px solid #2196F3;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        ' onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 10px 25px rgba(0,0,0,0.1)';" 
          onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 5px 15px rgba(0,0,0,0.05)';">
            <div style='font-size: 2.5rem; color: #2196F3; margin-bottom: 15px;'>📊</div>
            <h4 style='color: #2196F3; margin-bottom: 12px; font-weight: 600;'>Analyse des Données</h4>
            <p style='margin-bottom: 20px; color: #555; line-height: 1.5;'>
                Explorez la distribution des classes et visualisez les caractéristiques détaillées du jeu de données.
            </p>
        """, unsafe_allow_html=True)
        
        # Bouton intégré à la carte avec classe CSS personnalisée
        with st.container():
            st.markdown('<div class="explore-data-button">', unsafe_allow_html=True)
            if st.button("Explorer les données", key="explore_data", use_container_width=True):
                st.session_state.selected = "Données"
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Fermer la div
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Carte 2: Performance du modèle
    with col2:
        st.markdown("""
        <div style='
            background-color: white;
            border-radius: 12px;
            padding: 20px 20px 10px 20px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            height: 100%;
            border-top: 4px solid #9C27B0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        ' onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 10px 25px rgba(0,0,0,0.1)';" 
          onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 5px 15px rgba(0,0,0,0.05)';">
            <div style='font-size: 2.5rem; color: #9C27B0; margin-bottom: 15px;'>🧠</div>
            <h4 style='color: #9C27B0; margin-bottom: 12px; font-weight: 600;'>Performance du Modèle</h4>
            <p style='margin-bottom: 20px; color: #555; line-height: 1.5;'>
                Découvrez les performances et le fonctionnement de notre système de classification.
            </p>
        """, unsafe_allow_html=True)
        
        # Bouton intégré à la carte avec classe CSS personnalisée
        with st.container():
            st.markdown('<div class="view-model-button">', unsafe_allow_html=True)
            if st.button("Voir le modèle", key="view_model", use_container_width=True):
                st.session_state.selected = "Modèle"
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Fermer la div
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Carte 3: Classification d'images
    with col3:
        st.markdown("""
        <div style='
            background-color: white;
            border-radius: 12px;
            padding: 20px 20px 10px 20px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            height: 100%;
            border-top: 4px solid #4CAF50;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        ' onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 10px 25px rgba(0,0,0,0.1)';" 
          onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 5px 15px rgba(0,0,0,0.05)';">
            <div style='font-size: 2.5rem; color: #4CAF50; margin-bottom: 15px;'>🔍</div>
            <h4 style='color: #4CAF50; margin-bottom: 12px; font-weight: 600;'>Classification d'Images</h4>
            <p style='margin-bottom: 20px; color: #555; line-height: 1.5;'>
                Testez notre modèle en chargeant vos propres images de prunes africaines.
            </p>
        """, unsafe_allow_html=True)
        
        # Bouton intégré à la carte avec classe CSS personnalisée
        with st.container():
            st.markdown('<div class="test-classification-button">', unsafe_allow_html=True)
            if st.button("Tester la classification", key="test_classification", use_container_width=True):
                st.session_state.selected = "Classification"
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Fermer la div
        st.markdown("</div>", unsafe_allow_html=True)