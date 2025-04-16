import streamlit as st

def render_about_page():
    # Titre principal avec dégradé et style élégant
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h1 style='
            color: #5D3FD3; 
            background: linear-gradient(to right, #5D3FD3, #9C27B0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
            font-size: 2.5rem;
            letter-spacing: 0.5px;
            text-shadow: 0px 4px 8px rgba(93, 63, 211, 0.2);
            margin-bottom: 10px;
        '>
            À propos de PlumsMboa
        </h1>
        <p style="
            font-size: 1.2rem;
            color: #666;
            max-width: 800px;
            margin: 0 auto;
        ">Une solution innovante pour le tri automatique des prunes africaines</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Bannière du projet
    st.markdown("""
    <div style="
        background: linear-gradient(to right, rgba(93, 63, 211, 0.8), rgba(156, 39, 176, 0.8)), url('https://images.unsplash.com/photo-1512578000375-860e4a4901e3?ixlib=rb-1.2.1&auto=format&fit=crop&q=80');
        background-size: cover;
        background-position: center;
        padding: 40px;
        border-radius: 12px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    ">
        <h2 style="margin-top: 0; font-weight: 600;">Projet pour le Hackathon JCIA 2025</h2>
        <p style="font-size: 1.1rem; max-width: 800px; line-height: 1.6;">
            <b>PlumsMboa</b> est une application innovante de tri automatique des prunes africaines (Safou) développée dans le cadre
            du Hackathon de la Journée de l'Intelligence Artificielle (JCIA) 2025. Notre solution utilise des algorithmes d'apprentissage profond
            pour classifier les prunes en différentes catégories de qualité, aidant ainsi les agriculteurs et distributeurs au Cameroun.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section Objectif du projet avec icône et style de carte
    st.markdown("""
    <div style="
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border-left: 5px solid #5D3FD3;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="
                background-color: #5D3FD3;
                width: 50px;
                height: 50px;
                border-radius: 50%;
                display: flex;
                justify-content: center;
                align-items: center;
                margin-right: 15px;
                flex-shrink: 0;
            ">
                <span style="font-size: 1.5rem; color: white;">🎯</span>
            </div>
            <h2 style="margin: 0; color: #5D3FD3; font-weight: 600;">Objectif du projet</h2>
        </div>
        <p style="font-size: 1.05rem; line-height: 1.6; color: #444; margin-left: 65px;">
            Notre objectif est de développer un système de vision par ordinateur capable de classifier
            automatiquement les prunes africaines en six catégories, afin d'améliorer l'efficacité du tri,
            réduire les pertes post-récolte et augmenter la valeur ajoutée de la production fruitière au Cameroun.
            Cette solution s'inscrit dans une démarche d'innovation technologique au service de l'agriculture africaine.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section Technologies en style de grille avec icônes
    st.markdown("""
    <div style="
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border-left: 5px solid #9C27B0;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div style="
                background-color: #9C27B0;
                width: 50px;
                height: 50px;
                border-radius: 50%;
                display: flex;
                justify-content: center;
                align-items: center;
                margin-right: 15px;
                flex-shrink: 0;
            ">
                <span style="font-size: 1.5rem; color: white;">💻</span>
            </div>
            <h2 style="margin: 0; color: #9C27B0; font-weight: 600;">Technologies utilisées</h2>
        </div>    
        <div style="
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-left: 65px;
        ">
            <!-- Frontend Tech -->
            <div style="
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.03);
                transition: transform 0.2s;
            " onmouseover="this.style.transform='translateY(-5px)'" 
               onmouseout="this.style.transform='translateY(0)'">
                <h3 style="margin-top: 0; color: #2196F3; font-size: 1.1rem; display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 8px;">🖥️</span> Frontend
                </h3>
                <ul style="margin: 0; padding-left: 20px; color: #555;">
                    <li>Streamlit</li>
                    <li>HTML/CSS</li>
                </ul>
            </div>    
            <!-- Backend Tech -->
            <div style="
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.03);
                transition: transform 0.2s;
            " onmouseover="this.style.transform='translateY(-5px)'" 
               onmouseout="this.style.transform='translateY(0)'">
                <h3 style="margin-top: 0; color: #4CAF50; font-size: 1.1rem; display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 8px;">⚙️</span> Backend
                </h3>
                <ul style="margin: 0; padding-left: 20px; color: #555;">
                    <li>Python</li>
                    <li>TensorFlow</li>
                    <li>Keras</li>
                </ul>
            </div>           
            <!-- Image Processing -->
            <div style="
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.03);
                transition: transform 0.2s;
            " onmouseover="this.style.transform='translateY(-5px)'" 
               onmouseout="this.style.transform='translateY(0)'">
                <h3 style="margin-top: 0; color: #FF9800; font-size: 1.1rem; display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 8px;">🖼️</span> Traitement d'images
                </h3>
                <ul style="margin: 0; padding-left: 20px; color: #555;">
                    <li>OpenCV</li>
                    <li>Pillow</li>
                </ul>
            </div>           
            <!-- Visualization -->
            <div style="
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.03);
                transition: transform 0.2s;
            " onmouseover="this.style.transform='translateY(-5px)'" 
               onmouseout="this.style.transform='translateY(0)'">
                <h3 style="margin-top: 0; color: #9C27B0; font-size: 1.1rem; display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 8px;">📊</span> Visualisation
                </h3>
                <ul style="margin: 0; padding-left: 20px; color: #555;">
                    <li>Matplotlib</li>
                    <li>Plotly</li>
                    <li>Seaborn</li>
                </ul>
            </div>          
            <!-- Data Analysis -->
            <div style="
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.03);
                transition: transform 0.2s;
            " onmouseover="this.style.transform='translateY(-5px)'" 
               onmouseout="this.style.transform='translateY(0)'">
                <h3 style="margin-top: 0; color: #F44336; font-size: 1.1rem; display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 8px;">📈</span> Analyse de données
                </h3>
                <ul style="margin: 0; padding-left: 20px; color: #555;">
                    <li>Pandas</li>
                    <li>NumPy</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Section Équipe
    st.markdown("""
    <div style="
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border-left: 5px solid #2196F3;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <div style="
                background-color: #2196F3;
                width: 50px;
                height: 50px;
                border-radius: 50%;
                display: flex;
                justify-content: center;
                align-items: center;
                margin-right: 15px;
                flex-shrink: 0;
            ">
                <span style="font-size: 1.5rem; color: white;">👥</span>
            </div>
            <h2 style="margin: 0; color: #2196F3; font-weight: 600;">Notre Équipe</h2>
        </div>     
        <div style="margin-left: 65px;">
            <div style="
                display: flex;
                align-items: center;
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            ">
                <div style="
                    width: 60px;
                    height: 60px;
                    background-color: #e3f2fd;
                    border-radius: 50%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin-right: 15px;
                    font-size: 1.8rem;
                    color: #2196F3;
                ">🧑‍💻</div>
                <div>
                    <h3 style="margin: 0 0 5px 0; color: #333;">Maroua Innovation Technology (MIT)</h3>
                    <p style="margin: 0; color: #666;">
                        Une équipe dédiée au développement de solutions technologiques innovantes pour l'Afrique
                    </p>
                    <p style="margin: 5px 0 0 0;">
                        <a href="https://www.maroua-it.com" target="_blank" style="
                            display: inline-block;
                            padding: 3px 10px;
                            background-color: #e3f2fd;
                            color: #2196F3;
                            text-decoration: none;
                            border-radius: 15px;
                            font-size: 0.8rem;
                            transition: background-color 0.2s;
                        " onmouseover="this.style.backgroundColor='#bbdefb'" 
                           onmouseout="this.style.backgroundColor='#e3f2fd'">
                            Visiter le site
                        </a>
                    </p>
                </div>
            </div>          
            <div style="
                display: flex;
                align-items: center;
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            ">
                <div style="
                    width: 60px;
                    height: 60px;
                    background-color: #e3f2fd;
                    border-radius: 50%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin-right: 15px;
                    font-size: 1.8rem;
                    color: #2196F3;
                ">👨‍💻</div>
                <div>
                    <h3 style="margin: 0 0 5px 0; color: #333;">Touza Isaac</h3>
                    <p style="margin: 0; color: #666;">Chef d'équipe & Développeur IA full stack</p>
                    <div style="
                        display: flex;
                        gap: 10px;
                        margin-top: 8px;
                    ">
                        <a href="https://github.com/2zalab" target="_blank" style="
                            display: inline-block;
                            padding: 3px 10px;
                            background-color: #e3f2fd;
                            color: #2196F3;
                            text-decoration: none;
                            border-radius: 15px;
                            font-size: 0.8rem;
                            transition: background-color 0.2s;
                        " onmouseover="this.style.backgroundColor='#bbdefb'" 
                           onmouseout="this.style.backgroundColor='#e3f2fd'">
                            <span style="margin-right: 5px;">GitHub</span>
                        </a>
                        <a href="https://www.linkedin.com/in/touzaisaac/" target="_blank" style="
                            display: inline-block;
                            padding: 3px 10px;
                            background-color: #e3f2fd;
                            color: #2196F3;
                            text-decoration: none;
                            border-radius: 15px;
                            font-size: 0.8rem;
                            transition: background-color 0.2s;
                        " onmouseover="this.style.backgroundColor='#bbdefb'" 
                           onmouseout="this.style.backgroundColor='#e3f2fd'">
                            <span style="margin-right: 5px;">LinkedIn</span>
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Section remerciements
    st.markdown("""
    <div style="
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border-left: 5px solid #FF9800;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="
                background-color: #FF9800;
                width: 50px;
                height: 50px;
                border-radius: 50%;
                display: flex;
                justify-content: center;
                align-items: center;
                margin-right: 15px;
                flex-shrink: 0;
            ">
                <span style="font-size: 1.5rem; color: white;">🙏</span>
            </div>
            <h2 style="margin: 0; color: #FF9800; font-weight: 600;">Remerciements</h2>
        </div>
        <p style="font-size: 1.05rem; line-height: 1.6; color: #444; margin-left: 65px;">
            Nous tenons à remercier chaleureusement les organisateurs du <strong>JCIA 2025</strong> pour cette formidable opportunité
            de développer notre solution innovante. Notre gratitude va également au <strong>Dr. Arnaud Nguembang Fadja</strong>
            pour la mise à disposition du jeu de données <em>African Plums Dataset</em>, sans lequel ce projet n'aurait pas été possible.
            Enfin, merci à tous ceux qui ont soutenu notre équipe tout au long de ce hackathon.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer avec liens et contacts
    st.markdown("""
    <div style="
        background: linear-gradient(to right, #5D3FD3, #9C27B0);
        border-radius: 12px;
        padding: 30px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    ">
        <div style="
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 30px;
        ">
            <!-- Liens utiles -->
            <div>
                <h3 style="margin-top: 0; font-weight: 600; display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 10px;">🔗</span> Liens utiles
                </h3>
                <ul style="list-style-type: none; padding-left: 0; margin: 0;">
                    <li style="margin-bottom: 10px;">
                        <a href="https://github.com/2zalab/plumsmboa" target="_blank" style="
                            color: white;
                            text-decoration: none;
                            display: flex;
                            align-items: center;
                        ">
                            <span style="
                                display: inline-block;
                                width: 28px;
                                height: 28px;
                                background-color: rgba(255,255,255,0.2);
                                border-radius: 50%;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                margin-right: 10px;
                                font-size: 0.9rem;
                            ">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                                  <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                                </svg>
                            </span>
                            GitHub du projet
                        </a>
                    </li>
                    <li style="margin-bottom: 10px;">
                        <a href="https://www.kaggle.com/datasets/arnaudfadja/african-plums-quality-and-defect-assessment-data" target="_blank" style="
                            color: white;
                            text-decoration: none;
                            display: flex;
                            align-items: center;
                        ">
                            <span style="
                                display: inline-block;
                                width: 28px;
                                height: 28px;
                                background-color: rgba(255,255,255,0.2);
                                border-radius: 50%;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                margin-right: 10px;
                                font-size: 0.9rem;
                            ">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                                  <path d="M12 0c-1.467 0-2.882.307-4.204.907a11.437 11.437 0 0 0-1.161.56c-.47.028-.48.057-.96.086-3.092 1.87-5.337 5.078-5.539 8.842v.12l.003.073V23.86L6.244 18.2l.092-.034c.813-.307 1.74-.59 2.73-.84l.118-.028c3.06-.765 6.475-.765 9.536 0l.118.028c.992.25 1.918.533 2.73.84l.092.034 5.24 5.66V10.587l-.003-.073v-.12c-.202-3.764-2.447-6.972-5.539-8.842-.048-.03-.049-.058-.096-.086a11.437 11.437 0 0 0-1.162-.56A11.976 11.976 0 0 0 12 0Z"/>
                                </svg>
                            </span>
                            Dataset Kaggle
                        </a>
                    </li>
                    <li>
                        <a href="https://www.jcia-cameroun.com" target="_blank" style="
                            color: white;
                            text-decoration: none;
                            display: flex;
                            align-items: center;
                        ">
                            <span style="
                                display: inline-block;
                                width: 28px;
                                height: 28px;
                                background-color: rgba(255,255,255,0.2);
                                border-radius: 50%;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                margin-right: 10px;
                                font-size: 0.9rem;
                            ">🌐</span>
                            JCIA 2025
                        </a>
                    </li>
                </ul>
            </div>      
            <!-- Contact -->
            <div>
                <h3 style="margin-top: 0; font-weight: 600; display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 10px;">📞</span> Contact
                </h3>
                <p style="margin-top: 0;">Pour toute question ou suggestion concernant notre projet:</p>
                <div style="
                    display: flex;
                    align-items: center;
                    margin-bottom: 12px;
                ">
                    <span style="
                        display: inline-block;
                        width: 28px;
                        height: 28px;
                        background-color: rgba(255,255,255,0.2);
                        border-radius: 50%;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        margin-right: 10px;
                        font-size: 0.9rem;
                    ">📧</span>
                    <a href="mailto:contact@maroua-it.com" style="color: white; text-decoration: none;">
                        contact@maroua-it.com
                    </a>
                </div>
                <div style="
                    display: flex;
                    align-items: center;
                ">
                    <span style="
                        display: inline-block;
                        width: 28px;
                        height: 28px;
                        background-color: rgba(255,255,255,0.2);
                        border-radius: 50%;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        margin-right: 10px;
                        font-size: 0.9rem;
                    ">📱</span>
                    <a href="tel:+237672277579" style="color: white; text-decoration: none;">
                        +237 672 277 579
                    </a>
                </div>
            </div>          
            <!-- Licence -->
            <div>
                <h3 style="margin-top: 0; font-weight: 600; display: flex; align-items: center;">
                    <span style="font-size: 1.2rem; margin-right: 10px;">📄</span> Licence
                </h3>
                <p style="margin-top: 0;">Ce projet est sous licence MIT.</p>
                <p style="margin-bottom: 0;">© 2025 MIT - Tous droits réservés</p>
                <div style="margin-top: 15px;">
                    <a href="#" style="
                        display: inline-block;
                        padding: 5px 15px;
                        background-color: rgba(255,255,255,0.2);
                        color: white;
                        text-decoration: none;
                        border-radius: 20px;
                        font-size: 0.9rem;
                        transition: background-color 0.2s;
                    " onmouseover="this.style.backgroundColor='rgba(255,255,255,0.3)'" 
                       onmouseout="this.style.backgroundColor='rgba(255,255,255,0.2)'">
                        Voir la licence
                    </a>
                </div>
            </div>
        </div>
    </div>   
    <!-- Version et copyright -->
    <div style="text-align: center; margin-top: 20px; color: #666; font-size: 0.9rem;">
        <p>PlumsMboa v1.0.0 | Développé avec ❤️ au Cameroun</p>
    </div>
    """, unsafe_allow_html=True)