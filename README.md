# SpamSms
Un projet d'apprentissage developpeur intelligence artificielle sur la détection de SMS spam  
L'objectif est d'entrainer un modèle pour créer une application qui permet de détecter si un message est un spam ou non.

# Contenu
- Un notebook d'une implementation de classification bayesienne naïve  
- Un programme python qui permet de serialiser un model bayesien  
- un notebook qui deserialise un model pour pouvoir le tester  
- des modules pour automatiser la création de modèles et les tests sur des combinaisons de dataset / paramètres / modèles / vectorizer / features
- 2 applications streamlit pour tester si un message est un spam ou non:
  - 1 avec le module naive bayes recréé manuellement
  - 1 avec le module LinearSVC de scikit-learn entrainé sur les 3 jeux de données avec une vectotirisation de TFFIDF et 3 features : présence de symboles monétaires, présence de liens et présence de caractères spéciaux

# Installation et dépendance
- modules nécessaire : pandas , joblib , argsparse, plotly, scikit-learn, matplotlib
- Base de donnée inclus  

# Auteurs
Le groupe est composé de :  
Melody Duplaix  
Fabien Herry  