# Diagnostic de Maintenance des Véhicules

Cette application Dash aide à prédire les actions de maintenance les plus appropriées pour une flotte de véhicules en fonction de diverses caractéristiques du véhicule et des symptômes signalés.

## Installation

### Prérequis

- Python 3.x
- Pip (gestionnaire de paquets Python)

### Étapes

1. Clonez ce dépôt de code sur votre machine locale :

    ```sh
    git clone https://github.com/votre-nom-utilisateur/diagnostic-maintenance-vehicules.git
    cd diagnostic-maintenance-vehicules
    ```

2. Créez et activez un environnement virtuel (optionnel mais recommandé) :

    ```sh
    python -m venv venv
    source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`
    ```

3. Installez les dépendances requises :

    ```sh
    pip install -r requirements.txt
    ```

## Utilisation

1. Assurez-vous que les fichiers de données `OT_ODR.csv.bz2` et `EQUIPEMENTS.csv` sont présents dans le répertoire `data`.

2. Entraînez le modèle et préparez les fichiers nécessaires :

    ```sh
    python train_model.py
    ```

3. Lancez l'application Dash :

    ```sh
    python app.py
    ```

4. Ouvrez votre navigateur web et allez à l'adresse suivante :

    ```
    http://127.0.0.1:8050/
    ```

## Structure du projet

diagnostic-maintenance-vehicules/
├── app.py # Application principale Dash
├── train_model.py # Script d'entraînement du modèle
├── data/
│ ├── OT_ODR.csv.bz2 # Données des ordres de travail et ordres de réparation
│ ├── EQUIPEMENTS.csv # Données des équipements des véhicules
├── models/
│ ├── maintenance_model_ODR.pkl # Modèle entraîné
│ ├── label_encoder.pkl # Encodeur de labels
├── assets/
│ ├── styles.css # Fichier de styles CSS
├── requirements.txt # Dépendances Python
├── README.md # Documentation du projet


## Personnalisation du style

Le fichier `assets/styles.css` contient les styles personnalisés utilisés dans l'application. Vous pouvez le modifier pour adapter l'apparence de l'application à vos préférences.

## Contribuer

Les contributions sont les bienvenues ! Si vous avez des suggestions pour améliorer l'application, n'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Auteurs

- Evan BRISEBOIS (brisebois.e2100209@etud.univ-ubs.fr)
- Killian PERZO (perzo.e2202845@etud.univ-ubs.fr)
- Matthieu VIVIER (viver.e2101336@etud.univ-ubs.fr)


