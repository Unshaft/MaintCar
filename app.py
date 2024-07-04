import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import joblib
import os

# Charger les données
ot_odr_filename = os.path.join(".", "data/OT_ODR.csv.bz2")
ot_odr_df = pd.read_csv(ot_odr_filename, compression="bz2", sep=";")
equipements_filename = os.path.join(".", 'data/EQUIPEMENTS.csv')
equipements_df = pd.read_csv(equipements_filename, sep=";")

# Fusionner les deux jeux de données sur l'identifiant du véhicule (EQU_ID)
merged_df = pd.merge(ot_odr_df, equipements_df, on="EQU_ID")
print("OK")

# Charger le modèle entraîné et les encodeurs
model_ODR = joblib.load('models/maintenance_model_ODR.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
model_columns = joblib.load('models/model_columns.pkl')

# Créer l'application Dash
app = dash.Dash(__name__)
app.title = "Diagnostic de Maintenance"

# Inclure le fichier CSS
app.css.append_css({"external_url": "/assets/styles.css"})

# Layout de l'application
app.layout = html.Div([
    html.Nav([
        html.A("Diagnostic de Maintenance des Véhicules", className="navbar-brand"),
    ], className="navbar navbar-dark bg-dark"),

    html.Div([
        html.Div([
            html.Label("Constructeur :", className="form-label"),
            dcc.Dropdown(
                id='constructeur-dropdown',
                options=[{'label': i, 'value': i} for i in merged_df['CONSTRUCTEUR'].unique()],
                className="form-control"
            ),
        ], className="form-group"),

        html.Div([
            html.Label("Modèle :", className="form-label"),
            dcc.Dropdown(
                id='modele-dropdown',
                options=[{'label': i, 'value': i} for i in merged_df['MODELE'].unique()],
                className="form-control"
            ),
        ], className="form-group"),

        html.Div([
            html.Label("Moteur :", className="form-label"),
            dcc.Dropdown(
                id='moteur-dropdown',
                options=[{'label': i, 'value': i} for i in merged_df['MOTEUR'].unique()],
                className="form-control"
            ),
        ], className="form-group"),

        html.Div([
            html.Label("Organe :", className="form-label"),
            dcc.Dropdown(
                id='organe-dropdown',
                options=[{'label': i, 'value': i} for i in merged_df['SIG_ORGANE'].unique()],
                className="form-control"
            ),
        ], className="form-group"),

        html.Div([
            html.Label("Symptôme :", className="form-label"),
            dcc.Dropdown(
                id='symptome-dropdown',
                options=[],
                className="form-control"
            ),
        ], className="form-group"),

        html.Div([
            html.Label("Contexte :", className="form-label"),
            dcc.Dropdown(
                id='contexte-dropdown',
                options=[],
                className="form-control"
            ),
        ], className="form-group"),

        html.Button('Obtenir des prédictions', id='predict-button', n_clicks=0, className="btn btn-primary"),
    ], className="container mt-4 sidebar-custom"),

    html.Div(id='prediction-container', className="container mt-4"),

    html.Div(id='error-message', className="container mt-4", style={'color': 'red'})
])

# Callback pour mettre à jour les options des menus déroulants pour les problèmes signalés
@app.callback(
    [Output('symptome-dropdown', 'options'),
     Output('contexte-dropdown', 'options')],
    [Input('organe-dropdown', 'value'),
     Input('symptome-dropdown', 'value'),
     Input('contexte-dropdown', 'value')]
)
def set_problem_dropdown_options(organe, symptome, contexte):
    filtered_df = merged_df
    if organe:
        filtered_df = filtered_df[filtered_df['SIG_ORGANE'] == organe]
    if symptome:
        filtered_df = filtered_df[filtered_df['SIG_OBS'] == symptome]
    if contexte:
        filtered_df = filtered_df[filtered_df['SIG_CONTEXTE'] == contexte]

    symptome_options = [{'label': i, 'value': i} for i in filtered_df['SIG_OBS'].unique()]
    contexte_options = [{'label': i, 'value': i} for i in filtered_df['SIG_CONTEXTE'].unique()]

    return symptome_options, contexte_options

# Callback pour générer les prédictions en fonction des entrées
@app.callback(
    [Output('prediction-container', 'children'),
     Output('error-message', 'children')],
    [Input('predict-button', 'n_clicks')],
    [Input('organe-dropdown', 'value'),
     Input('symptome-dropdown', 'value'),
     Input('contexte-dropdown', 'value'),
     Input('modele-dropdown', 'value'),
     Input('constructeur-dropdown', 'value'),
     Input('moteur-dropdown', 'value')]
)
def update_predictions(n_clicks, organe, symptome, contexte, modele, constructeur, moteur):
    try:
        if n_clicks > 0:
            # Vérifier que toutes les entrées sont sélectionnées
            if not all([organe, symptome, contexte, modele, constructeur, moteur]):
                return "", "Veuillez sélectionner toutes les options avant de lancer une prédiction."

            # Préparer les données d'entrée pour le modèle
            input_data = pd.DataFrame({
                'MODELE': [modele],
                'CONSTRUCTEUR': [constructeur],
                'MOTEUR': [moteur],
                'SIG_ORGANE': [organe],
                'SIG_OBS': [symptome],
                'SIG_CONTEXTE': [contexte]
            })

            # Encoder les données catégorielles
            for col, le in label_encoders.items():
                input_data[col] = le.transform(input_data[col])

            # Ajouter les colonnes manquantes avec des zéros
            for col in model_columns:
                if col not in input_data.columns:
                    input_data[col] = 0

            # Réorganiser les colonnes pour correspondre à l'entraînement
            input_data = input_data[model_columns]

            # Faire des prédictions
            pred_odr = model_ODR.predict(input_data)

            # Convertir les prédictions en libellés
            pred_odr_label = label_encoder.inverse_transform(pred_odr)[0]

            return (
                html.Div([
                    html.H3("Prédictions :"),
                    html.P(f"ODR_LIBELLE : {pred_odr_label}")
                ]),
                ""
            )
        return "", ""
    except Exception as e:
        return "", f"Erreur lors de la prédiction : {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True)
