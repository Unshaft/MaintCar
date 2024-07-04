import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib

# Créer le répertoire de sauvegarde des modèles s'il n'existe pas
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Charger les données
ot_odr_filename = os.path.join(".", "data/OT_ODR.csv.bz2")
ot_odr_df = pd.read_csv(ot_odr_filename, compression="bz2", sep=";")
equipements_filename = os.path.join(".", 'data/EQUIPEMENTS.csv')
equipements_df = pd.read_csv(equipements_filename, sep=";")

# Fusionner les deux jeux de données sur l'identifiant du véhicule (EQU_ID)
merge = pd.merge(ot_odr_df, equipements_df, on='EQU_ID', how='left')

# Sélectionnez la colonne numérique 'KILOMETRAGE'
Data_num = merge[['KILOMETRAGE']]
imputer = SimpleImputer(strategy='mean')
Data_num = pd.DataFrame(imputer.fit_transform(merge[['KILOMETRAGE']]))
Data_num.rename(columns={0: 'KILOMETRAGE'}, inplace=True)

# Creation de la base
Data_model = merge.drop(columns=['KILOMETRAGE'])
Data_model = pd.concat([Data_model, Data_num], axis=1)

# Convertir les libellés d'ordres de travail en valeurs numériques avec LabelEncoder
y = Data_model['ODR_LIBELLE'].astype('category')
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Créer les colonnes pour chaque mot-clé de SIG_CONTEXTE
all_keywords = pd.concat([Data_model['SIG_CONTEXTE'].str.split('/')]).explode().unique()
keywords = [kw for kw in all_keywords if pd.notna(kw)]

for keyword in keywords:
    Data_model[keyword] = 0

def update_binary_columns(row, keywords):
    for keyword in keywords:
        if isinstance(row['SIG_CONTEXTE'], str) and keyword in row['SIG_CONTEXTE'].split('/'):
            row[keyword] = 1
    return row

Data_model = Data_model.apply(update_binary_columns, axis=1, keywords=keywords)
Data_model = Data_model.fillna(0)

col_supp = ['CONSTRUCTEUR', 'SIG_CONTEXTE', 'DATE_OT', 'DUREE_TRAVAIL', 'EQU_ID', 'MOTEUR', 'ODR_ID', 'ODR_LIBELLE', 'OT_ID', 'SYSTEM_N1', 'SYSTEM_N2', 'SYSTEM_N3', 'TYPE_TRAVAIL']
Data_model.drop(columns=col_supp, inplace=True)

categorical_columns = ['MODELE', 'LIGNE', 'SIG_OBS', 'SIG_ORGANE']
Data_cat = Data_model[categorical_columns]
Data_model.drop(columns=categorical_columns, inplace=True)

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    Data_cat[col] = le.fit_transform(Data_cat[col])
    label_encoders[col] = le

index_columns = Data_model.columns[15:49]
selected_columns = list(index_columns)
X = pd.concat([Data_model[selected_columns], Data_cat], axis=1)

# Entraîner le modèle avec une barre de progression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model_ODR = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=20, min_samples_leaf=40)

print("Training the model...")
model_ODR.fit(X_train, y_train)

# Sauvegarder le modèle, les encoders et les colonnes dans le répertoire de sauvegarde
joblib.dump(model_ODR, os.path.join(model_dir, 'maintenance_model_ODR.pkl'))
joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))
joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
joblib.dump(X.columns, os.path.join(model_dir, 'model_columns.pkl'))

# Évaluer le modèle
pred_odr = model_ODR.predict(X_test)
accuracy_ODR = accuracy_score(y_test, pred_odr)
print(f'Accuracy: {accuracy_ODR:.4f}')
