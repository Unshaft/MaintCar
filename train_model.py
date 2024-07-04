import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
ot_odr_filename = os.path.join(".", "data/OT_ODR.csv.bz2")
ot_odr_df = pd.read_csv(ot_odr_filename,
                        compression="bz2",
                        sep=";")

equipements_filename = os.path.join(".", 'data/EQUIPEMENTS.csv')
equipements_df = pd.read_csv(equipements_filename,
                             sep=";")

# Merge des deux bases de données, Fusionner les données sur 'equipment_id'
merge = pd.merge(ot_odr_df, equipements_df, on='EQU_ID', how='left')

var_sig = ["SIG_ORGANE", "SIG_CONTEXTE", "SIG_OBS"]
#merge[var_sig].describe()


var_sys = ["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3"]
#merge[var_sys].describe()


var_odr = ["TYPE_TRAVAIL", "ODR_LIBELLE"]
#merge[var_odr].describe()


var_cat = ['ODR_LIBELLE', 'TYPE_TRAVAIL',
           'SYSTEM_N1', 'SYSTEM_N2', 'SYSTEM_N3', 
           'SIG_ORGANE', 'SIG_CONTEXTE', 'SIG_OBS', 'LIGNE']

for var in var_cat:
    merge[var] = merge[var].astype('category')

#merge.info()

Data_appli = merge

print("Process OK 1")

# Sélectionnez la colonne numérique 'KILOMETRAGE'
Data_num = merge[['KILOMETRAGE']]
imputer = SimpleImputer(strategy='mean')  # Remplace les NaN par la moyenne
Data_num = pd.DataFrame(imputer.fit_transform(merge[['KILOMETRAGE']]))
Data_num.rename(columns={0: 'KILOMETRAGE'}, inplace=True)

print("Process OK 2")

#Creation de la base
Data_model = merge.drop(columns=['KILOMETRAGE'])
Data_model = pd.concat([Data_model, Data_num], axis=1)

# Convertir les libellés d'ordres de travail en valeurs numériques avec LabelEncoder
y = Data_model['ODR_LIBELLE'].astype('category')
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Sauvegarder le label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')


print("Process OK 3")

# Concatenate SIG_ORGANE and SIG_CONTEXTE values into one series
all_keywords = pd.concat([Data_model['SIG_CONTEXTE'].str.split('/')]).explode().unique()

print("Process OK 4")

# Remove NaN values and convert to a list of unique keywords
keywords = [kw for kw in all_keywords if pd.notna(kw)]

# Initialize columns for each keyword
for keyword in keywords:
    ot_odr_df[keyword] = 0


print("Process OK 5")

# Function to update the dataframe with binary values
def update_binary_columns(row, keywords):
    for keyword in keywords:
        if isinstance(row['SIG_CONTEXTE'], str) and keyword in row['SIG_CONTEXTE'].split('/'):
            row[keyword] = 1
    return row


print("Process OK 6")

# Apply the function to each row
data = Data_model.apply(update_binary_columns, axis=1, keywords=keywords)
Data_model = data.fillna(0)

col_supp = ['CONSTRUCTEUR', 'SIG_CONTEXTE', 'DATE_OT', 'DUREE_TRAVAIL', 'EQU_ID', 'MOTEUR', 'ODR_ID', 'ODR_LIBELLE', 'OT_ID', 'SYSTEM_N1', 'SYSTEM_N2', 'SYSTEM_N3', 'TYPE_TRAVAIL']
Data_model.drop(columns=col_supp, inplace=True)

# Sélectionner les colonnes spécifiques par nom
categorical_columns = ['MODELE', 'LIGNE', 'SIG_OBS', 'SIG_ORGANE']



print("Process OK 7")


# Combiner les sélections
selected_columns = categorical_columns
Data_cat = Data_model[selected_columns]
Data_model.drop(columns=categorical_columns, inplace=True)


print("Process OK 8")

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    Data_cat[col] = le.fit_transform(Data_cat[col])
    label_encoders[col] = le

# Sélectionner les colonnes de la 16ème à la dernière par index
index_columns = Data_model.columns[15:49]

# Combiner les sélections
selected_columns = list(index_columns)

X = pd.concat([Data_model[selected_columns], Data_cat], axis=1)

print("Process OK 9")

#######################################################################################

####### MODELE ARBRE DECISION SIMPLE

###################################################################################



# Diviser les données en ensembles d'entraînement et de test
X_train, X_test4, y_train, y_test4 = train_test_split(X, y, test_size=0.3, random_state=42)

# Construire le modèle d'arbre de décision
model_ODR = DecisionTreeClassifier(random_state=42, criterion= 'gini', max_depth= 20, min_samples_leaf= 40)

print("Process OK 10")




# Entraîner le modèle
model_ODR.fit(X_train, y_train)

# Faire des prédictions
pred_odr = model_ODR.predict(X_test4)

# Évaluer le modèle
accuracy_ODR = accuracy_score(y_test4, pred_odr)
print(f'Accuracy: {accuracy_ODR:.4f}')

# Sauvegarder le modèle
joblib.dump(model_ODR, 'maintenance_model_ODR.pkl')

print("Process OK 11")