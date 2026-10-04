import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Dossier où seront enregistrés les graphiques (affichés ensuite dans le README)
os.makedirs("images", exist_ok=True)


def sauvegarder(nom_fichier):
    """Enregistre la figure courante dans images/ puis l'affiche."""
    plt.tight_layout()
    plt.savefig(f"images/{nom_fichier}", dpi=150, bbox_inches="tight")
    plt.show()


# 
# 1. Chargement et exploration
# 
df = pd.read_csv("titanic.csv")
print(df.describe())
print("Valeurs manquantes :\n", df.isnull().sum())

# 
# 2. Valeurs manquantes
# 
# Embarked : remplacement par la valeur la plus fréquente (mode)
mode_embarked = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(mode_embarked)

# Age : imputation par KNN (k = 5)
features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Age"]
df_knn = df[features].copy()

# Encodage des variables catégorielles
df_knn["Sex"] = LabelEncoder().fit_transform(df_knn["Sex"])
df_knn["Embarked"] = LabelEncoder().fit_transform(df_knn["Embarked"])

# Normalisation (le KNN est basé sur des distances)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_knn)

# Imputation puis retour à l'échelle d'origine
data_imputed = KNNImputer(n_neighbors=5).fit_transform(data_scaled)
df_knn = pd.DataFrame(
    scaler.inverse_transform(data_imputed), columns=features, index=df.index
)

# Graphique : distribution de l'âge avant / après imputation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["Age"].dropna(), bins=20, ax=axes[0])
axes[0].set_title("Âge avant imputation")
sns.histplot(df_knn["Age"], bins=20, ax=axes[1], color="orange")
axes[1].set_title("Âge après imputation KNN")
sauvegarder("age_imputation.png")

# On ne récupère que l'âge imputé (Embarked garde ses lettres S, C, Q)
df["Age"] = df_knn["Age"]

# Cabin contient trop de valeurs manquantes : on la supprime
df = df.drop(columns=["Cabin"])
print("Après traitement des valeurs manquantes :")
print(df.isnull().sum())
print(df.describe())

# 3. Visualisation

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Survie selon le sexe
sns.countplot(data=df, x="Sex", hue="Survived", ax=axes[0])
axes[0].set_title("Survie selon le sexe")
axes[0].set_xlabel("Sexe")
axes[0].set_ylabel("Nombre de passagers")

# Âge des survivants
survived = df[df["Survived"] == 1]
sns.histplot(data=survived, x="Age", bins=20, ax=axes[1], color="green")
axes[1].set_title("Nombre de survivants par âge")
axes[1].set_xlabel("Âge")
axes[1].set_ylabel("Nombre de survivants")

sauvegarder("survie_sexe_age.png")

# 4. Valeurs aberrantes

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.boxplot(y=df["Age"], ax=axes[0])
axes[0].set_title("Boxplot de l'âge")

sns.boxplot(data=df, x="Pclass", y="Fare", ax=axes[1])
axes[1].set_title("Prix du billet selon la classe")
axes[1].set_xlabel("Classe")
axes[1].set_ylabel("Prix (Fare)")

sauvegarder("boxplots.png")