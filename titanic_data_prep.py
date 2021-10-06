##########################################################################################################
# FEATURE ENGINEERING
##########################################################################################################
# Business Problem: A data preprocessing and feature engineering script for a machine learning pipeline
# needs to be prepared.

# It is expected that the dataset will be ready for modelling when passed through this script.
##########################################################################################################
# STORY OF THE DATASET:

# The dataset is the dataset of the people who were in the Titanic shipwreck.
# It consists of 768 observations and 12 variables.
# The target variable is specified as "Survived";
# 0: indicates the person's inability to survive.
# 1: refers to the survival of the person.
##########################################################################################################
# ATTRIBUTES:
# PassengerId: ID of the passenger
# Survived: Survival status (0: not survived, 1: survived)
# Pclass: Ticket class (1: 1st class (upper), 2: 2nd class (middle), 3: 3rd class(lower))
# Name: Name of the passenger
# Sex: Gender of the passenger (male, female)
# Age: Age in years
# Sibsp: Number of siblings/spouses aboard the Titanic
    # Sibling = Brother, sister, stepbrother, stepsister
    # Spouse = Husband, wife (mistresses and fiances were ignored)
# Parch: Number of parents/children aboard the Titanic
    # Parent = Mother, father
    # Child = Daughter, son, stepdaughter, stepson
    # Some children travelled only with a nanny , therefore Parch = 0 for them.
# Ticket: Ticket number
# Fare: Passenger fare
# Cabin: Cabin number
# Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

##########################################################################################################
# Importing necessary libraries and modules
# Making necessary adjustments for the representation of the dataset
##########################################################################################################
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler
from helpers.data_prep import outlier_thresholds, check_outlier, grab_outliers, remove_outlier, replace_with_thresholds, \
    missing_values_table, missing_vs_target, label_encoder, one_hot_encoder, rare_analyser, rare_encoder
from helpers.eda import grab_col_names, cat_summary, num_summary

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

dff = load()

def titanic_data_prep(df):
    df.columns = [col.upper() for col in df.columns]
    # 1. FEATURE ENGINEERING
    df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
    df["NEW_NAME_COUNT"] = df["NAME"].str.len()
    df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
    df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
    df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
    df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]
    # 2. OUTLIERS
    for col in num_cols:
        print(col, check_outlier(df, col))
    for col in num_cols:
        replace_with_thresholds(df, col)
    for col in num_cols:
        print(col, check_outlier(df, col))
    # 3. MISSING VALUES
    missing_values_table(df)
    remove_cols = ["CABIN","TICKET", "NAME"]
    df.drop(remove_cols, inplace=True, axis=1)
    df.head()
    missing_values_table(df)
    df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
    df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
    # 4. LABEL ENCODING
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
    for col in binary_cols:
        df = label_encoder(df, col)
    # 5. RARE ENCODING
    rare_analyser(df, "SURVIVED", cat_cols)
    df = rare_encoder(df, 0.01, cat_cols)
    # 6. ONE HOT ENCODING
    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
    df = one_hot_encoder(df, ohe_cols)
    df.shape
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]
    rare_analyser(df, "SURVIVED", cat_cols)
    useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]   # ['NEW_NAME_WORD_COUNT_Rare']
    df.drop(useless_cols, axis=1, inplace=True)
    # 7. ROBUST SCALER
    rs = RobustScaler()
    df[num_cols] = rs.fit_transform(df[num_cols])
    return df

dff_prepared = titanic_data_prep(dff)

check_df(dff_prepared, quan=True)
dff_prepared.shape

##########################################################################################################
# Save the preprocessed data set to disk with pickle.
##########################################################################################################
dff_prepared.to_pickle("titanic.pkl")
pd.read_pickle("titanic.pkl")
##########################################################################################################