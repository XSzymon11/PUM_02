#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Read recipe inputs
titanic = dataiku.Dataset("titanic")
df = titanic.get_dataframe()

# Eksploracja danych
# print('Nazwy kolumn: ',df.columns)
# print('Liczba kolumn i wierszy: ',df.shape)
# print('Typy danych: ',df.dtypes)

# Przygotowanie danych
df_copy = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_copy['Age'].fillna(df_copy['Age'].median(), inplace=True)
df_copy['Sex'] = df_copy['Sex'].map({'male': 0, 'female': 1})
df_copy['Embarked'].fillna('S', inplace=True)
df_copy['Embarked'] = df_copy['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
# print(df_copy.isnull().sum()) # nie ma więcj braków

# Podział na zbiory
X = df_copy.drop('Survived', axis=1)
y = df_copy['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Trening modelu
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Ewaluacja
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion_mx = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Matrix: \n', confusion_mx)

results = X_test.copy()
results["y_true"] = y_test.values
results["y_pred"] = y_pred

# Write recipe outputs
model_summary = dataiku.Dataset('titanic_out')
model_summary.write_with_schema(results)

