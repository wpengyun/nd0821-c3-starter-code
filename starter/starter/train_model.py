# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
import pandas as pd
from ml.model import train_model, compute_model_metrics, inference
# Add the necessary imports for the starter code.

# Add code to load in the data.
data=pd.read_csv("/Users/Pengyun_Wang/gits/nd0821-c3-starter-code/starter/data/census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(test,label="salary", categorical_features=cat_features, encoder=encoder, lb=lb,training=False)
# Train and save a model.
model=train_model(X_train,y_train)

y_pred=inference(model, X_test)
prec, recall, fscore = compute_model_metrics(y_test, y_pred)

print(f'prec: {prec}, recall: {recall}, fscore:{fscore}')
