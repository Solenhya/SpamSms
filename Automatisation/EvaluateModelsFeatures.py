from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import re
import pandas as pd
from sklearn.svm import LinearSVC
from itertools import combinations
import time
from sklearn.base import clone


class DataExtraction(BaseEstimator, TransformerMixin):
    
    def __init__(self, features_list):
        self.features_list = features_list
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        retour = X
        for feature_name, feature_function in self.features_list.items():
            retour[feature_name] = X["text"].apply(feature_function)
        retour = retour.drop(columns = "text")
        return retour
    
def get_dictionnaire(string_list):
    """Get a dictionary of text feature extraction functions.

    Args:
        string_list (str or list): Either "all" to get all features, "combination" to get all possible combinations
                                  of features, or a list of specific feature names to select.

    Returns:
        dict: A dictionary containing the requested feature extraction functions. Keys are feature names and
              values are lambda functions that extract the corresponding features from text.
    """
    features = {
        "taille_phrase":lambda x: len(x),
        "Nombre_mot":lambda x: len(x.split()),
        "email": lambda x: len(re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', x)) > 0,
        "presence_monnaie": lambda x: 1 if re.search(r'[\$\€\£]', x) else 0 ,
        "presence telephone": lambda x: 1 if re.search(r'\b\d{10,}\b', x) else 0,
        "presence_caratere_speciaux": lambda x: 1 if re.search(r'[!@#$%^&*(),.?":{}|<>]', x) else 0,
        "proportion_majuscule": lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0,
        "presence_lien": lambda x: 1 if re.search(r'\b(http|www)\S+', x) else 0
    }
    if string_list == "all":
        return features
    elif string_list == "combination":
        # Générer toutes les combinaisons possibles
        all_combinations = []
        for r in range(1, len(features) + 1):
            combinations_r = list(combinations(features.keys(), r))
            all_combinations.extend(combinations_r)

        # Créer un dictionnaire de toutes les combinaisons
        combinations_dict = {}
        for i, combo in enumerate(all_combinations, 1):
            combinations_dict[f"combination_{i}"] = {k: features[k] for k in combo}
        return combinations_dict
    else:
        dict = {}
        try:
            for key in string_list:
                try:
                    dict[key] = features[key]
                except:
                    print(f"La fonction {key} n'existe pas dans le dictionnaire")
        except:
            print(f"La fonction demande une liste de string")
        return dict
    
def GenerateModel(features_names, model, X_train, y_train,vectorizer=TfidfVectorizer(stop_words="english"), scaler=StandardScaler()) :
    """
    Generate and train a machine learning pipeline for text classification.

    Args:
        features_names (dict): Dictionary of feature extraction functions
        model: The machine learning model to use
        X_train (pd.DataFrame): Training data containing text samples
        y_train (pd.Series): Training labels
        vectorizer: Text vectorizer (default: TfidfVectorizer with English stop words)
        scaler: Feature scaler (default: StandardScaler)

    Returns:
        Pipeline: Trained pipeline that combines feature extraction, text vectorization, and the model
        
    Usage exemple:
        >>> import EvaluateModelsFeatures as Eval
        >>> model = Eval.GenerateModel(model=RandomForestClassifier(),X_train=X1_train, y_train=y1_train, features_names=Eval.get_dictionnaire("all"))
        >>> print(classification_report(y1_test,model.predict(X1_test)))
        >>> print(model.predict(pd.DataFrame(['You won 200 billion dollars, call now!', 'Hi, how are you?'], columns=["text"])))

    """
    target = y_train
    data = X_train
    featurePipe = Pipeline(steps=[("extraction feature",DataExtraction(features_names)),("inputing",SimpleImputer(strategy="mean")),("scaling",scaler)])
    preparation = ColumnTransformer(transformers=
                                [("features",featurePipe,["text"]),
                                ("vectorisation",vectorizer,"text")]
                                )

    modelPipe = Pipeline(steps=[("prep données",preparation),("model",model)])
    modelPipe.fit(data,target)
    return modelPipe

def extract_metrics(row,generalName = ""):
    """
        Extract specific metrics from a classification report dictionary.
        
        Args:
            row (dict): A row containing classification metrics for different classes
            
        Returns:
            pd.Series: A pandas Series containing extracted metrics for 'ham' and 'spam' classes
        """
    
    metrics = {}
    for class_name, values in row.items():
        if isinstance(values, float):
            continue
        for metric, value in values.items():
            if class_name not in ['ham', 'spam']:
                continue
            if generalName == "":
                metrics[f'{metric}_{class_name}'] = value
            else:
                metrics[f'{metric}_{class_name} for {generalName}'] = value

    return pd.Series(metrics)


def calculate_precisions_for_all_combinations(X_train, y_train, X_test, y_test, model=LinearSVC(), scaler=StandardScaler()):
    """
        Calculate precision metrics for all feature combinations using a specified model.
    
        Args:
            X_train (pd.DataFrame): Training data containing text samples
            y_train (pd.Series): Training labels
            X_test (pd.DataFrame): Test data containing text samples
            y_test (pd.Series): Test labels
            model: The machine learning model to use (default: LinearSVC)
            scaler: Feature scaler (default: StandardScaler)
    
        Returns:
            pd.DataFrame: DataFrame containing combination details, accuracy metrics, training time,
                         and recall scores for both ham and spam classes
                         
        Usage exemple:
            >>> from pandas import DataFrame 
            >>> from sklearn.naive_bayes import MultinomialNB 
            >>> model = MultinomialNB() 
            >>> df_naive_bayes: DataFrame = Eval.calculate_precisions_for_all_combinations(X1_train,  y1_train, X1_test, y1_test, model=model, scaler=MinMaxScaler()) 
        """
    
    results_df = pd.DataFrame(columns=['combination', 'accuracy', 'time'])
    combinations_dict = get_dictionnaire("combination")
    for combination, dict in combinations_dict.items():
        start_time = time.time()
        # Create a fresh instance of the model for each iteration
        model_instance = clone(model)
        pipeline = GenerateModel(model=model_instance, X_train=X_train, y_train=y_train, features_names=dict, scaler=scaler)
        end_time = time.time()
        training_time = end_time - start_time
        accuracy = classification_report(y_test, pipeline.predict(X_test), output_dict=True)
        results_df = pd.concat([
            results_df, 
            pd.DataFrame({
                'combination': [list(dict.keys())], 
                'accuracy': [accuracy],
                'time': [training_time]
            })
        ])
    results_df.reset_index(drop=True, inplace=True)
    new_columns = results_df["accuracy"].apply(extract_metrics)
    df_precisions = pd.concat([results_df, new_columns], axis=1)
    return df_precisions[["combination", "accuracy", "time", "recall_ham", "recall_spam"]]

def calculatePrecisionOnBattery(X_train, y_train, testBattery, model=LinearSVC(), scaler=StandardScaler()):
    combinations_dict = get_dictionnaire("combination")
    lineToConct = []
    for combination, dict in combinations_dict.items():
        start_time = time.time()
        # Create a fresh instance of the model for each iteration
        model_instance = clone(model)
        pipeline = GenerateModel(model=model_instance, X_train=X_train, y_train=y_train, features_names=dict, scaler=scaler)
        end_time = time.time()
        training_time = end_time - start_time
        line = pd.DataFrame(columns=["combination"])
        line["combination"]=[list(dict.keys())]
        line["time"]=training_time
        for key , value in testBattery.items():
            X_test,y_test=value
            repport = classification_report(y_test,pipeline.predict(X_test),output_dict=True)
            line["dictionnary"] = [repport]
            new_coll = line["dictionnary"].apply(extract_metrics,generalName=f"{key}")
            line = pd.concat([line, new_coll], axis=1)
        lineToConct.append(line)
    results_df = pd.concat(lineToConct,ignore_index=True)

    return results_df

def calculatePrecisionOnBattery(X_train, y_train, testBattery, model=LinearSVC(), scaler=StandardScaler()):
    combinations_dict = get_dictionnaire("combination")
    lineToConct = []
    for combination, dict in combinations_dict.items():
        start_time = time.time()
        # Create a fresh instance of the model for each iteration
        model_instance = clone(model)
        pipeline = GenerateModel(model=model_instance, X_train=X_train, y_train=y_train, features_names=dict, scaler=scaler)
        end_time = time.time()
        training_time = end_time - start_time
        line = pd.DataFrame(columns=["combination"])
        line["combination"]=[list(dict.keys())]
        line["time"]=training_time
        for key , value in testBattery.items():
            X_test,y_test=value
            repport = classification_report(y_test,pipeline.predict(X_test),output_dict=True)
            line["dictionnary"] = [repport]
            new_coll = line["dictionnary"].apply(extract_metrics,generalName=f"{key}")
            line = pd.concat([line, new_coll], axis=1)
        lineToConct.append(line)
    results_df = pd.concat(lineToConct,ignore_index=True)

    return results_df