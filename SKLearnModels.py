from re import M
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

class NaiveBayesModel():
    
    def __init__(self, dataset, random=None):
        """Créer un modèle de classification

        Args:
            dataset (Dataframe): Dataframe d'entrainement du modèle
        """
        self.dataset = dataset
        self.dataset['spam'] = self.dataset['spam'].replace({'ham': 0, 'spam': 1})
        self.vectorizer = CountVectorizer(stop_words='english')
        self.X = self.vectorizer.fit_transform(self.dataset['text'])
        self.y = self.dataset['spam']
        self.random_state = 42 if random else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = MultinomialNB()
        self.model.fit(self.X_train, self.y_train)
 
    def Predict(self, test_data=None):
        """Applique le modèle sur des données

        Args:
            test_data (Serie, optional): Serie de messages à classifier. Defaults to None.

        Returns:
            list: Prédictions sous forme de labels ('ham' ou 'spam').
        """
        if test_data is None:
            test_data = self.X_test
        else:
            test_data = self.vectorizer.transform(test_data)
        predictions = self.model.predict(test_data)
        return predictions


