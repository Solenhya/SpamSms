from re import M
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class BaseCountModel():
    
    def __init__(self, dataset, random=None):
        """Initie la base du modèle

        Args:
            dataset (DataFrame): DataFrame d'entrainement du modèle
            random (Booléean, optional): Si True, split random établit à 42. Defaults to None.
        """
        self.dataset = dataset
        self.dataset['spam'] = self.dataset['spam'].replace({'ham': 0, 'spam': 1})
        self.vectorizer = CountVectorizer(stop_words='english')
        self.X = self.vectorizer.fit_transform(self.dataset['text'])
        self.y = self.dataset['spam']
        self.random_state = 42 if random else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
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
    
class BaseTFIDModel():
    
    def __init__(self, dataset, random=None):
        """Initie la base du modèle

        Args:
            dataset (DataFrame): DataFrame d'entrainement du modèle
            random (Booléean, optional): Si True, split random établit à 42. Defaults to None.
        """
        self.dataset = dataset
        self.dataset['spam'] = self.dataset['spam'].replace({'ham': 0, 'spam': 1})
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.X = self.vectorizer.fit_transform(self.dataset['text'])
        self.y = self.dataset['spam']
        self.random_state = 42 if random else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
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

class NaiveBayesModel(BaseCountModel):
    
    def __init__(self, dataset, random=None):
        """Créer un modèle de classification

        Args:
            dataset (Dataframe): Dataframe d'entrainement du modèle
        """
        super().__init__(dataset, random)
        self.model = MultinomialNB()
        self.model.fit(self.X_train, self.y_train)
        
class SCVModel(BaseCountModel):
    def __init__(self, dataset, random=None):
        """Créer un modèle de classification
        
        Args:
            dataset (Dataframe): Dataframe d'entrainement du modèle
        """
        super().__init__(dataset, random)
        self.model = LinearSVC()
        self.model.fit(X=self.X_train, y=self.y_train)

class RandomForestModel(BaseTFIDModel):
    def __init__(self, dataset, random=None):
        """Créer un modèle de classification
        Args:
                dataset (Dataframe): Dataframe d'entrainement du modèle
        """
        super().__init__(dataset, random)
        self.model = RandomForestClassifier()
        self.model.fit(X=self.X_train, y=self.y_train)


class LogisticRegressionModel(BaseCountModel):
    def __init__(self, dataset, random=None):
        """Créer un modèle de classification
        Args:
                dataset (Dataframe): Dataframe d'entrainement du modèle
        """
        super().__init__(dataset, random)
        self.model = LogisticRegression()
        self.model.fit(X=self.X_train, y=self.y_train)
