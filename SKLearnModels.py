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
import re


class BaseCountModel():
    
    def __init__(self, X, y, random=None):
        """Applique un modèle de classification sur des données

        Args:
            X (Série): Série d'entrainement de features
            y (Série): Série d'entrainement de label
            random (Booléan, optional): Si True randomise avec une seed de 42. Defaults to None.
        """
        self.y = y
        self.vectorizer = CountVectorizer(stop_words='english')
        self.X = self.vectorizer.fit_transform(X)
        self.random_state = 42 if random else None
        
    def Predict(self, test_data):
        """Applique le modèle sur des données

        Args:
            test_data (Serie, optional): Serie de messages à classifier. Defaults to None.

        Returns:
            list: Prédictions sous forme de labels ('ham' ou 'spam').
        """
        test_data = self.vectorizer.transform(test_data)
        predictions = self.model.predict(test_data)
        return predictions
    
class BaseTFIDModel():
    
    def __init__(self, X, y, random=None):
        """Applique un modèle de classification sur des données

        Args:
            X (Série): Série d'entrainement de features
            y (Série): Série d'entrainement de label
            random (Booléan, optional): Si True randomise avec une seed de 42. Defaults to None.
        """
        self.y = y
        self.vectorizer = CountVectorizer(stop_words='english')
        self.X = self.vectorizer.fit_transform(X)
        self.random_state = 42 if random else None
        
    def Predict(self, test_data):
        """Applique le modèle sur des données

        Args:
            test_data (Serie, optional): Serie de messages à classifier. Defaults to None.

        Returns:
            list: Prédictions sous forme de labels ('ham' ou 'spam').
        """
        test_data = self.vectorizer.transform(test_data)
        predictions = self.model.predict(test_data)
        return predictions

class BaseFeaturesModel():

    def __init__(self, X, y, random=None):
        """Applique un modèle de classification sur des données

        Args:
            X (Série): Série d'entrainement de features
            y (Série): Série d'entrainement de label
            random (Booléan, optional): Si True randomise avec une seed de 42. Defaults to None.
        """
        self.y = y
        self.X = self.Vectorize(X)
        self.X = self.X.drop(columns=['text'])
        self.random_state = 42 if random else None
        
    def Vectorize(self, df):
        """Transforme les données en vecteurs
        Args:
            text (Serie): Serie de messages à classifier
            
        Returns:
            list: Vecteurs de données
        """
        df = pd.DataFrame(df, columns=['text'])
        df['has_phone_number'] = df['text'].apply(lambda x: 1 if re.search(r'\b\d{10,}\b', x) else 0)
        df['has_currency_symbol'] = df['text'].apply(lambda x: 1 if re.search(r'[\$\€\£]', x) else 0)
        df["has_special_characters"] = df['text'].apply(lambda x: 1 if re.search(r'[!@#$%^&*(),.?":{}|<>]', x) else 0)
        df['message_length'] = df['text'].apply(len)
        df["number_count"] = df['text'].apply(lambda x: sum(c.isdigit() for c in x))
        df['word_count'] = df['text'].apply(lambda x: len(x.split()))
        df["uppercase_proportion"] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
        df["has_url"] = df['text'].apply(lambda x: 1 if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x) else 0)
        df["non_alpha_charaters_proportion"] = df['text'].apply(lambda x: sum(1 for c in x if not c.isalpha()) / len(x) if len(x) > 0 else 0)
        return df
        
    def Predict(self, test_data):
        """Applique le modèle sur des données

        Args:
            test_data (Serie, optional): Serie de messages à classifier. Defaults to None.

        Returns:
            list: Prédictions sous forme de labels ('ham' ou 'spam').
        """
        test_data = self.Vectorize(pd.DataFrame(test_data, columns=['text']))
        test_data = test_data.drop(columns=['text'])
        predictions = self.model.predict(test_data)
        return predictions



class NaiveBayesModel(BaseCountModel):
    
    def __init__(self, X, y, random=None):
        """Créer un modèle de classification

        Args:
            dataset (Dataframe): Dataframe d'entrainement du modèle
        """
        super().__init__(X, y, random)
        self.model = MultinomialNB()
        self.model.fit(self.X, self.y)
        
        
class NaiveBayesFeaturesModel(BaseFeaturesModel):
    def __init__(self, X, y, random=None):
        """Créer un modèle de classification
        
        Args:
            dataset (Dataframe): Dataframe d'entrainement du modèle
        """
        super().__init__(X, y, random)
        self.model = MultinomialNB()
        self.model.fit(self.X, self.y)
        
class SCVModel(BaseCountModel):
    def __init__(self, X, y, random=None):
        """Créer un modèle de classification
        
        Args:
            dataset (Dataframe): Dataframe d'entrainement du modèle
        """
        super().__init__(X, y, random)
        self.model = LinearSVC()
        self.model.fit(X=self.X, y=self.y)

class SCVFeaturesModel(BaseFeaturesModel):
    def __init__(self, X, y, random=None):
        """Créer un modèle de classification

        Args:
            dataset (Dataframe): Dataframe d'entrainement du modèle
        """
        super().__init__(X, y, random)
        self.model = LinearSVC()
        self.model.fit(X=self.X, y=self.y)

class RandomForestModel(BaseTFIDModel):
    def __init__(self, X, y, random=None):
        """Créer un modèle de classification
        Args:
                dataset (Dataframe): Dataframe d'entrainement du modèle
        """
        super().__init__(X, y, random)
        self.model = RandomForestClassifier(max_depth=20)
        self.model.fit(X=self.X, y=self.y)

class RandomForestFeaturesModel(BaseFeaturesModel):
    def __init__(self, X, y, random=None):
        """Créer un modèle de classification
        Args:
                dataset (Dataframe): Dataframe d'entrainement du modèle
        """
        super().__init__(X, y, random)
        self.model = RandomForestClassifier()
        self.model.fit(X=self.X, y=self.y)


class LogisticRegressionModel(BaseCountModel):
    def __init__(self, X, y, random=None):
        """Créer un modèle de classification
        Args:
                dataset (Dataframe): Dataframe d'entrainement du modèle
        """
        super().__init__(X, y, random)
        self.model = LogisticRegression()
        self.model.fit(X=self.X, y=self.y)
        
class LogisticRegressionFeaturesModel(BaseFeaturesModel):
    def __init__(self, X, y, random=None):
        """Créer un modèle de classification
        Args:
                dataset (Dataframe): Dataframe d'entrainement du modèle
        """
        super().__init__(X, y, random)
        self.model = LogisticRegression()
        self.model.fit(X=self.X, y=self.y)

