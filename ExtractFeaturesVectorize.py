import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
import joblib # type: ignore
import argparse
import re

# Create the parser
parser = argparse.ArgumentParser(prog="Vectorizer",description="Extrait les features du jeu de données et les sauvegarde dans un fichier")

# Add arguments
parser.add_argument('--TFile', type=str, help="Chemin du jeu de données à vectoriser", required=True)
parser.add_argument('--VectType', type = str, help="Type de vectorisation entre tf-id  (tf) ou features (feat) ou ensemble (tf-feat)", required=True)
parser.add_argument('--OFile',type = str,help="Chemin du jeu de données vectorisé")
# Parse arguments
args = parser.parse_args()


df = pd.read_csv(
    f"{args.TFile}", delimiter="\t", header=None, names=["spam", "text"]
)

if args.VectType in ["feat","tf-feat"]:
    df['has_phone_number'] = df['text'].apply(lambda x: 1 if re.search(r'\b\d{10,}\b', x) else 0)
    df['has_currency_symbol'] = df['text'].apply(lambda x: 1 if re.search(r'[\$\€\£]', x) else 0)
    df["has_special_characters"] = df['text'].apply(lambda x: 1 if re.search(r'[!@#$%^&*(),.?":{}|<>]', x) else 0)
    df['message_length'] = df['text'].apply(len)
    df["uppercase_proportion"] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
    df["has_url"] = df['text'].apply(lambda x: 1 if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x) else 0)
    df["non_alpha_charaters_proportion"] = df['text'].apply(lambda x: sum(1 for c in x if not c.isalpha()) / len(x) if len(x) > 0 else 0)

if args.VectType in ["tf","tf-feat"]:
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])

if args.VectType == "feat":
    features = ["message_length", "has_phone_number", "has_currency_symbol", "has_special_characters", "uppercase_proportion", "has_url", "non_alpha_charaters_proportion"]
    x = df[features]
elif args.VectType == "tf":
    x = pd.DataFrame(X.toarray())
else:
    features = ["message_length", "has_phone_number", "has_currency_symbol", "has_special_characters", "uppercase_proportion", "has_url", "non_alpha_charaters_proportion"]
    x = pd.concat([pd.DataFrame(X.toarray()), df[features]], axis=1)


df_vectorized = x

filepath=""
if args.OFile:
    filepath =args.OFile+'.joblib'   
else:
    filepath="VectorizedDataset.joblib"
joblib.dump((x,df["spam"]),filepath)