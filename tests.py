from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from scipy.io.arff import loadarff

import pandas

from tests_dicts import dicts

def extract_string(b: bytes):
    if type(b) != float:
        return b.decode('utf-8')
    return b

random_seed = 1337

file_path = "./datasets/Autism-Adult-Data.arff"

raw_data = loadarff(file_path)
df = pandas.DataFrame(raw_data[0])

df = df.rename(columns = {"austim": "autism", "contry_of_res": "country"})

df = df.map(lambda x: extract_string(x))

df = df.replace('1', 1)
df = df.replace('0', 0)
df = df.replace('yes', 1)
df = df.replace('no', 0)
df = df.replace('YES', 1)
df = df.replace('NO', 0)

df = df.dropna()

positive_labels = df[df["autism"] == 1]

df = pandas.concat([df, positive_labels, positive_labels])
df = df.sample(frac=1)

features = [
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score",
    "jundice",
    "used_app_before",
    "relation"
]

label = ["autism"]

df = df[features + label]

data = df.iloc[:, df.columns != label[0]] 
target = df.iloc[:, -1].values
XTrain, XTest, YTrain, YTest = train_test_split(data, target, test_size=0.2, random_state=random_seed)

# print(XTrain.head())

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, features)
    ]
)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=16,
    min_samples_split=2,
    splitter="random",
    random_state=random_seed
)

model_pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', model)
])

print("Fitting model...")
model_pipeline.fit(XTrain, YTrain)
print("Fitting complete!")


scores = []

print("autistic checker 2000:")
for person, data in dicts.items():
    print(f"Suspeito: {person}")
    total = 0
    for i in range(10):
        key = f"A{i+1}_Score"
        total += data[key]
    
    print(f"Pontuação no teste: {total}")
    
    scores.append((person, total))

    data_df = pandas.DataFrame([data])
    pred = model_pipeline.predict(data_df)

    verdict = "Autista" if pred[0] == 1 else "Não autista"

    print(f"Resultado: {verdict}")

    print("=" * 30)

sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
print("Os + artista que tá tendo (em ordem de arte):")
for score in sorted_scores:
    print(score)