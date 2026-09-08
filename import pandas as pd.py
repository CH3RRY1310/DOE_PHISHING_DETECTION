import pandas as pd

df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

print(df.shape)
print(df.head())
print(df.info())

print(df["label"].value_counts())

df = df.drop_duplicates(subset=["URL"])

print("Dataset size:", df.shape)

print("Duplicate URLs:", df["URL"].duplicated().sum())