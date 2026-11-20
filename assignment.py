import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv("customer_data.csv")

print(df.head(10))
print("Shape: ", df.shape)

print("Data types: ")
print(df.dtypes)

missing_summ=pd.DataFrame({
    "Missing values" : df.isnull().sum(),
    "Missing percentage": (df.isnull().sum()/len(df))*100,
})

print("Missing value and percentage: ")
print(missing_summ)

numeric_columns=["Age", "Salary", "Experience"]
numeric_imputer=SimpleImputer(strategy="median")
df[numeric_columns]=numeric_imputer.fit_transform(
    df[numeric_columns]
)


catagorical_column=["City", "Gender", "Education"]
catagorical_imputer=SimpleImputer(strategy="most_frequent")
df[catagorical_column]=catagorical_imputer.fit_transform(
    df[catagorical_column]
)


print ("Missing values after task2: ")
print(df.isnull().sum())


"""
handle_unknown="ignore" tells the OneHotEncoder to
ignore catagories that were not present in the
data rather than raising error.
"""
encoder=OneHotEncoder(
    handle_unknown="ignore"
)
encoded_data=encoder.fit_transform(
    df[catagorical_column]
)
encoded_col_names=encoder.get_feature_names_out(
    catagorical_column
)

print("Encoded col names: ")
print(encoded_col_names)
