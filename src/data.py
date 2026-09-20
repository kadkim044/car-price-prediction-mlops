import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_prepare_data():
    df=pd.read_csv("data/raw/used_car_dataset.csv")
    df=df.drop_duplicates()
    df["kmDriven"]=df["kmDriven"].str.replace("km","")
    df["kmDriven"]=df["kmDriven"].str.replace(",","")
    df["kmDriven"]=df["kmDriven"].astype(float)
    df["AskPrice"]=df["AskPrice"].str.replace("₹","")
    df["AskPrice"]=df["AskPrice"].str.replace(",","")
    df["AskPrice"]=df["AskPrice"].astype(float)
    df = df[(df["kmDriven"] <= 300000) | (df["kmDriven"].isna())]
    df=df.drop(columns=["Age"],axis=1)
    features=["Brand","model","Year","Transmission","kmDriven","Owner","FuelType"]
    target="AskPrice"
    X_train,X_test,y_train,y_test=train_test_split(df[features],df[target],train_size=0.8,random_state=42)
    return X_train,X_test,y_train,y_test
    
if __name__=="__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\nMissing kmDriven:")
    print("Train:", X_train["kmDriven"].isna().sum())
    print("Test:", X_test["kmDriven"].isna().sum())