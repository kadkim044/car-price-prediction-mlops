import pandas as pd



def load_and_prepare_data():
    df=pd.read_csv("../data/raw/used_car_dataset.csv")
    df=df.drop_duplicate()