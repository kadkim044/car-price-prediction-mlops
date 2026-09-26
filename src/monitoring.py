import psycopg2
import pandas as pd
def load_predictions():
    conn=psycopg2.connect(
        host="127.0.0.1",
        port=5432,
        database="car_prices",
        user="postgres",
        password="postgres"
    )
    cursor=conn.cursor()
    cursor.execute(
        """
            SELECT * FROM predictions;
        """
    )
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    cursor.close()
    conn.close()
    return df

if __name__ == "__main__":
    df = load_predictions()
    print(df)
    print(df.shape)