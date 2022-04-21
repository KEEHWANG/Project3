import pandas as pd
import psycopg2

connection = psycopg2.connect(
    host="ruby.db.elephantsql.com",
    database="vybsmhcp",
    user="vybsmhcp",
    password="MAVcVWGBpCH9-xL-BLQeNv-b4-CjYhXN")

df=pd.read_csv('diabetes.csv')
cur = connection.cursor()
val_list = df.values.tolist()

cur.execute("DROP TABLE IF EXISTS diabetes;")

cur.execute("""CREATE TABLE diabetes (
                index INT, Pregnancies INT,Glucose INT,Bloodpressure INT,
                SkinThickness INT,Insulin INT,BMI FLOAT,DiabetesPedigreeFunction FLOAT,
                Age INT,Outcome INT);
            """)

for i in val_list[0:]:
    cur.execute("""INSERT INTO diabetes (Pregnancies,Glucose,Bloodpressure,SkinThickness,Insulin,BMI,DiabetespedigreeFunction,Age,Outcome) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",i)

connection.commit()

cur.close()

connection.close()

print(connection)
