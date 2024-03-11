"""This code is to populate a database (temporary) that will
 hold the dataset to optimize the download and generation of the files for each patient """

import os
import pandas as pd
from pandas import DataFrame
from datetime import datetime
import mysql.connector

main_file = 'Datasets/Glucose_measurements.csv'
data = pd.read_csv(main_file)

def save_data_to_db(data):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="granada"
    )

    mycursor = mydb.cursor()

    for index, row in data.iterrows():
        pat_id = row['Patient_ID']
        measurement_date = row['Measurement_date']
        measurement_time = row['Measurement_time']
        measure = row['Measurement']

        sql = "INSERT INTO granada (patient_id, measurement_date,measurement_time,measurement) VALUES (%s, %s, %s, %s)"
        val = (pat_id, measurement_date, measurement_time, measure)
        mycursor.execute(sql, val)

        mydb.commit()


def save_data_to_files():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="granada"
    )

    mycursor = mydb.cursor()

    sql = "select DISTINCT patient_id from granada"
    mycursor.execute(sql)

    patients = [i[0] for i in mycursor.fetchall()]

    for i in range(0, len(patients)):
        one_patient = patients[i]

        sql_patient = f"SELECT patient_id,measurement,CONCAT(measurement_date,' ',measurement_time) as measurement_date FROM `granada` WHERE patient_id = '{one_patient}'"

        mycursor.execute(sql_patient)
        patient_details = mycursor.fetchall()

        name = []
        measurement_date = []
        val = []

        for patient in patient_details:
            patient_name = patient[0]
            patient_val = patient[1]
            patient_date = patient[2]
            measure_date = datetime.strptime(patient_date, '%Y-%m-%d %H:%M:%S')
            name.append(patient_name)
            val.append(patient_val)
            measurement_date.append(measure_date)

        file_name = "DataGranada/" + one_patient + ".xlsx"
        df = DataFrame({'date': measurement_date, 'mg/dl': val})
        if os.path.isfile(file_name):
            os.remove(file_name)
        df.to_excel(file_name, sheet_name='sheet1', index=False)

save_data_to_db(data)
