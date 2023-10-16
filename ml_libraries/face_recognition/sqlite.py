import os
import sqlite3
import sys

# package parameter
this = sys.modules[__name__]
this.sqlite_db_path = './temp_model/face_recognition/face_feature.db'
this.sqlite_conn = None
this.sqlite_cursor = None


def update_db(status):
    if not os.path.exists('./temp_model/face_recognition'):
        os.makedirs('./temp_model/face_recognition')

    if os.path.exists(this.sqlite_db_path) and status:
        os.remove(this.sqlite_db_path)

    this.sqlite_conn = sqlite3.connect(this.sqlite_db_path)
    this.sqlite_cursor = this.sqlite_conn.cursor()


def create_table():
    this.sqlite_conn.execute('''CREATE TABLE IF NOT EXISTS people (id INT NOT NULL, firstname character(20) NOT NULL, surname character(20) NOT NULL,  nickname character(15) NOT NULL, age integer NOT NULL, gender character(10) NOT NULL, face_id FLOAT [] NOT NULL)''')


def insert_varible_into_table(id, firstname, surname, nickname, age, gender, face_id):
    sqlite_insert = """INSERT INTO people (id, firstname, surname, nickname, age, gender, face_id) VALUES (?, ?, ?, ?, ?, ?, ?)"""
    data_tuple = (id, firstname, surname, nickname, age, gender, face_id)
    this.sqlite_cursor.execute(sqlite_insert, data_tuple)
    this.sqlite_conn.commit()


def update_sqlite_person_info(command, data):
    this.sqlite_cursor.execute(command, data)
    this.sqlite_conn.commit()


def getFeatureInfo():
    id = []
    names = []
    surnames = []
    nickname = []
    ages = []
    genders = []
    face_id = []

    sql_select = """select * from people"""
    this.sqlite_cursor.execute(sql_select)
    records = this.sqlite_cursor.fetchall()

    for person in records:

        if person[6] == 'null':
            continue

        id.append(person[0])
        names.append("".join(person[1].split()))
        surnames.append("".join(person[2].split()))
        nickname.append("".join(person[3].split()))
        ages.append(person[4])
        genders.append("".join(person[5].split()))
        face_id.append(eval(person[6]))

    return id, names, surnames, nickname, ages, genders, face_id
