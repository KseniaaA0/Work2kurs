import pprint 
import numpy as np  
import pandas as pd 

if __name__ == "__main__": 
    
    ptd1 = pd.read_csv('/content/part_01.csv')
    ptd2 = pd.read_csv('/content/part_02.csv')
    ptd3 = pd.read_csv('/content/part_03.csv')
    ptd4 = pd.read_csv('/content/part_04.csv')
    ptd5 = pd.read_csv('/content/part_05.csv')
    ptd6 = pd.read_csv('/content/part_06.csv')
    ptd7 = pd.read_csv('/content/part_07.csv')
    ptd8 = pd.read_csv('/content/part_08.csv')
    ptd9 = pd.read_csv('/content/part_09.csv')
    ptd10 = pd.read_csv('/content/part_10.csv')
    ptd11 = pd.read_csv('/content/part_11.csv')
    ptd12 = pd.read_csv('/content/part_12.csv')

    ptds = [ptd1, ptd2, ptd3, ptd4, ptd5, ptd6, ptd7, ptd8, ptd9, ptd10, ptd11, ptd12]

    shapes = [ptd1.shape[1], ptd2.shape[1], ptd3.shape[1], ptd4.shape[1], ptd5.shape[1], ptd6.shape[1], ptd7.shape[1], ptd8.shape[1], ptd9.shape[1], ptd10.shape[1], ptd11.shape[1], ptd12.shape[1]]

    db1 = pd.concat([ptd1, ptd9], ignore_index=True)
    db2 = pd.concat([ptd2, ptd11], ignore_index=True)
    db3 = pd.concat([ptd4, ptd7], ignore_index=True)
    db4 = pd.concat([ptd5, ptd12], ignore_index=True)
    db5 = pd.concat([ptd6, ptd10], ignore_index=True)
    db6 = pd.concat([ptd3, ptd8], ignore_index=True)

  
    dbraw = [db1, db2, db3, db4, db5, db6]

    
    db = db1


shapes_all = [ptd1.shape, ptd2.shape, ptd3.shape, ptd4.shape, ptd5.shape, ptd6.shape, ptd7.shape, ptd8.shape, ptd9.shape, ptd10.shape, ptd11.shape, ptd12.shape]
shapes_all  # Выводим размеры всех таблиц (строки и столбцы)


for i in range(1, len(dbraw)):
    db = db.merge(dbraw[i], left_on='ID', right_on='ID')


db.drop('APP_EMP_TYPE', axis=1, inplace=True)


db.drop_duplicates()


list_sm = list(db.select_dtypes(include='object').columns)


en_to_rus = {'specialist': 'специалист', 'manager': 'менеджер', 'friend': 'друг', 'mother': 'мать', 'relative': 'близкий ро', 'brother': 'брат', 'sister': 'сестра', 'father': 'отец', 'other': 'дальний ро', 'daughter': 'дочь', 'son': 'сын'}


for sln in list_sm:
    db[sln].where(db[sln].str.replace(' ','').str.isalpha(), np.nan, inplace=True)  # Заменяем неалфавитные символы на NaN
    db[sln] = db[sln].str.lower()  # Преобразуем строки к нижнему регистру
    db[sln] = db[sln].replace(en_to_rus)  # Заменяем английские слова на русские


print(db['CLNT_TRUST_RELATION'].unique())
print(db['APP_MARITAL_STATUS'].unique())
print(db['CLNT_JOB_POSITION'].nunique())


db = db.sort_values('ID')


print(db.head(10))


print("Rows: " + str(db.shape[0]) + " Cols: " + str(db.shape[1]))


for i in range(1, len(dbraw)):
    db = db.merge(dbraw[i], left_on='ID', right_on='ID')


db.drop_duplicates()

# Сортируем таблицу по колонке 'ID' (ещё раз для чистоты)
db = db.sort_values('ID')

# Выводим первые 10 строк таблицы (повторно)
print(db.head(10))

# Выводим количество строк и столбцов в таблице (повторно)
print("Rows: " + str(db.shape[0]) + " Cols: " + str(db.shape[1]))
