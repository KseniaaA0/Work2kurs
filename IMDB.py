import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB  
from google.colab import drive


drive.mount('/content/drive')


df = pd.read_csv("/content/IMDB_daataset.csv",
                 on_bad_lines='skip',
                 engine='python')

from google.colab import drive
drive.mount('/content/drive')


print("Начинаем предобработку текста...")


print(f"Пустые значения в колонке 'review': {df['review'].isnull().sum()}")


df['review'] = df['review'].fillna('')


df['review'] = df['review'].str.lower()

# Удаляем пунктуацию
import string
def remove_punctuation(text):
    if isinstance(text, str):
        return ''.join(char for char in text if char not in string.punctuation)
    return text

df['review'] = df['review'].apply(remove_punctuation)

print("Предобработка завершена!")

# Проверим, что датасеты создались правильно
print("Проверка датасетов...")
print(f"Размер df_imb: {len(df_imb) if 'df_imb' in locals() else 'не создан'}")
print(f"Размер df_half: {len(df_half) if 'df_half' in locals() else 'не создан'}")

# Если датасеты пустые, пересоздадим их
if 'df_imb' not in locals() or len(df_imb) == 0:
    print("Пересоздаем датасеты...")

    # Проверим наличие данных
    print(f"Размер исходного df: {len(df)}")
    print("Уникальные значения sentiment:", df['sentiment'].unique())
    print("Распределение sentiment:")
    print(df['sentiment'].value_counts())

    # Создаем датасеты с проверкой доступных данных
    all_positive = df[df['sentiment'] == 'positive']
    all_negative = df[df['sentiment'] == 'negative']

    print(f"Доступно positive: {len(all_positive)}")
    print(f"Доступно negative: {len(all_negative)}")

    # Берем доступное количество данных
    n_pos_imb = min(45000, len(all_positive))
    n_neg_imb = min(5000, len(all_negative))

    n_pos_bal = min(25000, len(all_positive))
    n_neg_bal = min(25000, len(all_negative))

    df_positive_imb = all_positive.iloc[:n_pos_imb]
    df_negative_imb = all_negative.iloc[:n_neg_imb]
    df_imb = pd.concat([df_positive_imb, df_negative_imb])

    df_positive_bal = all_positive.iloc[:n_pos_bal]
    df_negative_bal = all_negative.iloc[:n_neg_bal]
    df_half = pd.concat([df_positive_bal, df_negative_bal])

    print(f"Создан df_imb размером: {len(df_imb)}")
    print(f"Создан df_half размером: {len(df_half)}")


if len(df_imb) == 0 or len(df_half) == 0:
    print("Ошибка: Не удалось создать датасеты. Проверьте данные.")
    exit()

# Разделение на train/test
from sklearn.model_selection import train_test_split

train, test = train_test_split(df_imb, test_size=0.20, random_state=42)
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

train_half, test_half = train_test_split(df_half, test_size=0.20, random_state=42)
train_x_half, train_y_half = train_half['review'], train_half['sentiment']
test_x_half, test_y_half = test_half['review'], test_half['sentiment']



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(train_x)

train_x_vector = tfidf.transform(train_x)
test_x_vector = tfidf.transform(test_x)
train_x_half_vector = tfidf.transform(train_x_half)
test_x_half_vector = tfidf.transform(test_x_half)


log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)
log_reg_half = LogisticRegression()
log_reg_half.fit(train_x_half_vector, train_y_half)

dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)
dec_tree_half = DecisionTreeClassifier()
dec_tree_half.fit(train_x_half_vector, train_y_half)


gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)
gnb_half = GaussianNB()
gnb_half.fit(train_x_half_vector.toarray(), train_y_half)


DEC_TREE_ACC = dec_tree.score(test_x_vector, test_y)
GNB_ACC = gnb.score(test_x_vector.toarray(), test_y)
LR_ACC = log_reg.score(test_x_vector, test_y)

DEC_TREE_ACC_half = dec_tree_half.score(test_x_half_vector, test_y_half)
GNB_ACC_half = gnb_half.score(test_x_half_vector.toarray(), test_y_half)
LR_ACC_half = log_reg_half.score(test_x_half_vector, test_y_half)


import matplotlib.pyplot as plt

models = ['DEC_TREE', 'GNB', 'LR']
accuracy_scores = [DEC_TREE_ACC*100, GNB_ACC*100, LR_ACC*100]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracy_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison - Imbalanced Dataset')
plt.ylim(0, 100)
for i, v in enumerate(accuracy_scores):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center')
plt.show()

accuracy_scores_half = [DEC_TREE_ACC_half*100, GNB_ACC_half*100, LR_ACC_half*100]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracy_scores_half, color=['lightblue', 'green', 'pink'])
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison - Balanced Dataset')
plt.ylim(0, 100)
for i, v in enumerate(accuracy_scores_half):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center')
plt.show()
