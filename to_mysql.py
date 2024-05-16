import pandas as pd
import pymysql
from sqlalchemy import create_engine
engine = create_engine("mysql://root:123456@localhost:3307/spam?charset=utf8")

df = pd.read_csv('datafile/带标签信息.txt',sep='\t',names=['lable','youjian'])
df.index.name = 'id'
df.to_sql('web_dataset',engine,if_exists='append')
df1 = pd.read_csv('stopwords.txt',names=['name'])
df1.index.name = 'id'
df1.to_sql('web_stopword',engine,if_exists='append')