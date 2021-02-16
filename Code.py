# Подключаем библиотеки

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import tree, metrics

# Подключаем данные

df = pd.read_csv("CleanedData.csv")
