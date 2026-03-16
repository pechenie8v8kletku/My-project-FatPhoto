import pandas as pd
import matplotlib.pyplot as plt
df1 = pd.read_csv('benchmark.csv')
df2 = pd.read_csv('benchmark1.csv')


common_col = df1.columns[0]
df = pd.merge(df1, df2, on=common_col)
plt.figure(figsize=(12, 7))

for column in df.columns:
    if column != common_col:
        plt.plot(df[common_col], df[column], label=column, linewidth=2)
plt.title('Сравнение Loss моделей по эпохам', fontsize=15)
plt.xlabel('Эпохи (Epochs)', fontsize=12)
plt.ylabel('Потери (Loss)', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.show()