import pandas as pd

sub = pd.read_csv("submission4.csv")

sub["id"] = sub["id"].astype(int)

sub.to_csv("submission_fixed8.csv", index=False)

print("Готово. Новый файл: submission_fixed.csv")
print(sub.dtypes)