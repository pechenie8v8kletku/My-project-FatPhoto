import os

path = r"different_datasets/real people stranniye"

for i in range(4, 40):
    pathx = os.path.join(path, str(i))
    os.makedirs(pathx, exist_ok=True)
