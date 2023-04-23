import numpy as np
import pandas as pd
from tqdm import tqdm

def pad4(i):
    return "0"*(4-len(str(i)))+str(i)

df1 = pd.read_csv('./QA_model.csv')
df2 = pd.read_csv('./roberta_model.csv')
df3 = pd.read_csv('./macbert_model.csv')

dfs = [df1, df2, df3]
scores = [0.79738, 0.83427, 0.83314]   
results = np.empty(3524, dtype=object)

# reference https://blog.csdn.net/chiyukunpeng/article/details/107980934
for i in tqdm(range(3524)):
    dic = {}
    for j in range(len(dfs)):
        p = dfs[j].iloc[i]['Answer']
        if p == '-1': continue
        if p not in dic.keys():
            dic[p] = scores[j]
        else:
            dic[p] += scores[j]

    if not dic:
        results[i] = '-1'
        continue
    results[i] = sorted(dic, key=lambda x: dic[x])[-1]

df = pd.DataFrame()
df["Id"] = [i for i in range(0, len(results))]
df['Answer'] = results
df.to_csv('ensemble.csv', sep=',', index=False)


