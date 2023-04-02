import numpy as np
import pandas as pd
from tqdm import tqdm

def pad4(i):
    return "0"*(4-len(str(i)))+str(i)

df1 = pd.read_csv('./submission_EFFICIENTNET_V2_S.csv')
df2 = pd.read_csv('./submission_SHUFFLENET.csv')
df3 = pd.read_csv('./submission_vgg19bn.csv')
df4 = pd.read_csv('./submission_resnext101.csv')

dfs = [df1, df2, df3, df4]
scores = [0.83066, 0.78533, 0.85466, 0.85533]   
results = np.ones(3000, dtype=int) * -1  

# reference https://blog.csdn.net/chiyukunpeng/article/details/107980934
for i in tqdm(range(3000)):
    dic = {}
    for j in range(len(dfs)):
        p = dfs[j].iloc[i]['Category']  
        if p == -1: continue
        if p not in dic.keys():
            dic[p] = scores[j]
        else:
            dic[p] += scores[j]

    if not dic: continue
    results[i] = sorted(dic, key=lambda x: dic[x])[-1]

df = pd.DataFrame() 
df["Id"] = [pad4(i) for i in range(0, len(results))]
df['Category'] = results
df.to_csv('ensemble.csv', sep=',', index=False)