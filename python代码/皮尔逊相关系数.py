import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import seaborn as sb

mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

lis=['二氧化硅(SiO2)',
     '氧化钠(Na2O)',
     '氧化钾(K2O)',
     '氧化钙(CaO)',
     '氧化镁(MgO)',
     '氧化铝(Al2O3)',
     '氧化铁(Fe2O3)',
     '氧化铜(CuO)',
     '氧化铅(PbO)',
     '氧化钡(BaO)',
     '五氧化二磷(P2O5)',
     '氧化锶(SrO)',
     '氧化锡(SnO2)',
     '二氧化硫(SO2)']

Forms=pd.DataFrame(pd.read_excel("datafenghua.xlsx",sheet_name=0))

from scipy.stats import stats
correlation_lis= {}
for i in lis:
    correlation_lis[i]=[]
    for j in lis:
        a = pd.Series(Forms[i].to_list()) 
        b = pd.Series(Forms[j].to_list())
        r,p=stats.pearsonr(a,b)
        r=float(r)
        correlation_lis[i].append(r)
data=pd.DataFrame(correlation_lis,index =lis)#将字典转换成为数据框
pd.DataFrame(data).to_excel('皮尔逊相关系数.xlsx',sheet_name='data',index=lis)
print(data)
sb.heatmap(data = data,cmap="YlGnBu")#cmap设置色系
plt.show()