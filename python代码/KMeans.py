from sklearn.cluster import KMeans
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
bys_p=pd.DataFrame(pd.read_excel("四算法结果概率.xlsx",sheet_name=0))#贝叶斯结果概率
CNN_p=pd.DataFrame(pd.read_excel("四算法结果概率.xlsx",sheet_name=2))#多层神经网络概率
ran_p=pd.DataFrame(pd.read_excel("四算法结果概率.xlsx",sheet_name=3))#随机森林概率
sum_p=(bys_p+CNN_p+ran_p)/3
pre_dict={}
pre_dict['高钾']=[]
pre_dict['铅钡']=[]
count=0
for i in bys_p['预测结果']:
    if(i):
        pre_dict['铅钡'].append(sum_p.loc[count, '铅钡概率'])
    else:
        pre_dict['高钾'].append(sum_p.loc[count, '高钾概率'])
    count+=1

datax=[]
datay=[]
Forms=pd.DataFrame(pd.read_excel("datafenghua.xlsx",sheet_name=0))
Na2O=[]
P2O5=[]
SO2 =[]
CuO=[]
k_dict={}
k_dict['文物编号']=[]
k_dict['文物采样点']=[]
Pb_dict={}
Pb_dict['文物编号']=[]
Pb_dict['文物采样点']=[]
for i in range(len(bys_p)):
    if(bys_p['预测结果'][i]):
        Pb_dict['文物编号'].append(Forms.loc[i,'文物编号'])
        Pb_dict['文物采样点'].append(Forms.loc[i, '文物采样点'])
        P2O5.append(Forms.loc[i,'五氧化二磷(P2O5)'])
        SO2.append(Forms.loc[i,'二氧化硫(SO2)'])
        CuO.append(Forms.loc[i,'氧化铜(CuO)'])
    else:
        k_dict['文物编号'].append(Forms.loc[i,'文物编号'])
        k_dict['文物采样点'].append(Forms.loc[i, '文物采样点'])
        Na2O.append(Forms.loc[i,'氧化钠(Na2O)'])
#数据标准化
def data_stand(sr):
    sc=[]
    for i in range(len(sr)):
        sc.append((sr[i] - min(sr)) / (max(sr) - min(sr)))
    return sc

print(min(Na2O))
print(max(Na2O))
print(min(CuO))
print(max(CuO))
Na2O=data_stand(Na2O)
P2O5=data_stand(P2O5)
P2O5_SO2=[]
for i in range(len(P2O5)):
    P2O5_SO2.append(P2O5[i]+SO2[i])
print(min(P2O5_SO2))
print(max(P2O5_SO2))
P2O5_SO2=data_stand(P2O5_SO2)

K_Na=[]
for i in range(len(Na2O)):
    K_Na.append([pre_dict['高钾'][i],Na2O[i]])

Pb_sp=[]
for i in range(len(P2O5_SO2)):
    Pb_sp.append([pre_dict['铅钡'][i],P2O5_SO2[i]])

Pb_Cu=[]
for i in range(len(P2O5_SO2)):
    Pb_Cu.append([pre_dict['铅钡'][i],CuO[i]])

######################################################################钾玻璃-Na2O######################################################################
estimator = KMeans(n_clusters=2)#聚类分类数
estimator.fit(K_Na)#聚类
label_pred = estimator.labels_
print("聚类标签:")
print(label_pred)
centroids = estimator.cluster_centers_
print("聚类中心:")
print(centroids)

k_dict['亚分类']=label_pred
data=pd.DataFrame(k_dict)#将字典转换成为数据框
pd.DataFrame(k_dict).to_excel('高钾玻璃亚分类.xlsx',sheet_name='data',index=False)

mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')  # 美化

color = ["#6A5ACD", '#228B22', '#B8860B', '#B22222', '#FF69B4',
         '#1E90FF', '#4B0082', "#00FF7F", '#FFFACD', '#0000FF']

K_Na=np.array(K_Na)
flag0=0
flag1=0
x_z=(centroids[0][0]+centroids[1][0])/2
y_z=(centroids[0][1]+centroids[1][1])/2
#plt.axvline(x_z,c='#A19B9B')#竖线
plt.axhline(centroids[0][1],c="#6A5ACD", linestyle='--')  # 虚线
plt.axhline(centroids[1][1],c='#228B22', linestyle='--')  # 虚线
plt.axhline(y_z,c='#A19B9B')#横线
plt.axhline(y_z+y_z/10,c='#A19B9B', linestyle='--')  # 虚线
plt.axhline(y_z-y_z/10,c='#A19B9B', linestyle='--')  # 虚线
x_min=1
for i in range(len(label_pred)):
    x_min=min(x_min,K_Na[i][0])
    if(flag0==0 and label_pred[i]==0):
        plt.scatter(K_Na[i][0],K_Na[i][1], c=color[label_pred[i]], s=30, label=f"第一类")
        flag0=1
    elif(flag1==0 and label_pred[i]==1):
        plt.scatter(K_Na[i][0], K_Na[i][1], c=color[label_pred[i]], s=30, label=f"第二类")
        flag1 = 1
    else:
        plt.scatter(K_Na[i][0], K_Na[i][1], c=color[label_pred[i]], s=30)

plt.scatter(centroids[0][0],centroids[0][1], c=color[0], s=150, label=f"聚点",marker ='*')
plt.scatter(centroids[1][0],centroids[1][1], c=color[1], s=150,label=f"聚点" , marker ='*')

plt.title('高钾玻璃亚分类聚类图',fontsize=12,c='#322424')
plt.text(x_min-0.0063,y_z,"划分线（"+str(format(y_z,'.3f'))+")",verticalalignment='bottom',horizontalalignment='left')
plt.text(x_min-0.0063,centroids[0][1],"聚点线（"+str(format(centroids[0][1],'.3f'))+")",verticalalignment='bottom',horizontalalignment='left')
plt.text(x_min-0.0063,centroids[1][1],"聚点线（"+str(format(centroids[1][1],'.3f'))+")",verticalalignment='bottom',horizontalalignment='left')
plt.ylabel("Na2O含量归一化值",fontsize=12)
plt.xlabel("类型概率均值",fontsize=12)
plt.legend(loc=0, ncol=2,)  # 加上图列
plt.show()

##############################################################铅钡-磷硫#########################################################################
estimator = KMeans(n_clusters=2)#聚类分类数
estimator.fit(Pb_sp)#聚类
label_pred = estimator.labels_
print("聚类标签:")
print(label_pred)
centroids = estimator.cluster_centers_
print("聚类中心:")
print(centroids)

Pb_dict['亚分类']=label_pred
data=pd.DataFrame(Pb_dict)#将字典转换成为数据框
pd.DataFrame(Pb_dict).to_excel('铅钡-磷硫亚分类.xlsx',sheet_name='data',index=False)


mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')  # 美化

color = ["#6A5ACD", '#228B22', '#B8860B', '#B22222', '#FF69B4',
         '#1E90FF', '#4B0082', "#00FF7F", '#FFFACD', '#0000FF']

K_Na=np.array(Pb_sp)
flag0=0
flag1=0
x_z=(centroids[0][0]+centroids[1][0])/2
y_z=(centroids[0][1]+centroids[1][1])/2
#plt.axvline(x_z,c='#A19B9B')#竖线
plt.axhline(centroids[0][1],c="#6A5ACD", linestyle='--')  # 虚线
plt.axhline(centroids[1][1],c='#228B22', linestyle='--')  # 虚线
plt.axhline(y_z,c='#A19B9B')#横线
plt.axhline(y_z+y_z/10,c='#A19B9B', linestyle='--')  # 向上10%
plt.axhline(y_z-y_z/10,c='#A19B9B', linestyle='--')  # 向下10%
x_min=1
for i in range(len(label_pred)):
    x_min=min(x_min,K_Na[i][0])
    if(flag0==0 and label_pred[i]==0):
        plt.scatter(K_Na[i][0],K_Na[i][1], c=color[label_pred[i]], s=30, label=f"第一类")
        flag0=1
    elif(flag1==0 and label_pred[i]==1):
        plt.scatter(K_Na[i][0], K_Na[i][1], c=color[label_pred[i]], s=30, label=f"第二类")
        flag1 = 1
    else:
        plt.scatter(K_Na[i][0], K_Na[i][1], c=color[label_pred[i]], s=30)

plt.scatter(centroids[0][0],centroids[0][1], c=color[0], s=150, label=f"聚点",marker ='*')
plt.scatter(centroids[1][0],centroids[1][1], c=color[1], s=150,label=f"聚点" , marker ='*')

plt.title('铅钡玻璃—磷硫亚分类聚类图',fontsize=12,c='#322424')
plt.text(x_min-0.0063,y_z,"划分线（"+str(format(y_z,'.3f'))+")",verticalalignment='bottom',horizontalalignment='left')
plt.text(x_min-0.0063,centroids[0][1],"聚点线（"+str(format(centroids[0][1],'.3f'))+")",verticalalignment='bottom',horizontalalignment='left')
plt.text(x_min-0.0063,centroids[1][1],"聚点线（"+str(format(centroids[1][1],'.3f'))+")",verticalalignment='bottom',horizontalalignment='left')
plt.ylabel("P/S含量归一化值",fontsize=12)
plt.xlabel("类型概率均值",fontsize=12)
plt.legend(loc=0, ncol=2,)  # 加上图列
plt.show()

#############################################################铅钡-铜####################################################################
estimator = KMeans(n_clusters=2)#聚类分类数
estimator.fit(Pb_Cu)#聚类
label_pred = estimator.labels_
print("聚类标签:")
print(label_pred)
centroids = estimator.cluster_centers_
print("聚类中心:")
print(centroids)

Pb_dict['亚分类']=label_pred
data=pd.DataFrame(Pb_dict)#将字典转换成为数据框
pd.DataFrame(Pb_dict).to_excel('铅钡-铜亚分类.xlsx',sheet_name='data',index=False)

mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')  # 美化

color = ["#6A5ACD", '#228B22', '#B8860B', '#B22222', '#FF69B4',
         '#1E90FF', '#4B0082', "#00FF7F", '#FFFACD', '#0000FF']

K_Na=np.array(Pb_Cu)
flag0=0
flag1=0
x_z=(centroids[0][0]+centroids[1][0])/2
y_z=(centroids[0][1]+centroids[1][1])/2
#plt.axvline(x_z,c='#A19B9B')#竖线
plt.axhline(centroids[0][1],c="#6A5ACD", linestyle='--')  # 虚线
plt.axhline(centroids[1][1],c='#228B22', linestyle='--')  # 虚线
plt.axhline(y_z,c='#A19B9B')#横线
plt.axhline(y_z+y_z/10,c='#A19B9B', linestyle='--')  # 虚线
plt.axhline(y_z-y_z/10,c='#A19B9B', linestyle='--')  # 虚线
x_min=1
for i in range(len(label_pred)):
    x_min=min(x_min,K_Na[i][0])
    if(flag0==0 and label_pred[i]==0):
        plt.scatter(K_Na[i][0],K_Na[i][1], c=color[label_pred[i]], s=30, label=f"第一类")
        flag0=1
    elif(flag1==0 and label_pred[i]==1):
        plt.scatter(K_Na[i][0], K_Na[i][1], c=color[label_pred[i]], s=30, label=f"第二类")
        flag1 = 1
    else:
        plt.scatter(K_Na[i][0], K_Na[i][1], c=color[label_pred[i]], s=30)

plt.scatter(centroids[0][0],centroids[0][1], c=color[0], s=150, label=f"聚点",marker ='*')
plt.scatter(centroids[1][0],centroids[1][1], c=color[1], s=150,label=f"聚点" , marker ='*')

plt.title('铅钡玻璃—铜亚分类聚类图',fontsize=12,c='#322424')
plt.text(x_min-0.0063,y_z,"划分线（"+str(format(y_z,'.3f'))+")",verticalalignment='bottom',horizontalalignment='left')
plt.text(x_min-0.0063,centroids[0][1],"聚点线（"+str(format(centroids[0][1],'.3f'))+")",verticalalignment='bottom',horizontalalignment='left')
plt.text(x_min-0.0063,centroids[1][1],"聚点线（"+str(format(centroids[1][1],'.3f'))+")",verticalalignment='bottom',horizontalalignment='left')
plt.ylabel("CuO含量归一化值",fontsize=12)
plt.xlabel("类型概率均值",fontsize=12)
plt.legend(loc=0, ncol=2,)  # 加上图列
plt.show()