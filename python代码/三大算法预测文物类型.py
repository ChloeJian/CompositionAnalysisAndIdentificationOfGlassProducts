import pandas as pd
import math
import warnings

from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
#表单三
Forms_three=pd.DataFrame(pd.read_excel("附件.xlsx",sheet_name=2))
chemical_composition=Forms_three.to_dict('dict')
for i in range(len(Forms_three)):
    count_sum=0
    nan_number=0
    nan_index=[]
    for s in Forms_three:
        if(s !='文物编号' and s!= '表面风化' and math.isnan(Forms_three.loc[i,s])):
            nan_number+=1
            nan_index.append(s)
        elif(s !='文物编号' and s!= '表面风化'):
            count_sum+=Forms_three.loc[i,s]
    count_sum=100-count_sum
    for s in nan_index:
        chemical_composition[s][i]=count_sum/nan_number
data=pd.DataFrame(chemical_composition)#将字典转换成为数据框
try:
    pd.DataFrame(data).to_excel('处理后的表单三数据.xlsx',sheet_name='data',index=False)
except:
    print("文件已打开！")
yuce_data=chemical_composition
chemical_composition=Forms_three.to_dict('dict')
yuce_data.pop('文物编号')
yuce_data['风化数字化']={}
for i in range(len(yuce_data['表面风化'])):
    if(yuce_data['表面风化']=='风化'):
        yuce_data['风化数字化'][i]=1
    else:
        yuce_data['风化数字化'][i] = 0
yuce_data.pop('表面风化')
yuce_datax=[[] for j in range(len(Forms_three))]
for i in range(len(Forms_three)):
    for j in yuce_data:
        yuce_datax[i].append(yuce_data[j][i])
print(yuce_datax)
#波动测试使用
'''import numpy as np
yuce=np.array(yuce_datax)
import random
for i in range(len(yuce)):
    for j in range(len(yuce[i])):
        yuce+=yuce*random.randint(-100,100)*0.001 #-100,100 表示上下10%的波动
#yuce=yuce+yuce*0.4 #整体上下波动
#yuce=yuce-yuce*0.4
yuce_datax=yuce.tolist()'''
'''无波动结果：
贝叶斯预测结果：
[0 1 1 1 1 0 0 1]
神经网络预测结果：
[0 1 1 1 1 0 0 1]
随机森林预测结果：
[0 1 1 1 1 0 0 1]'''
pre_pro=[[0 for i in range(8)] for j in range(3)]
######################################################贝叶斯预测#####################################################
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
Forms=pd.DataFrame(pd.read_excel("datafenghua.xlsx",sheet_name=0))
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
     '二氧化硫(SO2)',
     '风化数字化']
data=Forms.to_dict('dict')
datax=[[] for j in range(len(Forms))]
datay=[]
for i in range(len(Forms)):
    for j in data:
        if(j=='类型数字化'):
            datay.append(data['类型数字化'][i])
        elif (j in lis):
            datax[i].append(data[j][i])
clf = MultinomialNB()
clf.fit(datax,datay)
print("贝叶斯预测结果：")
print(clf.predict(yuce_datax))
pre=clf.predict(yuce_datax)
pro=clf.predict_proba(yuce_datax)
for i in range(len(pre)):
    pre_pro[0][i]=pro[i][pre[i]]
chemical_composition['贝叶斯预测结果']= {}
for i in range(len(pre)):
    if(pre[i]):
        chemical_composition['贝叶斯预测结果'][i]='铅钡'
    else:
        chemical_composition['贝叶斯预测结果'][i] = '高钾'
##########################################################神经网络预测#########################################
from sklearn import neural_network
mlp=neural_network.MLPClassifier(hidden_layer_sizes=(20), #隐藏层
                                activation='relu',  #激活函数
                                solver='adam',
                                alpha=0.0001,  #正则化项系数
                                batch_size='auto',
                                learning_rate='constant',  #学习率
                                learning_rate_init=0.001,
                                power_t=0.5,
                                max_iter=300,#迭代次数
                                tol=1e-4)
mlp.fit(datax,datay)
print("神经网络预测结果：")
print(mlp.predict(yuce_datax))
pre=mlp.predict(yuce_datax)
pro=mlp.predict_proba(yuce_datax)
for i in range(len(pre)):
    pre_pro[1][i]=pro[i][pre[i]]
chemical_composition['神经网络预测结果']= {}
for i in range(len(pre)):
    if (pre[i]):
        chemical_composition['神经网络预测结果'][i] = '铅钡'
    else:
        chemical_composition['神经网络预测结果'][i] = '高钾'
#################################################随机森林#############################################
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(datax,datay)
print("随机森林预测结果：")
print(clf.predict(yuce_datax))
pre=clf.predict(yuce_datax)
pro=clf.predict_proba(yuce_datax)
for i in range(len(pre)):
    pre_pro[2][i]=pro[i][pre[i]]
chemical_composition['随机森林预测结果']= {}
for i in range(len(pre)):
    if (pre[i]):
        chemical_composition['随机森林预测结果'][i] = '铅钡'
    else:
        chemical_composition['随机森林预测结果'][i] = '高钾'
data=pd.DataFrame(chemical_composition)#将字典转换成为数据框
try:
    pd.DataFrame(data).to_excel('三算法预测文物结果.xlsx',sheet_name='data',index=False)
except:
    print("文件已打开！")

# 设置X,Y的范围
x =[]
for i in range(8):
    x.append("文物"+str(i+1))
y_1 =pre_pro[0]
y_2 =pre_pro[1]
y_3 =pre_pro[2]
# 设置图形大小
plt.rcParams['font.sans-serif'] =["KaiTi"]
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15,20), dpi=120)

plt.plot(x, y_1, label='贝叶斯分类', color='red', linestyle=':', marker='.', markersize=20)
plt.plot(x, y_2, label='神经网络', color='black', linestyle='--', marker='.', markersize=20)
plt.plot(x, y_3, label='随机森林', color='#228B22', linestyle='--', marker='.', markersize=20)
# 设置X刻度
_xtick_labels = [i for i in x]
plt.xticks(x, _xtick_labels, rotation=45,fontsize=20)
plt.yticks(fontsize=20)
# 设置X，Y轴标签
#plt.xlabel('预测结果权重',fontsize=20)
plt.ylabel('预测结果权重',fontsize=20)
plt.title('三个算法预测文物结果权重',fontsize=20)
plt.grid(alpha=0.8)
plt.legend(loc='upper left')
# 展示图形
plt.show()