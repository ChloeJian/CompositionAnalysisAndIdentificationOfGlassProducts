from sklearn import neural_network
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
mlp=neural_network.MLPClassifier(hidden_layer_sizes=(10), #隐藏层
                                activation='relu',  #激活函数
                                solver='adam',
                                alpha=0.0001,  #正则化项系数
                                batch_size='auto',
                                learning_rate='constant',  #学习率
                                learning_rate_init=0.001,
                                power_t=0.5,
                                max_iter=450,#迭代次数
                                tol=1e-4)
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
        elif(j in lis):
            datax[i].append(data[j][i])
x_train, x_test, y_train, y_test = train_test_split(datax, datay, test_size=0.2, random_state=10)
mlp.fit(x_train,y_train)
print("预测准确度：")
print(mlp.score(x_test,y_test))

pre=mlp.predict_proba(datax)
pre_dict={}
pre_dict['高钾概率']=[]
pre_dict['铅钡概率']=[]
pre_dict['预测结果']=mlp.predict(datax)
pre_dict['原始标签']=datay
for i in range(len(pre)):
    pre_dict['高钾概率'].append(pre[i][0])
    pre_dict['铅钡概率'].append(pre[i][1])
data=pd.DataFrame(pre_dict)#将字典转换成为数据框
pd.DataFrame(data).to_excel('神经网络结果概率.xlsx',sheet_name='data',index=0)

data=pd.DataFrame(mlp.coefs_)
data_list=mlp.coefs_[0].tolist()
cl=[]
for i in range(10):
    cl.append(mlp.coefs_[1].tolist()[i][0])
data_dict={}
data_dict['神经网络层级']=[i+1 for i in range(10)]
data_dict['二氧化硅(SiO2)']=data_list[0]
data_dict['氧化钠(Na2O)']=data_list[1]
data_dict['氧化钾(K2O)']=data_list[2]
data_dict['氧化钙(CaO)']=data_list[3]
data_dict['氧化镁(MgO)']=data_list[4]
data_dict['氧化铝(Al2O3)']=data_list[5]
data_dict['氧化铁(Fe2O3)']=data_list[6]
data_dict['氧化铜(CuO)']=data_list[7]
data_dict['氧化铅(PbO)']=data_list[8]
data_dict['氧化钡(BaO)']=data_list[9]
data_dict['五氧化二磷(P2O5)']=data_list[10]
data_dict['氧化锶(SrO)']=data_list[11]
data_dict['氧化锡(SnO2)']=data_list[12]
data_dict['二氧化硫(SO2)']=data_list[13]
data_dict['常量']=cl
data=pd.DataFrame(data_dict)#将字典转换成为数据框
pd.DataFrame(data_dict).to_excel('多层感知网络权重.xlsx',sheet_name='data',index=False)

y_score =mlp.predict_proba(x_test)
from sklearn import metrics
from pylab import *
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
name=["高钾","铅钡"]
for i in range(2):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score[:, i],pos_label=i)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr,
                 lw=2, label="多层感知网络_ROC "+name[i]+" (area = %0.2f)" % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()