import pandas as pd
#spss逻辑回归系数
coef=[-0.333,-0.943,-0.760,-0.541,-1.679,-0.266,-0.516,-0.068,-0.127,33.512]
lis=['二氧化硅(SiO2)',
     '氧化钠(Na2O)',
     '氧化钾(K2O)',
     '氧化铝(Al2O3)',
     '氧化铁(Fe2O3)',
     '氧化铅(PbO)',
     '氧化钡(BaO)',
     '五氧化二磷(P2O5)',
     '二氧化硫(SO2)',
     ]
def Logistic(lis1):
    index=[0,1,2,5,6,8,9,10,13]
    y=0
    for i in range(len(index)):
        y+=lis1[index[i]]*coef[i]
    y+=coef[len(coef)-1]
    return y
lis2=['二氧化硅(SiO2)',
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
     '类型数字化']
Forms=pd.DataFrame(pd.read_excel("datafenghua.xlsx",sheet_name=0))
data=Forms.to_dict('dict')
datax=[[] for j in range(len(Forms))]
datay=[]
for i in range(len(Forms)):
    for j in data:
        if(j=='风化数字化'):
            datay.append(data['风化数字化'][i])
        elif(j in lis2):
            datax[i].append(data[j][i])
y=[]
for i in range(len(datax)):
    if(Logistic(datax[i])<0.5):
        y.append(0)
    else:
        y.append(1)
count=0
for i in range(len(y)):
    if(y[i]==datay[i]):
        count+=1
print("逻辑回归预测正确率：")
print(count/len(y))

#############风化数据预测############################
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
     '类型数字化']
Forms=pd.DataFrame(pd.read_excel("风化.xlsx",sheet_name=0))
data=Forms.to_dict('dict')
datafh=[[] for j in range(len(Forms))]
for i in range(len(Forms)):
    for j in data:
        if(j in lis):
            datafh[i].append(data[j][i])
print(datafh)
def qjian(lis):
    flag2 = 999
    index = [0, 1, 2, 5, 6, 8, 9, 10, 13]
    lis2 = [lis[i] for i in range(len(lis))]
    n = 600
    lis1 = [lis[i] for i in range(len(lis))]
    while (n > 0):
        n-=1
        for i in range(len(index)):
            lis1[index[i]] -= lis1[index[i]]*coef[i]/10 #按系数递增或递减
        lis3 = []
        for j in range(len(lis)-1):
            lis3.append(lis1[j]/(sum(lis1)-lis1[len(lis1)-1])*100)
        lis3.append(lis1[len(lis1)-1])
        flag1=Logistic(lis3)
        if(flag1 < 0.05):
            lis2=lis3
            break
    return lis2
print(data)
datax={}
datax['文物采样点']=data['文物采样点']
for i in range(len(lis)-1):
    datax[lis[i]]={}
datax['逻辑回归结果值']={}
for i in range(len(datafh)):
    print("第%d次\n"%(i+1))
    print(Logistic(datafh[i]))
    lis1=qjian(datafh[i])
    for j in range(len(lis1)-1):
        datax[lis[j]][i]=lis1[j]
    datax['逻辑回归结果值'][i]=Logistic(lis1)
    print(lis1)
    print(Logistic(lis1))
data = pd.DataFrame(datax)  # 将字典转换成为数据框
pd.DataFrame(datax).to_excel('逻辑回归预测风化前化学成分.xlsx', sheet_name='data', index=False)