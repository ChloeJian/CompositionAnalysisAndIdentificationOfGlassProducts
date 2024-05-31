import numpy as np
import pandas as pd
import math
def is_number(s):
    try:
        float(s)
        return True
    except:
        return False
#表单一
Forms_one=pd.DataFrame(pd.read_excel("附件.xlsx",sheet_name=0))
#表单二
Forms_two=pd.DataFrame(pd.read_excel("附件.xlsx",sheet_name=1))
Glass_type=[]
Artifact={}
Artifact['文物编号']= {}
Artifact['类型']= {}
Artifact['表面风化']={}
for i in range(len(Forms_one)):
    Artifact['文物编号'][i]=Forms_one.loc[i,'文物编号']
    Artifact['类型'][i]=Forms_one.loc[i,'类型']
    Artifact['表面风化'][i] = Forms_one.loc[i, '表面风化']
chemical_composition=Forms_two.to_dict('dict')

chemical_biaohao=[]
for i in chemical_composition:
    if(i !='文物采样点'):
        chemical_biaohao.append(i)

chemical_composition['文物编号']={}
chemical_composition['类型']={}
chemical_composition['表面风化']={}
for i in range(len(Forms_two)):
    muber=int(chemical_composition['文物采样点'][i][:2])
    chemical_composition['文物编号'][i]=Artifact['文物编号'][muber-1]
    chemical_composition['类型'][i] = Artifact['类型'][muber - 1]
    chemical_composition['表面风化'][i] = Artifact['表面风化'][muber - 1]
for i in range(len(Forms_two)):
    count_sum=0
    nan_number=0
    nan_index=[]
    for s in chemical_biaohao:
        if(math.isnan(Forms_two.loc[i,s])):
            nan_number+=1
            nan_index.append(s)
        else:
            count_sum+=Forms_two.loc[i,s]
    if(count_sum<85 or count_sum>105):
        for s in chemical_composition:
            print(s)
            chemical_composition[s].pop(i)
    else:
        count_sum=100-count_sum
        for s in nan_index:
            chemical_composition[s][i]=count_sum/nan_number
chemical_composition['类型数字化']={}
for i in chemical_composition['类型']:
    if(chemical_composition['类型'][i]=='高钾'):
        chemical_composition['类型数字化'][i]=0
    elif(chemical_composition['类型'][i]=='铅钡'):
        chemical_composition['类型数字化'][i] =1
data=pd.DataFrame(chemical_composition)#将字典转换成为数据框
pd.DataFrame(data).to_excel('datafenghua0.xlsx',sheet_name='data',index=False)