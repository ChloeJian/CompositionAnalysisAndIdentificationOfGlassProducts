clc
clear all
data=xlsread('斯皮尔曼.xlsx',1,'A2:D59');
figure
% 求维度之间的相关系数
rho = corr(data, 'type','Spearman');
% 绘制热图
string_name={'纹饰','类型','颜色','表面风化'};
xvalues = string_name;
yvalues = string_name;
h = heatmap(xvalues,yvalues, rho, 'FontSize',10, 'FontName','宋体');
h.Title = 'Spearman相关系数矩阵';
colormap summer