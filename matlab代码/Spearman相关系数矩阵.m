clc
clear all
data=xlsread('˹Ƥ����.xlsx',1,'A2:D59');
figure
% ��ά��֮������ϵ��
rho = corr(data, 'type','Spearman');
% ������ͼ
string_name={'����','����','��ɫ','����绯'};
xvalues = string_name;
yvalues = string_name;
h = heatmap(xvalues,yvalues, rho, 'FontSize',10, 'FontName','����');
h.Title = 'Spearman���ϵ������';
colormap summer