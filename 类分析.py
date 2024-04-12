# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:28:57 2023

@author: Lijim
"""

import warnings
warnings.filterwarnings('ignore')
import os
import xlrd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri#不单调序列的等高线画图，单调序列可使用contour
import matplotlib.ticker as ticker#colorbar函数刻度数量调整
from scipy.interpolate import griddata#基于地形的3D平面图 
from sklearn.cluster import KMeans#聚类
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False #解决负数坐标显示问题
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

shu0=pd.read_excel('./cumcm2011A附件_数据.xls',sheet_name=0,header=2,index_col=0)
shu1=pd.read_excel('./cumcm2011A附件_数据.xls',sheet_name=1,header=2,index_col=0)
shu2=pd.read_excel('./cumcm2011A附件_数据.xls',sheet_name=2,header=2,index_col=0)
shu3=pd.read_csv('./a.csv',index_col=0)

class city():
    def __init__(self,shu0,shu1,shu2,shu3,figure_save_path = "figures"):
        '''
        shu0 : 数据集1
        shu1 : 数据集2
        shu2 : 数据集3
        shu3 : 自建数据集，描述重金属元素的化学性质
        x,y,z: 为取样地点分布空间位置
        grouped:基于功能区的分组
        figure_save_path: 图片保存文件夹，不必存在，后面会新建该文件夹

        '''
        data=pd.merge(shu0.iloc[:,:4],shu1,how='left',on='编号')
        z_b=data['功能区'].copy()
        z_b[z_b==1]=5.5
        z_b[z_b==2]=1.5
        z_b[z_b==3]=2.5
        z_b[z_b==4]=3.5
        z_b[z_b==5]=6
        data.loc[data['功能区']==1,'功能区']='生活区'
        data.loc[data['功能区']==2,'功能区']='工业区'
        data.loc[data['功能区']==3,'功能区']='山区'
        data.loc[data['功能区']==4,'功能区']='交通区'
        data.loc[data['功能区']==5,'功能区']='公园绿地区'
        metal=pd.merge(shu2,shu3,how='left',on='元素')
        metal[["下限",'上限']]=metal["范围"].str.split('~',expand= True)
        metal['上限']=metal['上限'].astype('float')
        metal['下限']=metal['下限'].astype('float')
        df_dict={}
        df_dict['As (μg/g)']=data.loc[data['As (μg/g)'] >metal.loc['As (μg/g)','上限']]
        df_dict['Cd (ng/g)']=data.loc[data['Cd (ng/g)'] >metal.loc['Cd (ng/g)','上限']]
        df_dict['Cr (μg/g)']=data.loc[data['Cr (μg/g)'] >metal.loc['Cr (μg/g)','上限']]
        df_dict['Cu (μg/g)']=data.loc[data['Cu (μg/g)'] >metal.loc['Cu (μg/g)','上限']]
        df_dict['Hg (ng/g)']=data.loc[data['Hg (ng/g)'] >metal.loc['Hg (ng/g)','上限']]
        df_dict['Ni (μg/g)']=data.loc[data['Ni (μg/g)'] >metal.loc['Ni (μg/g)','上限']]
        df_dict['Pb (μg/g)']=data.loc[data['Pb (μg/g)'] >metal.loc['Pb (μg/g)','上限']]
        df_dict['Zn (μg/g)']=data.loc[data['Zn (μg/g)'] >metal.loc['Zn (μg/g)','上限']]
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
            os.makedirs(figure_save_path+'./pdf/去噪后')
            os.makedirs(figure_save_path+'./png/去噪后')
            os.makedirs(figure_save_path+'./pdf/污染源')
            os.makedirs(figure_save_path+'./png/污染源')
            
         
        self.z_b=z_b#数字表示的功能区
        self.data=data
        self.x=data['x(m)']
        self.y=data['y(m)']
        self.z=data['海拔(m)']
        self.grouped = data.groupby(['功能区'])#基于功能分组
        self.living = data.loc[data['功能区']=='生活区',:]
        self.indusity = data.loc[data['功能区']=='工业区',:]
        self.mountain = data.loc[data['功能区']=='山区',:]
        self.traffic = data.loc[data['功能区']=='交通区',:]
        self.park = data.loc[data['功能区']=='公园绿地区',:]
        self.metal = metal
        self.df_dict = df_dict
        self.path_png = figure_save_path+'./png'
        self.path_pdf = figure_save_path+'./pdf'
        
        
    
    def landfrom(self):
        '''
        取样地点类型与海拔的三维曲面图
        '''
        
        xi=np.linspace(min(self.x),max(self.x))
        yi=np.linspace(min(self.y),max(self.y))
        xi,yi=np.meshgrid(xi,yi)
        zi=griddata(self.data.iloc[:,0:2],self.z,(xi,yi),method='cubic')
        #作图
        fig=plt.figure(figsize=(10,7))
        ax = fig.add_axes(Axes3D(fig)) #,'BuPu'
        surf=ax.plot_surface(xi,yi,zi,cmap='copper',alpha=0.6,linewidth=0,antialiased=False)
        fig.colorbar(surf)
        ax.scatter(self.living['x(m)'] , self.living['y(m)'] ,  self.living['海拔(m)'],marker='^',label='生活区')
        ax.scatter(self.indusity['x(m)'],self.indusity['y(m)'], self.indusity['海拔(m)'],marker='*',label='工业区')
        ax.scatter(self.mountain['x(m)'],self.mountain['y(m)'], self.mountain['海拔(m)'],marker='+',label='山区')
        ax.scatter(self.traffic['x(m)'] ,self.traffic['y(m)'] , self.traffic['海拔(m)'],marker='s',label='交通区')
        ax.scatter(self.park['x(m)']  ,  self.park['y(m)']   ,  self.park['海拔(m)'],label='公园绿地区')
        ax.legend(loc=0, fontsize=14)
        # 设置坐标轴标题和刻度
        ax.set(xlabel='X',
               ylabel='Y',
               zlabel='Z',
               #title='取样地点类型与海拔的三维曲面图'
               )
        # 调整视角
        #ax.view_init(elev=20,    # 仰角
        #             azim=315    # 方位角315,225,135
        #             )
        plt.savefig(os.path.join(self.path_pdf,'取样地点类型与海拔的三维曲面图.pdf'), dpi=600)
        plt.savefig(os.path.join(self.path_png,'取样地点类型与海拔的三维曲面图.png'), dpi=600)
        plt.show()
    
    def altitudinal(self):
        '''
        取样地点类型与海拔的等高线图
        '''
        plt.subplot(1,1,1)
        plt.tricontour( self.x, self.y, self.z, linewidths=0.5, colors='k')  
        plt.tricontourf(self.x, self.y, self.z)  # 等高线的数目紧接在坐标后面
        plt.colorbar()
        #plt.scatter(self.x,self.y,label='location',s=self.z*0.3)
        plt.scatter(self.living['x(m)'] , self.living['y(m)'],marker='^',label='生活区')
        plt.scatter(self.indusity['x(m)'],self.indusity['y(m)'],marker='*',label='工业区')
        plt.scatter(self.mountain['x(m)'],self.mountain['y(m)'],marker='+',label='山区')
        plt.scatter(self.traffic['x(m)'] ,self.traffic['y(m)'],marker='s',label='交通区')
        plt.scatter(self.park['x(m)']  ,  self.park['y(m)'],label='公园绿地区')
        plt.legend(loc=0, fontsize=14)
        plt.xlabel ("X")
        plt.ylabel ("Y")

        #plt.title('取样地点类型与海拔的等高线图')
        plt.savefig(os.path.join(self.path_pdf,'取样地点类型与海拔的等高线图.pdf'), dpi=600)
        plt.savefig(os.path.join(self.path_png,'取样地点类型与海拔的等高线图.png'), dpi=600)
        plt.show()
        
    def ribbon(self):#主要目的不是调用该函数，而是规范作图代码的书写
        '''
        功能区分布情况
        '''        
        plt.subplot(1,1,1)
        plt.tricontour( self.x, self.y, self.z_b, 4, linewidths=0.5, colors='k')  
        plt.tricontourf(self.x, self.y, self.z_b, 4)  
        cbar=plt.colorbar(ticks=[1.5,2.5,3.5,4.5,5.5])
        cbar.ax.set_yticklabels(['工业区','山区', '交通区','公园绿地区','生活区'])
        plt.scatter(self.x,self.y,label='location',c=self.z_b)
        plt.legend(loc=0, fontsize=14)
        plt.xlabel ("X")
        plt.ylabel ("Y")

        #plt.title('功能区分布图')
        plt.savefig(os.path.join(self.path_pdf,'功能区分布图.pdf'), dpi=600)
        plt.savefig(os.path.join(self.path_png,'功能区分布图.png'), dpi=600)
        plt.show()

    
    def scatter(self,name='Zn (μg/g)',multiple=0.8,df=None):
        '''
        重金属元素分布与海拔与功能区的关系

        '''
        if isinstance(df, pd.DataFrame):
            living  = df.loc[df['功能区']=='生活区',:]
            indusity= df.loc[df['功能区']=='工业区',:]
            mountain= df.loc[df['功能区']=='山区',:]
            traffic = df.loc[df['功能区']=='交通区',:]
            park    = df.loc[df['功能区']=='公园绿地区',:]
        else:
            living  = self.living
            indusity= self.indusity
            mountain= self.mountain
            traffic = self.traffic
            park    = self.park
            

        plt.subplot(1,1,1)
        plt.tricontour( self.x, self.y, self.z,linewidths=0.5,colors='k')  # 15:等高线的数目
        plt.tricontourf(self.x, self.y, self.z)   # 等高线的数目紧接在坐标后面
        plt.colorbar()
        plt.scatter(living['x(m)'] , living['y(m)']  ,s=living[name]*multiple,label='生活区')
        plt.scatter(indusity['x(m)'],indusity['y(m)'],s=indusity[name]*multiple,label='工业区')
        plt.scatter(mountain['x(m)'],mountain['y(m)'],s=mountain[name]*multiple,label='山区')
        plt.scatter(traffic['x(m)'] ,traffic['y(m)'] ,s=traffic[name]*multiple,label='交通区')
        plt.scatter(park['x(m)']  ,  park['y(m)']  ,  s=park[name]*multiple,label='公园绿地区')
        plt.legend(loc=0, fontsize=14)
        plt.xlabel ("X")
        plt.ylabel ("Y")
        if isinstance(df, pd.DataFrame):
            #plt.title('去除白噪声后的不同功能区下的{}元素分布图'.format(name))
            #plt.savefig(os.path.join(self.path ,'./pdf/去噪后/','quzao_'+name[:2]+'.pdf'),dpi=600)#分别命名图片
            #plt.savefig(os.path.join(self.path ,'./png/去噪后/'+'quzao_'+name[:2]+'.png'),dpi=600)#分别命名图片
            plt.savefig(os.path.join(self.path_pdf,'./去噪后/'+name[:2]+'_quzao.pdf'), dpi=600)
            plt.savefig(os.path.join(self.path_png,'./去噪后/'+name[:2]+'_quzao.png'), dpi=600)
        else:
            #plt.title('不同功能区下的{}元素分布图'.format(name))
            plt.savefig(os.path.join(self.path_pdf ,name[:2]+'.pdf'),dpi=600)#分别命名图片
            plt.savefig(os.path.join(self.path_png ,name[:2]+'.png'),dpi=600)#分别命名图片
        plt.show()
    
    def group(self):
        '''
        分组变量的应用
        '''
        for name, group in self.grouped:
            #ax.plot(group['timepoint'], group['signal'], label=name)
            1
    
    def group_stats(self):
        '''
        分组后的计算统计
        grouped.agg()中加入自定义函数时，似乎只能为单变量函数，故而选择匿名函数
        '''
                    
        grouped = self.data.groupby(['功能区'])#基于功能分组
        data_df = grouped.agg({ 'As (μg/g)':[np.max, np.min, np.mean, np.median,np.std,lambda x:x.loc[x>self.metal.loc['As (μg/g)','上限']].size/x.size],
                               'Cd (ng/g)':[np.max, np.min, np.mean, np.median,np.std,lambda x:x.loc[x>self.metal.loc['Cd (ng/g)','上限']].size/x.size],
                               'Cr (μg/g)':[np.max, np.min, np.mean, np.median,np.std,lambda x:x.loc[x>self.metal.loc['Cr (μg/g)','上限']].size/x.size],
                               'Cu (μg/g)':[np.max, np.min, np.mean, np.median,np.std,lambda x:x.loc[x>self.metal.loc['Cu (μg/g)','上限']].size/x.size],
                               'Hg (ng/g)':[np.max, np.min, np.mean, np.median,np.std,lambda x:x.loc[x>self.metal.loc['Hg (ng/g)','上限']].size/x.size],
                               'Ni (μg/g)':[np.max, np.min, np.mean, np.median,np.std,lambda x:x.loc[x>self.metal.loc['Ni (μg/g)','上限']].size/x.size],
                               'Pb (μg/g)':[np.max, np.min, np.mean, np.median,np.std,lambda x:x.loc[x>self.metal.loc['Pb (μg/g)','上限']].size/x.size],
                               'Zn (μg/g)':[np.max, np.min, np.mean, np.median,np.std,lambda x:x.loc[x>self.metal.loc['Zn (μg/g)','上限']].size/x.size]})

        data_df.columns = ['_'.join(column) for column in data_df.columns.values]#修改列名
        data_df['数量']=self.data['功能区'].value_counts()
        return data_df.round(2)#数据保留两位小数
    
    def corr(self,df_corr=None,name=None):
        '''
        相关性分析
        
        df_corr:dataframes文件
        name:图片命名

        '''
        if isinstance(df_corr, pd.DataFrame):
            df=df_corr.loc[:,df_corr.columns !='功能区']  
            plt.figure(figsize=(18,15), dpi= 80)
            sns.heatmap(df.corr(), xticklabels=df.corr().columns,
                        yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
            #设置
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            if name==None:
                plt.title('去噪后_相关性图表', fontsize=22)
                plt.savefig(os.path.join(self.path_pdf ,'./去噪后/'+'去噪后_特征相关性分析表'+'.pdf'),dpi=600)#分别命名图片
                plt.savefig(os.path.join(self.path_png ,'./去噪后/'+'去噪后_特征相关性分析表'+'.png'),dpi=600)#分别命名图片
                plt.show()
            else:
                plt.title(str(name)+'_相关性图表', fontsize=22)
                plt.savefig(os.path.join(self.path_pdf ,'./'+str(name)+'_特征相关性分析表'+'.pdf'),dpi=600)#分别命名图片
                plt.savefig(os.path.join(self.path_png ,'./'+str(name)+'_特征相关性分析表'+'.png'),dpi=600)#分别命名图片
                plt.show()
            
        else:
            df=self.data.loc[:,self.data.columns !='功能区']
            plt.figure(figsize=(18,15), dpi= 80)
            sns.heatmap(df.corr(), xticklabels=df.corr().columns,
                        yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
            #设置
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('相关性图表', fontsize=22)
            plt.savefig(os.path.join(self.path_pdf ,'特征相关性分析表.pdf'),dpi=600)#分别命名图片
            plt.savefig(os.path.join(self.path_png ,'特征相关性分析表.png'),dpi=600)#分别命名图片
            plt.show()
    
        
        
    
    def corr_element(self,df=None):
        if isinstance(df, pd.DataFrame):
            plt.plot(self.metal['相对原子质量'],np.abs(df.loc[:,df.columns !='功能区'].corr()['海拔(m)'][-8:]), '*', alpha=0.5, linewidth=1, label='acc')
            #plt.title('去除白噪声后元素分布与海拔的关系')
            plt.savefig(os.path.join(self.path_pdf ,'./去噪后/','quzao_元素_海拔.pdf'),dpi=600)#分别命名图片
            plt.savefig(os.path.join(self.path_png ,'./去噪后/'+'quzao_元素_海拔.png'),dpi=600)#分别命名图片
            plt.show()
            element_corr=np.corrcoef(p.metal['相对原子质量'],np.abs(df.loc[:,df.columns !='功能区'].corr()['海拔(m)'][-8:]), rowvar=True, bias=np._NoValue, ddof=np._NoValue)[0,1]
            print('去除白噪声后元素分布高度与相对原子质量的相关系数为{:.3f}'.format(element_corr))
            
        else:
            plt.plot(self.metal['相对原子质量'],np.abs(self.data.loc[:,p.data.columns !='功能区'].corr()['海拔(m)'][-8:]), '*', alpha=0.5, linewidth=1, label='acc')
            #plt.title('元素分布与海拔的关系')
            plt.savefig(os.path.join(self.path_pdf ,'元素_海拔'+'.pdf'),dpi=600)#分别命名图片
            plt.savefig(os.path.join(self.path_png ,'元素_海拔'+'.png'),dpi=600)#分别命名图片
            plt.show()
            element_corr=np.corrcoef(p.metal['相对原子质量'],np.abs(self.data.loc[:,self.data.columns !='功能区'].corr()['海拔(m)'][-8:]), rowvar=True, bias=np._NoValue, ddof=np._NoValue)[0,1]
            print('元素分布高度与相对原子质量的相关系数为{:.3f}'.format(element_corr))
            
    def white_noise(self,n0=0):
        '''
        去除白噪声
        n0:表示背景噪声。平均背景噪声n0=0；最小背景噪声n0=3，最大背景噪声n0=4
        '''
        df=self.data.copy()
        df['As (μg/g)']-=self.metal.iloc[0,n0]
        df['Cd (ng/g)']-=self.metal.iloc[1,n0]
        df['Cr (μg/g)']-=self.metal.iloc[2,n0]
        df['Cu (μg/g)']-=self.metal.iloc[3,n0]
        df['Hg (ng/g)']-=self.metal.iloc[4,n0]
        df['Ni (μg/g)']-=self.metal.iloc[5,n0]
        df['Pb (μg/g)']-=self.metal.iloc[6,n0]
        df['Zn (μg/g)']-=self.metal.iloc[7,n0]
        
        df.loc[df['As (μg/g)']<0,'As (μg/g)']=0
        df.loc[df['Cd (ng/g)']<0,'Cd (ng/g)']=0
        df.loc[df['Cr (μg/g)']<0,'Cr (μg/g)']=0
        df.loc[df['Cu (μg/g)']<0,'Cu (μg/g)']=0
        df.loc[df['Hg (ng/g)']<0,'Hg (ng/g)']=0
        df.loc[df['Ni (μg/g)']<0,'Ni (μg/g)']=0
        df.loc[df['Pb (μg/g)']<0,'Pb (μg/g)']=0
        df.loc[df['Zn (μg/g)']<0,'Zn (μg/g)']=0
        return df
    
    def less_white_noise(self,n0=3):
        '''
        重金属含量低于自然水平 
        n0:表示背景噪声。平均背景噪声n0=0；最小背景噪声n0=3，最大背景噪声n0=4
        '''
        df=self.data.copy()
        df.loc[df['As (μg/g)']-self.metal.iloc[0,n0]>0,'As (μg/g)']=0
        df.loc[df['Cd (ng/g)']-self.metal.iloc[1,n0]>0,'Cd (ng/g)']=0
        df.loc[df['Cr (μg/g)']-self.metal.iloc[2,n0]>0,'Cr (μg/g)']=0
        df.loc[df['Cu (μg/g)']-self.metal.iloc[3,n0]>0,'Cu (μg/g)']=0
        df.loc[df['Hg (ng/g)']-self.metal.iloc[4,n0]>0,'Hg (ng/g)']=0
        df.loc[df['Ni (μg/g)']-self.metal.iloc[5,n0]>0,'Ni (μg/g)']=0
        df.loc[df['Pb (μg/g)']-self.metal.iloc[6,n0]>0,'Pb (μg/g)']=0
        df.loc[df['Zn (μg/g)']-self.metal.iloc[7,n0]>0,'Zn (μg/g)']=0
        return df
    
    def screen(self,r=None):
        '''
        筛选符合要求的点，作为聚类的集合
        r:表示筛选的依据，每一个数值对应于每一种重金属的筛选标准

        '''
        if isinstance(r, list):
            1
        else:
            r=self.metal['上限']
        df_dict={}
        df_dict['As (μg/g)']=self.data.loc[self.data['As (μg/g)'] >r[0]]
        df_dict['Cd (ng/g)']=self.data.loc[self.data['Cd (ng/g)'] >r[1]]
        df_dict['Cr (μg/g)']=self.data.loc[self.data['Cr (μg/g)'] >r[2]]
        df_dict['Cu (μg/g)']=self.data.loc[self.data['Cu (μg/g)'] >r[3]]
        df_dict['Hg (ng/g)']=self.data.loc[self.data['Hg (ng/g)'] >r[4]]
        df_dict['Ni (μg/g)']=self.data.loc[self.data['Ni (μg/g)'] >r[5]]
        df_dict['Pb (μg/g)']=self.data.loc[self.data['Pb (μg/g)'] >r[6]]
        df_dict['Zn (μg/g)']=self.data.loc[self.data['Zn (μg/g)'] >r[7]]
        
        return df_dict

    def K_means(self,df=None,n=3,name='As (μg/g)'):
        '''
        聚类函数，并给出聚类中心与可视化图像
        n:表示聚类中心的数量
        df:数据文档要求为dataframes，且包含地点污染物浓度信息
        name:表示对哪一种重金属进行聚类

        '''
        if isinstance(df, pd.DataFrame):
            X=df[['x(m)','y(m)']].values
        else:
            X=self.data[['x(m)','y(m)']].values
            df=self.data
        # 设置为三个聚类中心
        Kmeans = KMeans(n_clusters=n)
        # 训练模型
        Kmeans.fit(X)
        #获取聚类中心
        C=Kmeans.cluster_centers_
        #获取类别
        df['类别']=Kmeans.labels_
        print(str(name)+'元素污染源预测位置\n{}\n'.format(C))
        
        plt.subplot(1,1,1)
        plt.tricontour( df['x(m)'],df['y(m)'], df[name], linewidths=0.5,colors='k')  # 15:等高线的数目
        plt.tricontourf(df['x(m)'],df['y(m)'], df[name],alpha=0.3)   # 等高线的数目紧接在坐标后面
        plt.colorbar()
        plt.scatter(df['x(m)'],df['y(m)'],c=df['类别'],s=df[name])
        plt.scatter(C[:,0],C[:,1],marker='*',c='r',s=500,label='污染源')
        #cbar=plt.colorbar(ticks=[1.5,2.5,3.5,4.5,5.5])
        #cbar.ax.set_yticklabels(['工业区','山区', '交通区','公园绿地区','生活区'])
        plt.legend(loc=0, fontsize=14)
        #plt.title(str(name)+'元素浓度分布与污染源预测')
        plt.xlabel ("X(m)")
        plt.ylabel ("Y(m)")
        
        plt.savefig(os.path.join(self.path_pdf ,'./污染源/'+name[:2])+'.pdf',dpi=600)#分别命名图片
        plt.savefig(os.path.join(self.path_png ,'./污染源/'+name[:2])+'.png',dpi=600)#分别命名图片
        plt.show()
   
    
    
    


if __name__ == "__main__":
    p=city(shu0,shu1,shu2,shu3)
    
    #取样地点地形与取样地点类型
    p.landfrom()
    p.altitudinal()

    #重金属元素的分布情况（地形分布，含量分布，功能区所属）
    ####定性分析
    p.scatter(name='As (μg/g)' , multiple= 15)
    
    p.scatter(name='Cd (ng/g)' , multiple= 0.5)
    p.scatter(name='Cr (μg/g)' , multiple= 2)
    p.scatter(name='Cu (μg/g)' , multiple= 1)
    p.scatter(name='Hg (ng/g)' , multiple= 0.25)
    p.scatter(name='Ni (μg/g)' , multiple= 10)
    p.scatter(name='Pb (μg/g)' , multiple= 1)
    p.scatter(name='Zn (μg/g)' , multiple= 0.5)
    ####定量分析
    df=p.group_stats()
    df_mean=df.iloc[: , range(2,len(df.columns),int(len(df.columns)/8))]
    s = df.style.highlight_max(axis=True,
                               props='cellcolor:{red}; bfseries: ;')
    s.to_html('./output/df.html')
    s = df_mean.style.highlight_max(axis=True,
                               props='cellcolor:{red}; bfseries: ;')
    s.to_html('./output/df_mean.html')
    
    #数据相关性分析
    p.corr()#化学性质
    p.corr_element()#海拔与相对原子质量
    #数据相关性分析
    p.corr(df_corr=p.living,name='生活区')#化学性质
    p.corr(df_corr=p.indusity,name='工业区')
    p.corr(df_corr=p.mountain,name='山区')
    p.corr(df_corr=p.traffic,name='交通区')
    p.corr(df_corr=p.park,name='公园绿地区')
    
    
    
    
    ######################################################################
    #去除噪声
    data_white=p.white_noise(n0=-1)
    
    p.scatter(name='As (μg/g)' , multiple= 15  ,df=data_white)
    p.scatter(name='As (μg/g)' , multiple= 15  ,df=data_white)
    p.scatter(name='Cd (ng/g)' , multiple= 0.5 ,df=data_white)
    p.scatter(name='Cr (μg/g)' , multiple= 2   ,df=data_white)
    p.scatter(name='Cu (μg/g)' , multiple= 1   ,df=data_white)
    p.scatter(name='Hg (ng/g)' , multiple= 0.25,df=data_white)
    p.scatter(name='Ni (μg/g)' , multiple= 10  ,df=data_white)
    p.scatter(name='Pb (μg/g)' , multiple= 1   ,df=data_white)
    p.scatter(name='Zn (μg/g)' , multiple= 0.5 ,df=data_white)
    #去除噪声后的相关性分析
    p.corr(df_corr=data_white)
    p.corr_element(df=data_white)
   
    
    ##污染源的寻找，聚类方法
    df_dict=p.screen(r=[7,600,100,160,100,27,130,350])#取点
    n_0=[5,5,3,4,7,5,6,5]#估测的可能的污染点数量
    i=-1
    for key,value in df_dict.items():
        i+=1
        p.K_means(df=value,n=n_0[i],name=key)#聚类取中心，找污染点






