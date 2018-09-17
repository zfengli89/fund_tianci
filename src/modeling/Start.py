# -*- coding:utf-8 -*-
# ********************************用xgboost进行预测**************************
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import  train_test_split
from sklearn import metrics
import graphviz

# 读取数据
df1 = pd.read_csv('/home/260216/PycharmProjects/tianci/data/Purchase&Redemption/user_profile_table.csv')
df2 = pd.read_csv('/home/260216/PycharmProjects/tianci/data/Purchase&Redemption/user_balance_table.csv')
df3 = pd.read_csv('/home/260216/PycharmProjects/tianci/data/Purchase&Redemption/mfd_day_share_interest.csv')
df4 = pd.read_csv('/home/260216/PycharmProjects/tianci/data/Purchase&Redemption/mfd_bank_shibor.csv')

# 连表
a = pd.merge(df1,df2,on='user_id')
b = pd.merge(df3,df4,on='mfd_date',how='outer')
c = pd.merge(a,b,left_on='report_date',right_on='mfd_date')
d = c.sort_values(by=['user_id','report_date'])
# d.to_csv('/home/260208/桌面/资金流入流出预测/Purchase&Redemption Data/Purchase&Redemption Data/all.csv')

#数据集
# l1 = a[a['report_date']>=20140501]
# data = l1[l1['report_date'] <= 20140731]
#
# l2 = a[a['report_date']>=20140801]
# label = l2[l2['report_date'] <= 20140831]

data = np.array(d.loc[:,['user_id','sex','city','report_date','tBalance','yBalance','direct_purchase_amt','purchase_bal_amt',
                         'purchase_bank_amt','total_redeem_amt','consume_amt','transfer_amt','tftobal_amt','tftocard_amt',
                         'share_amt','category1','category2','category3','category4','mfd_date','mfd_daily_yield',
                         'mfd_7daily_yield','Interest_O_N','Interest_1_W','Interest_2_W','Interest_1_M','Interest_3_M',
                         'Interest_6_M','Interest_9_M','Interest_1_Y']])
label = np.array(d['total_purchase_amt'])

# 训练集与测试集
train_x,test_x,train_y,test_y=train_test_split(data,label,random_state=0)  #随机划分训练集及测试集
# print train_x
# print test_x
# print train_y
# print test_y
# ************Xgboost建模************
# 模型初始化

dtrain = xgb.DMatrix(train_x,label=train_y)  #加载数据、缓存文件
dtest = xgb.DMatrix(test_x)
# print dtrain
# print dtest
params={'booster':'gbtree',
    'objective': 'reg:linear',
    'eval_metric': 'auc',
    'max_depth':4,   #树的深度[1:]
    'lambda':10,     #L2正则项权重
    'subsample':0.75, #采样训练数据，设置为0.5，随即选择一般的数据实例(0:1]
    'colsample_bytree':0.75,    #根据特征个数判断,采样比率(0:1]
    'min_child_weight':2,       #节点的最少特征数
    'eta': 0.025,
    'seed':0,
    'nthread':8,    #cpu线程数，根据自己U的个数适当调整
    'silent':1}

watchlish = [(dtrain,'train')]
# print watchlish
# 建模与预测
num_trees = 45
bst = xgb.train(params,dtrain,num_trees,evals=watchlish, early_stopping_rounds=50, verbose_eval=True)
ypred = bst.predict(dtest)
print ypred



# y_pred = (ypred >= 0.5)*1
#
# print 'AUC:%.4f'%metrics.roc_auc_score(test_y,ypred)
# print 'AUC:%.4f'%metrics.accuracy_score(test_y,y_pred)
# print 'Recall:%.4f'%metrics.recall_score(test_y,y_pred)
# print 'F1-score:%.4f'%metrics.f1_score(test_y,y_pred)
# print 'Precesion:%.4f'%metrics.precision_score(test_y,y_pred)
# print  metrics.confusion_matrix(test_y,y_pred)
#
# # *******************可视化输出*******************
# #对于预测的输出有三种方式：bst.predict、pred_leaf:bool、pred_contribs:bool
#
# ypred = bst.predict(dtest)   #默认的输出，就是得分，转换为实际得分f(x) = 1/(1+exp(-x))
# # print ypred
#
# ypred_leaf = bst.predict(dtest, pred_leaf=True) #所属的叶子节点
# # print ypred_leaf
#
# #看每棵树以及相应的叶子节点的分值，两种方法：可视化树或者直接输出模型
#
# print xgb.to_graphviz(bst, num_trees=0)  #可视化第一棵树的生成情况
# bst.dump_model("/home/260208/桌面/model.txt")  #输出模型的迭代工程
#
#
if __name__ == '__main__':
    print "Ok"
