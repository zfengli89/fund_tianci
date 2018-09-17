#coding=utf-8
#function: regression modeling
#author: 260216

from matplotlib.font_manager import FontProperties
import src.dataFetch.ReadData as rd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# 中文字体格式配置
myfont = FontProperties(fname='/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

## 表头
# 日期
key_date = "report_date"
key_mfd_date = "mfd_date"
# 今日总购买量制作
key_total_purchase_amt = "total_purchase_amt"
# 今日总赎回量
key_total_redeem_amt = "total_redeem_amt"
# 用户id
uid = "user_id"
# csv保存路径
csv_path = "/home/260208/PycharmProjects/anaconda/tianci/src/out/result.csv"


# 数据源
def dataSource():
    print "data read"
    balanceDF = rd.get_balance()
    shareDF = rd.get_shareInterest()
    shiborDF = rd.get_shibor()
    return balanceDF, shareDF, shiborDF

# 数据预处理
def preprocess(balanceDF, shareDF, shiborDF):
    print "data preproces"
    balanceDF.fillna(0, inplace=True)
    balanceDF.drop([uid], axis=1, inplace=True)
    sumbyDateDF = balanceDF.groupby(key_date).sum()
    # index
    dateIndex = sumbyDateDF.index
    # make wide table
    sumbyDateDF[key_date] = dateIndex
    # merge
    sumbyDateDF = pd.merge(sumbyDateDF, shareDF, left_on=key_date, right_on=key_mfd_date, how='left')\
        .fillna(0).drop([key_mfd_date], axis=1)
    sumbyDateDF = pd.merge(sumbyDateDF, shiborDF, left_on=key_date, right_on=key_mfd_date, how='left')\
        .fillna(0).drop([key_mfd_date, key_date], axis=1)
    # 还原时间index
    sumbyDateDF.index = dateIndex
    return sumbyDateDF

# 生成时序样本对
def generalSample(df, featureCycle, labelCycle, labelKey):
    '''
     funciton： 输入时间序列Dataframe, 构建时序样本对
     input： df -- 时间序列Dataframe
             featureCycle -- 特征时间区间阈值（天）
             labelCycle   -- 标签时间区间阈值（天）
             labelKey     -- 标签表头
     output: 样本对  -- 前featureCycle天作为特征  后labelCycle天作为结果
    '''
    lenDF = len(df)
    fristTime = datetime.strptime(min(df.index), '%Y%m%d')
    # 总样本数
    sampleNum = lenDF-featureCycle-labelCycle
    resFeature = None
    resLabel = None

    # 遍历时间序列,构造样本对
    for cnt in range(sampleNum):
        # 时间段节点
        # 特征时间区间
        begTime = fristTime
        midTime = begTime + timedelta(days=(featureCycle-1))
        # 标签时间区间
        midTime2 = midTime + timedelta(days=1)
        endTime = midTime + timedelta(days=labelCycle)
        # time2str
        strBegTime = begTime.strftime('%Y%m%d')
        strMidTime = midTime.strftime('%Y%m%d')
        strMidTime2 = midTime2.strftime('%Y%m%d')
        strEndTime = endTime.strftime('%Y%m%d')
        # 提取时序样本区间
        featureMat = df[strBegTime:strMidTime].as_matrix()
        labelMat = df[[labelKey]][strMidTime2:strEndTime].as_matrix()
        hstackFeature = np.hstack(featureMat)
        hstackLabel = np.hstack(labelMat)
        # 行串联矩阵
        if resFeature is None or resLabel is None:
            resFeature = hstackFeature
            resLabel = hstackLabel
        else:
            resFeature = np.row_stack((resFeature, hstackFeature))
            resLabel = np.row_stack((resLabel, hstackLabel))
        # t++
        fristTime = fristTime + timedelta(days=1)

    #制作待预测的九月份特征
    predictFeature = df.iloc[(lenDF-featureCycle):lenDF].as_matrix()
    predictFeature = np.hstack(predictFeature)
    return resFeature, resLabel, predictFeature

# 建模参数调优
def Algthrim_Tune(X, Y, n_components):
    # 特征归一化
    X_scaled = preprocessing.scale(X)
    # pca降维
    pcaModel = PCA(n_components=n_components)
    lowDimX = pcaModel.fit(X_scaled).transform(X_scaled)
    # 再次归一化
    lowDimX_scaled = preprocessing.scale(lowDimX)
    # 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(lowDimX_scaled, Y, test_size = 0.2, random_state = 0)
    # print "训练集样本shape：%d, %d"%(X_train.shape[0], X_train.shape[1])
    # print "测试集样本shape：%d, %d"%(X_test.shape[0], X_test.shape[1])
    # 模型训练
    reg_model = linear_model.LinearRegression()
    reg_model.fit(X_train, Y_train)
    # 模型预测
    y_predict = reg_model.predict(X_test)
    # 模型评估
    score = metric_plot(Y_test, y_predict, plotFLag=False)
    return score

# 预测结果
def Algthrim_Predict(X, Y, n_components, predictFeature):
    X_rowStack = np.row_stack((X, predictFeature))
    # 特征归一化
    X_scaled = preprocessing.scale(X_rowStack)
    # pca降维
    pcaModel = PCA(n_components=n_components)
    lowDimX = pcaModel.fit(X_scaled).transform(X_scaled)
    # 再次归一化
    lowDimX_scaled = preprocessing.scale(lowDimX)
    # 划分数据集
    X_train = lowDimX_scaled[0:-1, :]
    Y_train = Y
    X_predict = lowDimX_scaled[-1, :]
    # 模型训练
    reg_model = linear_model.LinearRegression()
    reg_model.fit(X_train, Y_train)
    # 模型预测
    Y_predict = reg_model.predict(X_predict)
    return Y_predict

# 得分评估
def metric_plot(actual, predict, plotFLag=False):
    row, col = actual.shape
    scoreMat = None
    df = pd.DataFrame()
    for day in range(col):
        col_actual = actual[:, day]
        col_predict = predict[:, day]
        # 计算得分
        zipRes = zip(col_actual, col_predict)
        indexScore = map(ScoreFormula, zipRes)
        if scoreMat is None:
            scoreMat = indexScore
        else:
            scoreMat = np.column_stack((scoreMat, indexScore))
        # 按天图形输出,方便人为观测结果大致情况
        if plotFLag:
            actualKey = 'actualValue'
            predictKey = 'predictValue'
            df[actualKey] = col_actual
            df[predictKey] = col_predict
            df.plot()
            plt.title(u'预测第%d天结果'%(day+1))
            plt.show()
    # 每个样本的最终得分
    resScore = scoreMat.sum(axis=1)
    # 所有结果的平均得分
    lastResScore = np.mean(resScore)
    # print "最终得分为：%f"%lastResScore
    return lastResScore

# 天池得分计算公式/单天
def ScoreFormula(tup2):
    act_y, pre_y = tup2
    score = 10 * (1 - float(abs(act_y-pre_y))/float(act_y))
    # 低于3分 不得分
    if score < 3:
        score = 0
    return score

# 生成csv结果
def general_csvResult(purchaseRes, redeemRes):
    tableHeader = "report_date,purchase,redeem\n"
    with open(csv_path, "wb") as fp:
        fp.write(tableHeader)
        time = 20140901
        for (purchase, redeem) in zip(purchaseRes, redeemRes):
            line = "%s,%0.2f,%0.2f\n"%(str(time), abs(purchase), abs(redeem))
            time = time + 1
            fp.write(line)

if __name__ == '__main__':
    flag = "tune"  #tune & predict
    # make samples
    balanceDF, shareDF, shiborDF = dataSource()
    pdata = preprocess(balanceDF, shareDF, shiborDF)
    print pdata.shape
    # 降维参数
    n_components = 40
    forward = 400
    backward = 30

    # 参数调优化
    if flag == "tune":
        for forward_param in range(10, forward, 10):
            orgX, orgY, preFeature = generalSample(pdata, forward_param, backward, key_total_purchase_amt)
            # modeling
            score = Algthrim_Tune(orgX, orgY, n_components)
            print "forward=%d, score=%d"%(forward_param, score)
    # 9月份结果预测
    if flag == "predict":
        opt_forward = 220
        #赎回预测
        orgX, orgY, preFeature = generalSample(pdata, opt_forward, backward, key_total_redeem_amt)
        pre_redeem_amt = Algthrim_Predict(orgX, orgY, n_components, preFeature)
        #购买预测
        orgX, orgY, preFeature = generalSample(pdata, opt_forward, backward, key_total_purchase_amt)
        pre_purchase_amt = Algthrim_Predict(orgX, orgY, n_components, preFeature)
        print pre_redeem_amt[0]
        print pre_purchase_amt[0]
        general_csvResult(pre_purchase_amt[0], pre_redeem_amt[0])

