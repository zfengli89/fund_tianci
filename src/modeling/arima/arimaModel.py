#coding=utf-8

#function: arima modeling
#author: 260216

from matplotlib.font_manager import FontProperties
import src.dataFetch.ReadData as rd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# 中文字体格式配置
myfont = FontProperties(fname='/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

## 表头
# 日期
key_date = "report_date"
# 今日总购买量
key_total_purchase_amt = "total_purchase_amt"
# 今日总赎回量
key_total_redeem_amt = "total_redeem_amt"

# 数据源
def dataSource():
    print "data read"
    balanceDF = rd.get_balance()[[key_date, key_total_purchase_amt, key_total_redeem_amt]]
    return balanceDF

# 数据预处理
def preprocess(pdDF, plot=False):
    print "data preproces"
    pdDF.fillna(0, inplace=True)
    summaryByDate_DF = pdDF.groupby(key_date).sum()
    summaryByDate_DF.index = summaryByDate_DF.index.to_datetime()
    if plot:
        #plot
        plt.title("原始时间序列数据")
        summaryByDate_DF.plot()
        plt.show()
    return summaryByDate_DF

# 算法模型
def Algrithm_Arima(tdDF, order):
    model_arima = sm.tsa.ARIMA(tdDF[[key_total_purchase_amt]], order=order).fit()
    return model_arima

# 检验时间序列的平稳性与随机性
def checkout_StableAndRandom(df, title):
    #plot 时序图
    df.plot()
    plt.title(title)
    #plot acf pacf图
    plot_acf(df)
    plt.title(''.join([title, " Autocorrelation"]))
    plot_pacf(df)
    plt.title(''.join([title, " Partail Autocorrelation"]))
    #输出随机检验结果
    print title + u" 随机性检查结果: " + str(acorr_ljungbox(df, lags=1))
    plt.show()

# 启动时序数据检测
def run_checkout(df):
    print "序列平稳性与随机性检验, 确定差分阶数."
    one_diffDF = df.diff(1, axis=0).dropna(axis=0)
    two_diffDF = df.diff(2, axis=0).dropna(axis=0)
    checkout_StableAndRandom(df, u"原始序列信号")
    checkout_StableAndRandom(one_diffDF, u"一阶差分信号")
    checkout_StableAndRandom(two_diffDF, u"二阶差分信号")
    print "检验完成"

# 最小的模型参数p值 q值为：11 3
# 搜索模型参数空间，确定最佳p q参数
def run_getOptParamaters(df, d):
    print "运行ARIMA模型， 确定最佳p q参数"
    bic_matrix = []
    lengthDF = len(df)
    print "信号总长度为： %s"%lengthDF
    pmax = int(lengthDF/10)
    qmax = int(lengthDF/10)
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try:
                order = (p, d, q)
                print "parameters: %s"%str(order)
                model = Algrithm_Arima(df, order)
                tmp.append(model.bic)
            # except ValueError:
            #跳出各种error
            except :
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix)
    bic_matrix = pd.DataFrame(bic_matrix)
    p,q = bic_matrix.stack().idxmin()
    print "最小的模型参数p值 q值为：%s %s"%(p, q)
    print "确定参数完成"

#模型预测
def model_predict(df, order, start, end):
    model = Algrithm_Arima(df, order)
    # res = model.forecast(num)[0]
    res = model.predict(start, end, typ='levels')
    return res

#模型评估
def model_metric(actual, predict):
    None

if __name__ == '__main__':
    pdata = dataSource()
    tdata = preprocess(pdata)
    #总购买量时间序列
    purchaseDF = tdata[[key_total_purchase_amt]]
    trainDF =  purchaseDF['2013-07-01':'2014-07-31']
    # start = '2014-08-01'
    # end = '2014-08-31'
    start = '2014-01-01'
    end = '2014-08-31'
    testDF = purchaseDF[start:end]
    run_checkout(trainDF)
    # flag = False
    # #差分阶数
    # diffOrder = 2
    # if flag:
    #     run_getOptParamaters(trainDF, diffOrder)
    # if not flag:
    #     p = 11
    #     q = 3
    #     predictRes = model_predict(trainDF, (p, diffOrder, q), start, end)
    #     print predictRes
    #     testDF['predict'] = predictRes
    #     #sub res
    #     subRes = testDF[key_total_purchase_amt]-testDF['predict']
    #     subRes.plot()
    #     testDF.plot()
    #     plt.show()
    #     print testDF