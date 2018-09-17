#coding=utf-8
import src.dataFetch.ReadData as rd
import pandas as pd
from src.conf.constants import *
from matplotlib import pyplot as plt
from matplotlib.font_manager import *
from collections import Counter
import numpy as np
from multiprocessing import Process
import multiprocessing

#matplotlib中文字体设置
myfont = FontProperties(fname='/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc')

#dataframe统计分析
def statisic_df(dataDF, tableName):
    keys = dataDF.keys()._values
    (total_rowNum, total_colNum) = dataDF.shape
    logger.info("%s 表头:%s", tableName, keys)
    logger.info("%s 总行数：%s", tableName, total_rowNum)
    logger.info("%s 总列数：%s", tableName, total_colNum)
    # logger.info("discribe: ")
    # logger.info(dataDF.describe())

#绘制柱形图
def draw_bar(title, x_values, y_values, figureName):
    plt.figure(figureName)
    index = np.arange(len(x_values))
    plt.title(title, fontproperties=myfont)
    plt.bar(index, y_values)
    plt.xticks(index+0.0, x_values, fontproperties=myfont)
    plt.show()

def analysis_profile():
    logger.info("****分析用户基本数据表****")
    tableName = "用户基本数据表"
    #读取数据
    profile_dataDF = rd.get_profile()
    keys = profile_dataDF.keys()._values
    statisic_df(profile_dataDF, tableName)
    #画图
    for key in keys:
        if key=="user_id":
            continue
        else:
            columnList = profile_dataDF[key].tolist()
            #统计类别数目
            counter = Counter(columnList)
            keys = counter.keys()
            values = counter.values()
            #绘制bar图
            draw_bar(key, keys, values, key)

def analysis_blance():
    logger.info("****分析用户申购赎回表****")
    tableName = "用户申购赎回表"
    balance_dataDF = rd.get_balance()
    #统计
    statisic_df(balance_dataDF, tableName)
    userid_set = set(balance_dataDF['user_id'].tolist())
    logger.info("%s 用户总人数：%s", tableName, len(userid_set))
    #统计每个用户申购赎回次数
    group_res = balance_dataDF.groupby("user_id").size()
    freq_userid = group_res.values.tolist()
    # filter_freq = freq_userid.filter(lambda )

    # # 统计类别数目
    counter = Counter(freq_userid)
    # keys = counter.keys()
    # values = counter.values()
    # print keys
    # print values
    # # 绘制bar图
    # draw_bar("aa", keys, values, "aa")
    #
    # # date_list = balance_dataDF['report_date'].tolist()
    # # counter = Counter(date_list)
    # # keys = counter.keys()
    # # values = counter.values()
    # # logger.info(set(values))

def analysis_share():
    logger.info("分析余额宝收益表")
    tableName = "余额宝收益表"
    share_dataDF = rd.get_shareInterest()
    statisic_df(share_dataDF, tableName)
    userid_list = share_dataDF['user_id'].tolist()
    counter = Counter(userid_list)
    values = set(counter.values())
    logger.info(values)

def analysis_shibor():
    logger.info("分析银行间利率表")
    tableName = "银行间利率表"
    shibor_dataDF = rd.get_shibor()
    statisic_df(shibor_dataDF, tableName)
    userid_list = shibor_dataDF['mfd_date'].tolist()
    counter = Counter(userid_list)
    values = set(counter.values())
    logger.info(values)

if __name__ == '__main__':
    # analysis_profile()
    analysis_blance()
    # analysis_share()
    # analysis_shibor()