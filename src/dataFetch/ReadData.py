#coding=utf-8

import pandas as pd
import numpy as np
import chardet
from src.conf.constants import *
data_url = '/home/260208/桌面/Data/Data1/'
path_profile = data_url + 'user_profile_table.csv'
path_balance = data_url + 'user_balance_table.csv'
path_shareInterest = data_url + 'mfd_day_share_interest.csv'
path_shibor = data_url + 'mfd_bank_shibor.csv'

#检测文本格式
def detect_fileCode(file_path):
    f = open(file_path, 'r')
    result = chardet.detect(f.read())
    file_name = file_path.split('/')[-1]
    logger.info(file_name + "文件编码格式:" + str(result))

#用户信息表
def get_profile():
    return pd.read_csv(path_profile, encoding="utf-8", dtype={'user_id':str, 'sex':str, 'city':str, 'constellation':str})

#申购赎回表20130701-20140831(14个月)
def get_balance():
    return pd.read_csv(path_balance, encoding='ascii', dtype={'user_id':str, 'report_date':str, 'total_purchase_amt':float})

#余额宝收益表 14个月
def get_shareInterest():
    return pd.read_csv(path_shareInterest, encoding='ascii', dtype={'mfd_date':str})

#银行利率表  14个月
def get_shibor():
    return pd.read_csv(path_shibor, encoding='ascii', dtype={'mfd_date':str})

if __name__ == '__main__':
    detect_fileCode(path_profile)
    detect_fileCode(path_balance)
    detect_fileCode(path_shareInterest)
    detect_fileCode(path_shibor)
