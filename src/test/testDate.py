#coding=utf-8
from datetime import datetime

import pandas as pd
import numpy as np

def test_overview():
    rng = pd.date_range('1/1/2011', periods=72, freq='H')
    ts = pd.Series(np.random.randn(len(rng)), index=rng)
    print "ts:", ts
    converted = ts.asfreq('45Min', method='pad')
    resamp = ts.resample('D').mean()
    print resamp

def test_timestamp():
    ts1 = pd.Timestamp(datetime(2012, 5, 1))
    print "ts1:", type(ts1), ts1
    ts2 = pd.Period('2011-01')
    print "ts2:", type(ts2), ts2
    ts3 = pd.Period('2012-05', freq='D')
    print "ts3:", type(ts3), ts3

def convertTimestamp():
    res1 = pd.to_datetime(pd.Series(['jul 31, 2009', '2010-01-10', None]))
    print res1
    res2 = pd.to_datetime(['04-01-2010 10:00'], dayfirst=True)
    print res2
    res3 = pd.to_datetime(['14-01-2012', '01-14-2012'], dayfirst=True)
    print res3
    res4 = pd.to_datetime('2010/11/12')
    print res4
    res5 = pd.to_datetime('2010/11/12', format='%Y/%m/%d')
    print res5
    res6 = pd.to_datetime('12-11-2010 00:00', format='%d-%m-%Y %H:%M')
    print res6
    res7 = pd.to_datetime([1349720105, 1349806505, 1349892905,1349979305, 1350065705], unit='s')
    print res7
    res8 = pd.to_datetime([1349720105100, 1349720105200, 1349720105300,1349720105400, 1349720105500 ], unit='ms')
    print res8
    stamps = pd.date_range('2012-10-08 18:15:05', periods=4, freq='D')
    stamps.view('int64')
    print pd.Timedelta(1, unit='s')

def testgroupby():
    print "ok"
    df2 = pd.DataFrame({'A': 1.,
                        'B': pd.Timestamp('20130102'),
                        'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                        'D': np.array([3] * 4, dtype='int32'),
                        'E': pd.Categorical(["1", "3", "2", "3"]),
                        'F': 'foo'})
    print df2
    df3 = df2.groupby("E").size()
    print df3

if __name__ == '__main__':
    # convertTimestamp()
    testgroupby()