import datetime

import akshare as ak
import numpy as np


def attribute_history(security, count, unit='1d', fields=None, skip_paused=True, df=True, fq='qfq'):
    """
    获取股票历史行情数据
    :return: pandas.DataFrame
    """
    # 将unit参数转换为pandas的时间频率字符串
    freq_map = {'1d': 'D', '1w': 'W', '1M': 'M'}
    freq = freq_map[unit]
    # 获取当前时间
    end_date = datetime.datetime.now()
    # 计算开始时间
    if unit == '1d':
        start_date = end_date - datetime.timedelta(days=count)
    elif unit == '1w':
        start_date = end_date - datetime.timedelta(weeks=count)
    elif unit == '1M':
        start_date = end_date - datetime.timedelta(months=count)
    # 获取股票历史行情数据
    print(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    data = ak.stock_zh_a_daily(symbol=security, start_date=start_date.strftime('%Y-%m-%d'),
                               end_date=end_date.strftime('%Y-%m-%d'), adjust=fq)
    # 将日期作为索引
    data = data.set_index('date')
    # 将数据按照时间频率进行重采样
    data = data.resample(freq).last()
    # 如果skip_paused为True，则跳过停牌的日期
    if skip_paused:
        paused_dates = data[data['volume'] == 0].index
        data = data.drop(paused_dates)
    # 返回指定的字段数据
    result = data[list(fields)]
    return result


def attribute_history_etf(security, count, unit='1d', fields=None, skip_paused=True, df=True, fq='qfq'):
    """
    获取股票历史行情数据
    :return: pandas.DataFrame
    """
    # 根据count和uint计算需要的数据量
    real_count = 0
    if unit == '1d':
        real_count = count * 1
    elif unit == '1w':
        real_count = count * 5
    elif unit == '1M':
        real_count = count * 21

    # 获取股票历史行情数据
    data = ak.fund_etf_hist_sina(symbol=security)
    # data.sort_values(by='date', ascending=False, inplace=True)
    last = data.tail(real_count)
    last = last[list(fields)]
    last = last.rename(index={x: x - last.index[0] for x in last.index})
    return last


def get_current_data(security):
    """
    获取当前的股票数据
    :return:
    """
    df = ak.fund_etf_category_sina(symbol='ETF基金')
    df = df.rename(columns={'代码': 'security', '最新价': 'last_price'})
    result = df.loc[df['security'] == security]
    return result


# 计算威廉指标
def get_william(security, n=14):
    data = attribute_history_etf(security, n, unit='1d', fields=['date', 'high', 'low', 'close'])
    # print(data)
    high = data.high.values
    # print(high)
    low = data.low.values
    # print(low)
    # 计算high和low的最大值和最小值
    high_max = np.max(high)
    low_min = np.min(low)
    # print(high_max)
    # print(low_min)
    # 计算威廉指标
    william = 100 * (high_max - data.close.values[-1]) / (high_max - low_min)
    # print(william)
    # (4132.295 - 4100.148) / (4132.295 - 3983.896)
    return william


if __name__ == '__main__':
    # data = attribute_history_etf('sh510300', 5, unit='1d', fields=['date', 'open', 'high', 'low', 'close', 'volume'])
    # print(data)
    # print(data.close.values[-1])
    # adr = 100 * (data.close.values[-1] - data.close.values[-2]) / data.close.values[-2]
    # print(adr)
    # data = get_current_data('sh510300').last_price.values[0]
    # print(data)
    wr1 = get_william('sh000300', 14)
    # wr2 = get_william('sh000300', 21)
    # print(f'WR1: {wr1}, WR2: {wr2}')
