# @Time    : 2018/12/4 14:57
# @Email  : wangchengo@126.com
# @File   : STMatrix.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0
import numpy as np
import pandas as pd
from .timestamp import string2timestamp


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps# [b'2013070101', b'2013070102']
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()  # 将时间戳：做成一个字典，也就是给每个时间戳一个序号

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):  # 给定时间戳返回对于的数据
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        """current version

        """
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)  # 时间偏移 minutes = 30
        # XC = []
        # XP = []
        # XT = []
        X_CP = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)]]
        # print depends # [range(1, 4), [48, 96, 144], [336, 672, 1008]]
        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps):
            check_seq = [self.pd_timestamps[i] - (PeriodInterval * self.T * p + c) * offset_frame for p in range(len_period + 1) for c in range(1, len_closeness + 1)]
            if not self.check_it(check_seq):
                i += 1
                continue


            x_cp = [[self.get_matrix(self.pd_timestamps[i] - (PeriodInterval * self.T * p + c) * offset_frame) for c in range(1, len_closeness + 1)] for p in range(len_period + 1)]

            y = self.get_matrix(self.pd_timestamps[i])

            X_CP.append(np.array(x_cp))

            Y.append(y)
            timestamps_Y.append(self.timestamps[i])#[]
            i += 1

        X_CP = np.asarray(X_CP)
        Y = np.asarray(Y)# [?,2,32,32]
        print("X_CP shape: ", X_CP.shape, "Y shape:", Y.shape)
        return X_CP, Y, timestamps_Y


if __name__ == '__main__':
    # depends = [range(1, 3 + 1),
    #            [1 * 48 * j for j in range(1, 3 + 1)],
    #            [7 * 48 * j for j in range(1, 3 + 1)]]
    # print(depends)
    # print([j for j in depends[0]])
    str = ['2013070101']
    t = string2timestamp(str)
    offset_frame = pd.DateOffset(minutes=24 * 60 // 48)  # 时间偏移 minutes = 30
    print(t)
    o = [t[0] - j * offset_frame for j in range(1, 4)]
    print(o)