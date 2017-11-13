'''
从文件中读取行数，选择相应的column进行统计
'''
import json
from collections import defaultdict
import matplotlib.pylab as plt

path = 'ch2/file.txt'
records = [json.loads(line) for line in open(path)]
time_zones = [rec['tz'] for rec in records]

def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
        return counts

def get_counts2(sequence):
    ## 初始值都为0
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts

def top_conts(count_dict, n = 10):
    key_val = [(count, tz) for tz, count in count_dict.items()]
    key_val.sort()
    ## 返回top数值，key_val.sort(reverse = True)
    return key_val[-n:]

'''
from collections import Counter
counts = Counter(records)
counts.most_common(10)
'''

import pandas as pd
import numpy as np

frame = pd.DataFrame(records)
## 对某行的数据进行统计
tz_counts = frame['tz'].value_counts()
tz_counts[:10]

## 替换未知的数值
clean_tz = frame('tz').fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()
tz_counts[:10].plot(kind = 'bar', rot = 0)
