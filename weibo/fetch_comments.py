# '''
#     url template
#     某条微博的id，返回页面
#     https://m.weibo.cn/api/comments/show?id={id}&page={page}
#     该微博用户的微博列表，page定位到第几页，获取id
#     https://m.weibo.cn/api/container/getIndex?containerid={oid}&type=uid&value={uid}&page={page}
#     微博用户主页，获取uid和oid
#     https://m.weibo.cn/api/container/getIndex?type=uid&value={usr_id}
# '''
# from urllib import request
#
#
# class weibo_comment:
#     def __init__(self):
#         return self
#
#     def usr_info(usr_id):
#         url = 'http://m.weibo.cn/api/container/getIndex/type=uid&value={usr_id}'.format(usr_id = usr_id)
#         res = request.

import random
def random_pick(seq, prob):
    x = random.uniform(0,1)
    cul_prob = 0.0
    for item, item_prob in zip(seq, prob):
        cul_prob += item_prob
        if x < cul_prob:
            break
        return item

for i in range(10):
    print(random_pick('abc',[0.1,0.3,0.6]))