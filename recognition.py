import jieba
import jieba.posseg as pseg
import sklearn.feature_extraction.text
import joblib
from time import time


class TfidfVectorizer(sklearn.feature_extraction.text.TfidfVectorizer):
    def build_analyzer(self):
        def analyzer(doc):
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x')
            words = jieba.cut(new_doc)
            return words

        return analyzer


def rec(text):
    # gpus = '一次价值xxx元王牌项目；可充值xxx元店内项目卡一张；可以参与V动好生活百分百抽奖机会一次！预约电话：xxxxxxxxxxx'
    '''gpus = ['妈妈很有耐心的拿pad给他放歌和玩耍',
            '感谢致电杭州萧山全金釜韩国烧烤店，本店位于金城路xxx号。韩式烧烤等，价格实惠、欢迎惠顾【全金釜韩国烧烤店】',
            '这款UVe智能杀菌机器人是扫地机的最佳伴侣',
            '一次价值xxx元王牌项目；可充值xxx元店内项目卡一张；可以参与V动好生活百分百抽奖机会一次！预约电话：xxxxxxxxxxx',
            '此类皮肤特别容易招惹粉刺、黑头等',
            '乌兰察布丰镇市法院成立爱心救助基金',
            '(长期诚信在本市作各类资格职称（以及印 /章、牌、 ……等。祥：x x x x x x x x x x x 李伟%',
            '《依林美容》三．八．女人节倾情大放送活动开始啦！！！！超值套餐等你拿，活动时间x月x日一x月xx日，     详情进店咨询。美丽热线x',
            '品牌墙/文化墙设计参考',
            '苏州和无锡两地警方成功破获了一起劫持女车主的案件',
            ]'''

    '''for item in gpus:
        text = [item]
        X1 = []
        X2 = []

        X1.append(item)

        for i in range(len(X1)):
            words = pseg.cut(X1[i])
            str1 = ""
            for key in words:
                str1 += key.word
                str1 += ' '
            X2.append(str1)  # 短信内容'''

    start = time()
    vec_tfidf = joblib.load("datafile/vec_tfidf")  #note absolute path
    data_tfidf = vec_tfidf.transform([text])
    modelb = joblib.load('model/SVM_sklearn.pkl')
    predict = modelb.predict(data_tfidf)
    end = time()

    if predict == "0":
        print("识别结果: 非垃圾短信  用时: %0.3fs" % (end - start))
    elif predict == "1":
        print("识别结果: 垃圾短信!  用时: %0.3fs" % (end - start))
    return predict


# print(rec('一次价值xxx元王牌项目；可充值xxx元店内项目卡一张；可以参与V动好生活百分百抽奖机会一次！预约电话：xxxxxxxxxxx'))
