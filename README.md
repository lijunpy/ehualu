# ehualu
DC竞赛 易华录杯 公交线路准点预测 比赛代码。
比赛信息：http://www.dcjingsai.com/common/cmpt/%E5%85%AC%E4%BA%A4%E7%BA%BF%E8%B7%AF%E5%87%86%E7%82%B9%E9%A2%84%E6%B5%8B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html

## 经验汇总：

1. 10月1-8日为法定节假日，与其他日期的运行规律不同，不用该时段数据；
2. 所有数据中，8、9、10、14、18日有雨，可以删除不用；
3. 由于部分公司周六上班，因此可以考虑将日期类型分为 工作日、周六、周日三类；
4. 查找预测数据前一站数据时，如果没有对应数据，可以将预测得到的用时*0.618作为首站用时。

## 代码结构：

utils.py --通用函数；

settings.py --配置参数；

base.py --基础类；

preprocess.py --预处理；

fit.py --拟合；

predict.py --预测；