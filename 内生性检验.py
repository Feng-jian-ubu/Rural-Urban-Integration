import pandas as pd
import statsmodels.api as sm

# 读取数据
did_data = pd.read_excel(r"C:\Users\21026\Desktop\统计建模大赛\数据集\DID.xlsx")
theil_data = pd.read_excel(r"C:\Users\21026\Desktop\统计建模大赛\数据集\泰尔系数\城市2019年前泰尔系数平均值.xlsx")

# 合并两个数据集，按照城市进行匹配，确保城市相同
merged_data = pd.merge(did_data, theil_data, left_on="城市", right_on="城市", how="inner")

# 创建 Post_t 变量（2019及以后为1，2019年之前为0）
merged_data['Post_t'] = merged_data['年份'].apply(lambda x: 1 if x >= 2019 else 0)

# 创建工具变量：泰尔系数 × Post_t
merged_data['IV'] = merged_data['泰尔指数'] * merged_data['Post_t']

# 设置因变量和自变量（去除控制变量，只考虑城市固定效应和时间固定效应）
X = sm.add_constant(merged_data['IV'])  # 加入常数项
y = merged_data['Treat×Time']

# 运行回归模型
model = sm.OLS(y, X).fit()

# 输出回归结果
print(model.summary())

# 提取相关回归结果
coefficients = model.params
std_errors = model.bse
p_values = model.pvalues
f_statistic = model.fvalue

# 检查系数是否在 1% 水平上显著
significant_results = {}
for var, p_val in p_values.items():
    if p_val < 0.01:
        significant_results[var] = "显著（1%水平）"
    else:
        significant_results[var] = "不显著"

# 打印回归系数和显著性水平
print("回归系数：")
print(coefficients)
print("标准误差：")
print(std_errors)
print(f"F统计量：{f_statistic}")

print("\n显著性检验结果：")
print(significant_results)