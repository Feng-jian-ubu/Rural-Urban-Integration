import pandas as pd
import statsmodels.api as sm

# 读取数据
did_data = pd.read_excel(r"C:\Users\21026\Desktop\统计建模大赛\数据集\DID.xlsx")
theil_data = pd.read_excel(r"C:\Users\21026\Desktop\统计建模大赛\数据集\泰尔系数\城市2019年前泰尔系数平均值.xlsx")
control_data = pd.read_excel(r"C:\Users\21026\Desktop\统计建模大赛\数据集\控制变量.xlsx")

# 合并数据集，确保城市匹配
merged_data = pd.merge(did_data, theil_data, left_on="城市", right_on="城市", how="inner")
merged_data = pd.merge(merged_data, control_data, on=["城市", "年份"], how="inner")

# 创建 Post_t 变量（2019及以后为1，2019年之前为0）
merged_data['Post_t'] = merged_data['年份'].apply(lambda x: 1 if x >= 2019 else 0)

# 创建工具变量：泰尔系数 × Post_t
merged_data['IV'] = merged_data['泰尔指数'] * merged_data['Post_t']

# 选择控制变量（假设控制变量在控制数据表的后五列）
control_vars = control_data.columns[2:]  # 去掉前两列城市和年份

# 设置因变量和自变量，加入控制变量，考虑城市和时间固定效应
X = sm.add_constant(merged_data[['IV'] + list(control_vars)])  # 加入常数项和控制变量
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