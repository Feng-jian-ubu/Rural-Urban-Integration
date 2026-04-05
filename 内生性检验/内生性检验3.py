import pandas as pd
import statsmodels.api as sm

# 读取数据
did_data = pd.read_excel(r"C:\Users\21026\Desktop\统计建模大赛\数据集\DID.xlsx")
theil_data = pd.read_excel(r"C:\Users\21026\Desktop\统计建模大赛\数据集\泰尔系数\城市2019年前泰尔系数平均值.xlsx")
control_data = pd.read_excel(r"C:\Users\21026\Desktop\统计建模大赛\数据集\控制变量.xlsx")
eco_data = pd.read_excel(r"C:\Users\21026\Desktop\统计建模大赛\数据集\城市生态韧性\熵权法_城市生态韧性.xlsx")

# 合并数据集，确保城市匹配
merged_data = pd.merge(did_data, theil_data, left_on="城市", right_on="城市", how="inner")
merged_data = pd.merge(merged_data, control_data, on=["城市", "年份"], how="inner")
merged_data = pd.merge(merged_data, eco_data, on=["城市", "年份"], how="inner")

# 创建 Post_t 变量（2019及以后为1，2019年之前为0）
merged_data['Post_t'] = merged_data['年份'].apply(lambda x: 1 if x >= 2019 else 0)

# 创建工具变量：泰尔系数 × Post_t
merged_data['IV'] = merged_data['泰尔指数'] * merged_data['Post_t']

# 第一阶段回归：使用工具变量预测 DID
X_first_stage = sm.add_constant(merged_data[['IV']])  # 加入常数项
y_first_stage = merged_data['Treat×Time']

# 运行第一阶段回归
first_stage_model = sm.OLS(y_first_stage, X_first_stage).fit()

# 获取第一阶段回归拟合值，即预测的 DID
merged_data['Predicted_DID'] = first_stage_model.fittedvalues

# 第二阶段回归：用拟合的 DID 作为自变量进行回归
X_second_stage = sm.add_constant(merged_data[['Predicted_DID']])  # 加入常数项
y_second_stage = merged_data['Eco_Resilience']

# 运行第二阶段回归
second_stage_model = sm.OLS(y_second_stage, X_second_stage).fit()

# 输出回归结果
print(second_stage_model.summary())

# 提取相关回归结果
coefficients = second_stage_model.params
std_errors = second_stage_model.bse
p_values = second_stage_model.pvalues

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

print("\n显著性检验结果：")
print(significant_results)