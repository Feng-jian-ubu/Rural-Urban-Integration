import pandas as pd
from linearmodels.panel import PanelOLS

# 加载数据
pca_data = pd.read_excel(r"C:\Users\21026\Desktop\统计建模大赛\数据集\城市生态韧性\PCA_城市生态韧性.xlsx")
did_data = pd.read_excel(r"C:\Users\21026\Desktop\统计建模大赛\数据集\DID.xlsx")
controls_data = pd.read_excel(r"C:\Users\21026\Desktop\统计建模大赛\数据集\控制变量.xlsx")

# 合并数据，基于'城市'和'年份'列
data = pd.merge(pca_data, did_data, on=["城市", "年份"])
data = pd.merge(data, controls_data, on=["城市", "年份"])

# 设置面板数据索引
data = data.set_index(['城市', '年份'])

# 显式添加常数项
data['constant'] = 1  # 手动添加常数项

# 定义回归公式，因变量是'Eco_Resilience'（来自PCA数据），解释变量是'DID'，同时加入控制变量和双向固定效应
# 控制变量列名：'人口规模'，'经济发展水平'，'对外开放水平'，'城镇化率'，'医疗卫生水平'
formula = 'Eco_Resilience ~ DID + 人口规模 + 经济发展水平 + 对外开放水平 + 城镇化率 + 医疗卫生水平 + constant + EntityEffects + TimeEffects'

# 执行面板数据回归，考虑城市和时间的固定效应，使用聚类标准误（按城市聚类）
model = PanelOLS.from_formula(formula, data)
results = model.fit(cov_type='clustered', cluster_entity=True)

# 输出回归结果（包括常数项）
print("回归结果：")
print(results.summary)  # 打印回归结果

# 显示回归系数
print("\n回归系数：")
print(results.params)

# 提取常数项（截距项）的系数和标准误
# 打印常数项和它的标准误
const_coeff = results.params['constant']  # 常数项系数
const_se = results.std_errors['constant']  # 常数项标准误

print(f"常数项系数: {const_coeff}")
print(f"常数项标准误: {const_se}")

# 提取回归结果：调整后的R^2、回归系数和标准误
r_squared = results.rsquared
coefficients = results.params
standard_errors = results.std_errors

# 打印R^2、回归系数和标准误
print(f"调整后的R^2: {r_squared}")
print(f"回归系数:\n{coefficients}")
print(f"标准误差:\n{standard_errors}")