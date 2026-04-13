import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# =============================
# 1. 文件路径
# =============================

control_path = r"D:\统计建模\数据集\控制变量.xlsx"
resilience_path = r"D:\统计建模\数据集\熵权法_城市生态韧性.xlsx"
policy_path = r"D:\统计建模\数据集\DID.xlsx"

# =============================
# 2. 读取数据
# =============================

control = pd.read_excel(control_path)
resilience = pd.read_excel(resilience_path)
policy = pd.read_excel(policy_path)

# =============================
# 3. 合并数据
# =============================

resilience.rename(columns={"City": "城市", "Year": "年份"}, inplace=True)

df = control.merge(resilience, on=["城市", "年份"]) \
            .merge(policy, on=["城市", "年份"])

df.rename(columns={"Treat×Time": "Treat_Time"}, inplace=True)

# =============================
# 4. 变量设定
# =============================

y = "Eco_Resilience"

did_var = "Treat_Time"

controls = [
    "人口规模",
    "经济发展水平",
    "对外开放水平",
    "城镇化率",
    "医疗卫生水平"
]

all_vars = [y, did_var] + controls

# =============================
# 变量中文解释
# =============================

var_label = {
    "Eco_Resilience": "生态韧性",
    "Treat_Time": "政策变量",
    "人口规模": "户籍人口（取对数）",
    "经济发展水平": "人均地区生产总值（取对数）",
    "对外开放水平": "实际利用外资额/地区生产总值",
    "城镇化率": "非农业人口/户籍人口",
    "医疗卫生水平": "每百人医院、卫生院床位"
}

# =========================================================
# 相关性矩阵
# =========================================================

corr = df[all_vars].corr().round(4)

corr_copy = corr.copy()

for i in range(len(corr_copy.columns)):
    for j in range(i+1, len(corr_copy.columns)):
        corr_copy.iloc[i, j] = np.nan

for i in range(len(corr_copy.columns)):
    corr_copy.iloc[i, i] = 1.0

print("\n===== 相关性分析 =====\n")
print(corr_copy)
