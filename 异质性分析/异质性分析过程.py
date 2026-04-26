import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# =====================================================
# 1. 文件路径
# =====================================================

eco_path = r"C:\Users\21026\Desktop\统计建模大赛\数据集\城市生态韧性\熵权法_城市生态韧性.xlsx"
did_path = r"C:\Users\21026\Desktop\统计建模大赛\数据集\DID.xlsx"
control_path = r"C:\Users\21026\Desktop\统计建模大赛\数据集\控制变量.xlsx"
region_path = r"C:\Users\21026\Desktop\地域.xlsx"

output_path = r"C:\Users\21026\Desktop\异质性分析_东西部结果.xlsx"


# =====================================================
# 2. 工具函数
# =====================================================

def clean_city_name(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x = x.replace("\u3000", "")
    x = x.replace(" ", "")
    x = x.replace("　", "")
    if x.endswith("市"):
        x = x[:-1]
    return x


def clean_year(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    year = pd.Series([x]).str.extract(r"(\d{4})")[0].iloc[0]
    if pd.isna(year):
        return np.nan
    return int(year)


def significance_stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    else:
        return ""


# =====================================================
# 3. 读取数据
# =====================================================

eco_raw = pd.read_excel(eco_path)
did_raw = pd.read_excel(did_path)
control_raw = pd.read_excel(control_path)
region_raw = pd.read_excel(region_path)

eco_raw = eco_raw.loc[:, ~eco_raw.columns.duplicated()]
did_raw = did_raw.loc[:, ~did_raw.columns.duplicated()]
control_raw = control_raw.loc[:, ~control_raw.columns.duplicated()]
region_raw = region_raw.loc[:, ~region_raw.columns.duplicated()]


# =====================================================
# 4. 按位置提取数据
# =====================================================

eco = eco_raw.iloc[:, [0, 1, 2]].copy()
eco.columns = ["城市", "年份", "Eco_Resilience"]

did = did_raw.iloc[:, [0, 1, 2]].copy()
did.columns = ["城市", "年份", "DID"]

control = control_raw.iloc[:, [0, 1, 2, 3, 4, 5, 6]].copy()
control.columns = [
    "城市", "年份",
    "人口规模", "经济发展水平", "对外开放水平", "城镇化率", "医疗卫生水平"
]

region = region_raw.iloc[:, [0, 1, 2]].copy()
region.columns = ["城市", "年份", "所属地域"]


# =====================================================
# 5. 清洗城市和年份
# =====================================================

for df_temp in [eco, did, control, region]:
    df_temp["城市"] = df_temp["城市"].apply(clean_city_name)
    df_temp["年份"] = df_temp["年份"].apply(clean_year)

eco = eco.dropna(subset=["城市", "年份"])
did = did.dropna(subset=["城市", "年份"])
control = control.dropna(subset=["城市", "年份"])

# 注意：地域表只需要城市和所属地域，年份不重要
region = region.dropna(subset=["城市", "所属地域"])

eco["年份"] = eco["年份"].astype(int)
did["年份"] = did["年份"].astype(int)
control["年份"] = control["年份"].astype(int)


# =====================================================
# 6. 只保留 2014—2023 年
# 注意：地域表不筛选年份
# =====================================================

eco = eco[(eco["年份"] >= 2014) & (eco["年份"] <= 2023)]
did = did[(did["年份"] >= 2014) & (did["年份"] <= 2023)]
control = control[(control["年份"] >= 2014) & (control["年份"] <= 2023)]


# =====================================================
# 7. 构造东部虚拟变量
# =====================================================

region["所属地域"] = region["所属地域"].astype(str).str.strip()

region["East"] = np.where(region["所属地域"] == "东部", 1, 0)

region_city = (
    region[["城市", "East", "所属地域"]]
    .drop_duplicates(subset=["城市"], keep="first")
)

print("=" * 80)
print("地域信息检查")
print("=" * 80)
print("地域信息表城市数量：", region_city["城市"].nunique())
print("东部城市数量：", region_city[region_city["East"] == 1]["城市"].nunique())
print("中西部城市数量：", region_city[region_city["East"] == 0]["城市"].nunique())
print("\n地域信息前10行：")
print(region_city.head(10))


# =====================================================
# 8. 合并前检查城市交集
# =====================================================

print("\n" + "=" * 80)
print("合并前城市交集检查")
print("=" * 80)

eco_cities = set(eco["城市"].unique())
did_cities = set(did["城市"].unique())
control_cities = set(control["城市"].unique())
region_cities = set(region_city["城市"].unique())

print("生态韧性城市数：", len(eco_cities))
print("DID城市数：", len(did_cities))
print("控制变量城市数：", len(control_cities))
print("地域信息城市数：", len(region_cities))

print("生态韧性 ∩ DID 城市数：", len(eco_cities & did_cities))
print("生态韧性 ∩ DID ∩ 控制变量 城市数：", len(eco_cities & did_cities & control_cities))
print("四表共同城市数：", len(eco_cities & did_cities & control_cities & region_cities))

if len(eco_cities & did_cities & control_cities & region_cities) == 0:
    print("\n警告：四张表没有共同城市。下面打印部分城市名称供你检查：")
    print("生态韧性城市示例：", list(eco_cities)[:20])
    print("DID城市示例：", list(did_cities)[:20])
    print("控制变量城市示例：", list(control_cities)[:20])
    print("地域信息城市示例：", list(region_cities)[:20])


# =====================================================
# 9. 合并数据
# =====================================================

print("\n" + "=" * 80)
print("合并过程检查")
print("=" * 80)

df_eco_did = eco.merge(did, on=["城市", "年份"], how="inner")
print("生态韧性 × DID 合并后样本量：", len(df_eco_did))
print("生态韧性 × DID 合并后城市数：", df_eco_did["城市"].nunique())

df_eco_did_control = df_eco_did.merge(control, on=["城市", "年份"], how="inner")
print("继续合并控制变量后样本量：", len(df_eco_did_control))
print("继续合并控制变量后城市数：", df_eco_did_control["城市"].nunique())

df = df_eco_did_control.merge(region_city[["城市", "East"]], on="城市", how="inner")
print("继续合并地域信息后样本量：", len(df))
print("继续合并地域信息后城市数：", df["城市"].nunique())


# =====================================================
# 10. 如果最终为空，停止并打印诊断
# =====================================================

if len(df) == 0:
    print("\n最终合并样本为空。请重点看上面的“四表共同城市数”。")

    print("\n生态韧性表前5行：")
    print(eco.head())

    print("\nDID表前5行：")
    print(did.head())

    print("\n控制变量表前5行：")
    print(control.head())

    print("\n地域表前5行：")
    print(region_city.head())

    raise ValueError("最终合并样本为空，无法回归。")


# =====================================================
# 11. 变量格式处理
# =====================================================

num_cols = [
    "Eco_Resilience", "DID", "East",
    "人口规模", "经济发展水平", "对外开放水平", "城镇化率", "医疗卫生水平",
    "年份"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[
    "Eco_Resilience", "DID", "East",
    "人口规模", "经济发展水平", "对外开放水平", "城镇化率", "医疗卫生水平",
    "城市", "年份"
])

df["年份"] = df["年份"].astype(int)

df["DID_East"] = df["DID"] * df["East"]


# =====================================================
# 12. 最终样本检查
# =====================================================

print("\n" + "=" * 80)
print("最终样本检查")
print("=" * 80)

print("最终样本量：", len(df))
print("最终城市数量：", df["城市"].nunique())
print("年份范围：", int(df["年份"].min()), "-", int(df["年份"].max()))

print("\n东部与中西部城市数量：")
print(df.groupby("East")["城市"].nunique().rename(index={0: "中西部", 1: "东部"}))

print("\nDID=1 的试点城市分布：")
treated_check = df[df["DID"] == 1].groupby("East")["城市"].nunique()
if len(treated_check) == 0:
    print("没有识别到 DID=1 的样本，请检查 DID 列。")
else:
    print(treated_check.rename(index={0: "中西部试点", 1: "东部试点"}))


# =====================================================
# 13. 异质性 DID 回归
# =====================================================

formula = """
Eco_Resilience ~ DID + DID_East
+ 人口规模 + 经济发展水平 + 对外开放水平 + 城镇化率 + 医疗卫生水平
+ C(城市) + C(年份)
"""

model = smf.ols(formula=formula, data=df)

result = model.fit(
    cov_type="cluster",
    cov_kwds={"groups": df["城市"]}
)


# =====================================================
# 14. 整理拟合结果
# =====================================================

main_vars = [
    "DID", "DID_East",
    "人口规模", "经济发展水平", "对外开放水平", "城镇化率", "医疗卫生水平"
]

result_table = pd.DataFrame({
    "变量": main_vars,
    "回归系数": [result.params.get(v, np.nan) for v in main_vars],
    "标准误": [result.bse.get(v, np.nan) for v in main_vars],
    "t值": [result.tvalues.get(v, np.nan) for v in main_vars],
    "P值": [result.pvalues.get(v, np.nan) for v in main_vars],
})

result_table["显著性"] = result_table["P值"].apply(significance_stars)

print("\n" + "=" * 80)
print("异质性分析回归结果：东部 vs 中西部")
print("=" * 80)
print(result_table)


# =====================================================
# 15. 计算东部总政策效应：DID + DID_East
# =====================================================

b_did = result.params["DID"]
b_inter = result.params["DID_East"]

cov = result.cov_params()

east_effect = b_did + b_inter

east_var = (
    cov.loc["DID", "DID"]
    + cov.loc["DID_East", "DID_East"]
    + 2 * cov.loc["DID", "DID_East"]
)

east_se = np.sqrt(east_var)
east_t = east_effect / east_se
east_p = 2 * (1 - stats.norm.cdf(abs(east_t)))

effect_table = pd.DataFrame({
    "变量": [
        "中西部政策效应 DID",
        "东部额外效应 DID_East",
        "东部总政策效应 DID + DID_East"
    ],
    "回归系数": [
        b_did,
        b_inter,
        east_effect
    ],
    "标准误": [
        result.bse["DID"],
        result.bse["DID_East"],
        east_se
    ],
    "t值": [
        result.tvalues["DID"],
        result.tvalues["DID_East"],
        east_t
    ],
    "P值": [
        result.pvalues["DID"],
        result.pvalues["DID_East"],
        east_p
    ]
})

effect_table["显著性"] = effect_table["P值"].apply(significance_stars)

print("\n" + "=" * 80)
print("政策效应分解结果")
print("=" * 80)
print(effect_table)


# =====================================================
# 16. 模型拟合信息
# =====================================================

fit_info = pd.DataFrame({
    "指标": [
        "样本量",
        "城市数量",
        "年份数量",
        "R方",
        "调整后R方",
        "城市固定效应",
        "年份固定效应",
        "聚类标准误"
    ],
    "结果": [
        int(result.nobs),
        df["城市"].nunique(),
        df["年份"].nunique(),
        result.rsquared,
        result.rsquared_adj,
        "是",
        "是",
        "城市层面聚类"
    ]
})

print("\n" + "=" * 80)
print("模型拟合信息")
print("=" * 80)
print(fit_info)


# =====================================================
# 17. 保存结果
# =====================================================

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    result_table.to_excel(writer, sheet_name="主要回归结果", index=False)
    effect_table.to_excel(writer, sheet_name="政策效应分解", index=False)
    fit_info.to_excel(writer, sheet_name="模型拟合信息", index=False)
    df.to_excel(writer, sheet_name="合并后数据", index=False)

print("\n" + "=" * 80)
print("结果已保存到：")
print(output_path)
print("=" * 80)