import os
import json
import requests
import urllib3
import numpy as np
import pandas as pd
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================
# 0. Metris login config
# ============================================================

METRIS_URI = "https://localhost:9000"
METRIS_USERNAME = "Yiting"
METRIS_PASSWORD = 'Metris123*'   # 测试可以硬编码，正式建议放 .env 或 getpass


def login_metris():
    """
    登录 Metris，返回 headers。
    后面所有 API 请求都要用这个 headers。
    """
    auth_url = f"{METRIS_URI}/api/account/authenticate"

    payload = {
        "username": METRIS_USERNAME,
        "password": METRIS_PASSWORD
    }

    print("正在登录 Metris:")
    print(auth_url)

    r = requests.post(
        auth_url,
        json=payload,
        verify=False,
        timeout=(5, 20)
    )
    #verify验证访问一个http地址的时候，python要不要检查服务器的证书 ， timeout=(连接超时时间, 读取超时时间)，最多花5s取连接服务器 然后20s让服务器返回数据
    print("login status_code:", r.status_code)
    print("login response 前200字符:")
    print(r.text[:200])

    r.raise_for_status()

    token = r.json().get("id")

    if not token:
        raise RuntimeError("登录成功但没有拿到 token，请检查返回 JSON。")

    headers = {
        "Authorization": f"Bearer {token}"
    }

    print("登录成功，token 前20字符:", token[:20])

    return headers


# 全局 headers，后面函数都会用它
headers = login_metris()


# ============================================================
# 0. Import libraries
# ============================================================

import os
import json
import requests
import urllib3
import numpy as np
import pandas as pd
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================
# 2. 基本配置：10线 ER 炉次号 tag + prediction 变量 tag
# ============================================================
# 下面把“人能看懂的变量名”和“Metris 里面真正用来取数的 tag_id”做了一个映射表。
# 10线 / ER 炉次号 tag
LINE10_BATCH_NAME_TAG = 6288

# 只提取你 prediction 需要的变量
# key = 最终表里的变量名
# tag_id = Metris tag
# agg = 一个炉次窗口内如何聚合
TARGET_TAG_CONFIG = {
    "10#熔炼炉固体料重量比例": {
        "tag_id": 6370,
        "agg": "max"
    },
    "10#熔炼炉总投料重量(kg)": {
        "tag_id": 6368,
        "agg": "max"
    },
    "熔炼炉B当前批次熔炼时间_PLC": {
    "tag_id": 6272,
    "agg": "max"
    },
    "熔炼炉B当前批次炉门打开时长_PLC": {
        "tag_id": 6266,
        "agg": "max"
    },
    "熔炼炉B当前批次炉门打开次数_PLC": {
        "tag_id": 6264,
        "agg": "max"
    },
    "熔炼炉B当前批次等待时长_PLC": {
        "tag_id": 6268,
        "agg": "max"
    },
    "熔炼炉B当前批次总气耗_PLC": {
        "tag_id": 6279,
        "agg": "max"
    },
}

# ============================================================
# 3. 拉炉次号变化记录
# ============================================================

def get_string_history(tag_id, start_time, end_time):

    """
    拉炉次号 string history。

    返回:
        DataFrame columns:
            timestamp
            tagID
            valueString
    根据给定的 tag_id,从 Metris Historian 里拉取 start_time 到 end_time 之间这个 tag 的字符串历史数据。
    """

    url = f"{METRIS_URI}/api/historian/valuestringhistory"

    params = {
        "tagid": tag_id,
        "start": datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").isoformat(),
        "end": datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").isoformat(),
        "timeshift": 0,
        "interpolationmethod": 1,
        "interpolationresolution": 0,
        "interpolationresolutiontype": 0,
        "aggregatefunction": 0,
        "trackingreferencestep": None
    }

    r = requests.get(
        url,
        headers=headers,
        params=params,
        verify=False,
        timeout=(5, 60)
    )

    print("[string history] status_code:", r.status_code)
    r.raise_for_status()

    df = pd.DataFrame(json.loads(r.text))

    if df.empty:
        print("[WARN] 炉次号接口返回空数据")
        return pd.DataFrame(columns=["timestamp", "tagID", "valueString"])

    return df


# ============================================================
# 4. 根据炉次号变化构造 batch window
# ============================================================

def build_batch_windows(name_df, end_time=None, include_last_partial=False):
    """
    根据 valueString 的变化时间点构造炉次窗口。

    例如:
        ER033 timestamp -> ER034 timestamp
        就认为 ER033 的窗口是 [ER033_time, ER034_time)

    include_last_partial:
        False: 最后一个炉次因为没有下一个切换点，默认丢掉
        True : 最后一个炉次用传入的 end_time 作为结束时间
    """
    df_name = name_df.copy()

    if df_name.empty:
        return pd.DataFrame(columns=[
            "batch_id", 
            "start_time", 
            "end_time", 
            "start_time_cn", 
            "end_time_cn"
        ])

    df_name["timestamp_dt"] = pd.to_datetime(df_name["timestamp"], utc=True) #原始时间段转化成panda可以识别的
    df_name["batch_id"] = (
        df_name["valueString"]
        .astype(str)
        .str.strip() #去除空格
        .str.strip('"') #去除前后双引号
    ) #生成一列标准炉次号

    df_name = df_name.dropna(subset=["timestamp_dt", "batch_id"]) 
    df_name = df_name.sort_values("timestamp_dt").reset_index(drop=True)

    rows = []

    for i in range(len(df_name) - 1):
        #用相邻两个炉次记录构造窗口
        batch_id = df_name.loc[i, "batch_id"]
        st = df_name.loc[i, "timestamp_dt"]
        et = df_name.loc[i + 1, "timestamp_dt"]

        rows.append({
            "batch_id": batch_id,
            "start_time": st,
            "end_time": et,
            "start_time_cn": st.tz_convert("Asia/Shanghai"),
            "end_time_cn": et.tz_convert("Asia/Shanghai"),
        })

    # 是否保留最后一个不完整炉次
    if include_last_partial and len(df_name) >= 1 and end_time is not None:
        last_batch_id = df_name.loc[len(df_name) - 1, "batch_id"]
        last_st = df_name.loc[len(df_name) - 1, "timestamp_dt"]

        last_et = (
            pd.to_datetime(end_time)
            .tz_localize("Asia/Shanghai")
            .tz_convert("UTC")
        )

        rows.append({
            "batch_id": last_batch_id,
            "start_time": last_st,
            "end_time": last_et,
            "start_time_cn": last_st.tz_convert("Asia/Shanghai"),
            "end_time_cn": last_et.tz_convert("Asia/Shanghai"),
        })

    window_df = pd.DataFrame(rows)

    # 只保留 ER 批次
    if not window_df.empty:
        window_df = window_df[
            window_df["batch_id"].astype(str).str.startswith("ER")
        ].reset_index(drop=True)

    return window_df


# ============================================================
# 5. 拉某个 tag 在整段时间内的趋势数据
#    为了效率：每个 tag 只请求一次，然后本地按炉次窗口切片
# ============================================================

def get_numeric_trend_full_period(tag_id, start_time, end_time):
    """
    拉某个数值 tagid 在整段时间内的趋势数据。

    返回:
        DataFrame columns:
            t
            v
            timestamp
            timestamp_cn
    """
    url = f"{METRIS_URI}/api/historian/v02/trendvalues"

    params = {
        "tagid": tag_id,
        "start": datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").isoformat(),
        "end": datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").isoformat(),
        "timeshift": 0,
        "interpolationmethod": 1,
        "interpolationresolution": 0,
        "interpolationresolutiontype": 1,
        "aggregatefunction": 0,
        "trackingreferencestep": None
    }

    r = requests.get(
        url,
        headers=headers,
        params=params,
        verify=False,
        timeout=(5, 120)
    )

    print(f"[trend] tag_id={tag_id}, status_code={r.status_code}")
    r.raise_for_status()

    df = pd.DataFrame(r.json())

    if df.empty:
        return pd.DataFrame(columns=["t", "v", "timestamp", "timestamp_cn"])

    if "t" not in df.columns or "v" not in df.columns:
        print(f"[WARN] tag_id={tag_id} 返回列异常:", df.columns.tolist())
        return pd.DataFrame(columns=["t", "v", "timestamp", "timestamp_cn"])

    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df["timestamp_cn"] = df["timestamp"].dt.tz_convert("Asia/Shanghai")
    df["v"] = pd.to_numeric(df["v"], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# 6. 一个炉次窗口内聚合取值
# ============================================================

def pick_value_in_window(trend_df, start_ts, end_ts, agg="max"):
    """
    从整段趋势数据中，截取某个炉次窗口 [start_ts, end_ts)，然后聚合成一个值。
    trend_df:某一个id在一段时间内的趋势数据
    start_ts:某个炉次的开始时间
    end_ts:某个炉次结束时间
    agg
    """
    if trend_df is None or trend_df.empty:
        return np.nan

    if "timestamp" not in trend_df.columns or "v" not in trend_df.columns:
        return np.nan

    mask = (
        (trend_df["timestamp"] >= start_ts)
        & (trend_df["timestamp"] < end_ts)
    )

    s = trend_df.loc[mask, "v"]
    s = pd.to_numeric(s, errors="coerce").dropna()

    if len(s) == 0:
        return np.nan

    if agg == "max":
        return s.max()
    elif agg == "last":
        return s.iloc[-1]
    elif agg == "mean":
        return s.mean()
    elif agg == "min":
        return s.min()
    elif agg == "sum":
        return s.sum()
    else:
        return s.max()


# ============================================================
# 7. 主函数：拉指定时间范围，输出三种结构
# ============================================================

def fetch_prediction_data_excel_like(
    start_time,
    end_time,
    save_excel=False,
    output_dir=None,
    include_last_partial=False,
    show_detail=True
):
    """
    拉取 prediction 需要的数据，并生成原 Excel 类似结构。

    参数:
        start_time: str, 例如 "2026-02-10 00:00:00"
        end_time  : str, 例如 "2026-02-15 23:59:59"
        save_excel: 是否保存 Excel，默认 False，不保存
        output_dir: 保存路径，save_excel=True 时才用
        include_last_partial: 是否保留最后一个未闭合炉次
        show_detail: 是否 display 中间结果

    返回:
        matrix_df:
            行=变量，列=炉次号

        model_df:
            行=炉次，列=变量
            这个可以直接用于 prediction

        excel_like_raw_df:
            完全模拟你原始 Excel 的 header=None 格式

        window_df:
            炉次窗口表
    """

    print("=" * 80)
    print("1. 拉取 10线 ER 炉次号")
    print("=" * 80)

    name_df = get_string_history(
        tag_id=LINE10_BATCH_NAME_TAG,
        start_time=start_time,
        end_time=end_time
    )

    print("name_df shape:", name_df.shape)
    if show_detail:
        display(name_df.head(20))

    print("=" * 80)
    print("2. 构造炉次窗口")
    print("=" * 80)

    window_df = build_batch_windows(
        name_df=name_df,
        end_time=end_time,
        include_last_partial=include_last_partial
    )

    print("window_df shape:", window_df.shape)
    if show_detail:
        display(window_df)

    if window_df.empty:
        raise RuntimeError("没有构造出任何 ER 炉次窗口，请检查时间范围或 6288 炉次号数据。")

    print("=" * 80)
    print("3. 拉取所有需要的 tag 趋势数据")
    print("=" * 80)

    trend_cache = {}

    for variable_name, cfg in TARGET_TAG_CONFIG.items():
        tag_id = cfg["tag_id"]

        print(f"\n正在拉取变量: {variable_name}, tag_id={tag_id}")

        trend_df = get_numeric_trend_full_period(
            tag_id=tag_id,
            start_time=start_time,
            end_time=end_time
        )

        print(f"{variable_name} trend shape:", trend_df.shape)

        trend_cache[variable_name] = trend_df

    print("=" * 80)
    print("4. 按炉次窗口聚合成建模表 model_df")
    print("=" * 80)

    model_rows = []

    for _, batch_row in window_df.iterrows():
        batch_id = batch_row["batch_id"]
        st = batch_row["start_time"]
        et = batch_row["end_time"]

        row = {
            "batch_id": batch_id,
            "start_time": st,
            "end_time": et,
            "start_time_cn": batch_row["start_time_cn"],
            "end_time_cn": batch_row["end_time_cn"],
        }

        for variable_name, cfg in TARGET_TAG_CONFIG.items():
            trend_df = trend_cache[variable_name]
            agg = cfg.get("agg", "max")

            value = pick_value_in_window(
                trend_df=trend_df,
                start_ts=st,
                end_ts=et,
                agg=agg
            )

            row[variable_name] = value

        model_rows.append(row)

    model_df = pd.DataFrame(model_rows)

    print("model_df shape:", model_df.shape)
    if show_detail:
        display(model_df)

    print("=" * 80)
    print("5. 转成原 Excel 类似结构 matrix_df：行=变量，列=炉次号")
    print("=" * 80)

    batch_ids = model_df["batch_id"].tolist()
    variable_names = list(TARGET_TAG_CONFIG.keys())

    matrix_df = model_df.set_index("batch_id")[variable_names].T
    matrix_df = matrix_df[batch_ids]
    matrix_df.index.name = None

    print("matrix_df shape:", matrix_df.shape)
    if show_detail:
        display(matrix_df)

    print("=" * 80)
    print("6. 生成 raw-like 格式 excel_like_raw_df")
    print("=" * 80)

    excel_like_rows = []

    # 第一行：左上角是 NaN，后面是 ER 炉次号
    excel_like_rows.append([np.nan] + batch_ids)

    # 后续每一行：变量名 + 各炉次值
    for var in variable_names:
        values = matrix_df.loc[var, batch_ids].tolist()
        excel_like_rows.append([var] + values)

    excel_like_raw_df = pd.DataFrame(excel_like_rows)

    print("excel_like_raw_df shape:", excel_like_raw_df.shape)
    if show_detail:
        display(excel_like_raw_df)

    print("=" * 80)
    print("7. 缺失值检查")
    print("=" * 80)

    check_cols = list(TARGET_TAG_CONFIG.keys())

    missing_summary = model_df[["batch_id"] + check_cols].isna().sum().reset_index()
    missing_summary.columns = ["column", "missing_count"]
    missing_summary["missing_ratio"] = missing_summary["missing_count"] / len(model_df)

    display(missing_summary)

    # ========================================================
    # 可选保存，默认不保存
    # ========================================================

    if save_excel:
        if output_dir is None:
            output_dir = r"C:\Users\fshhan17\Desktop\output"

        os.makedirs(output_dir, exist_ok=True)

        clean_start = start_time.replace(":", "").replace(" ", "_").replace("-", "")
        clean_end = end_time.replace(":", "").replace(" ", "_").replace("-", "")

        matrix_path = os.path.join(
            output_dir,
            f"metris_prediction_matrix_{clean_start}_to_{clean_end}.xlsx"
        )

        model_path = os.path.join(
            output_dir,
            f"metris_prediction_model_df_{clean_start}_to_{clean_end}.xlsx"
        )

        raw_like_path = os.path.join(
            output_dir,
            f"metris_prediction_raw_like_{clean_start}_to_{clean_end}.xlsx"
        )

        matrix_df.to_excel(matrix_path, index=True)
        model_df.to_excel(model_path, index=False)
        excel_like_raw_df.to_excel(raw_like_path, index=False, header=False)

        print("\n已保存:")
        print("1. matrix_df:", matrix_path)
        print("2. model_df:", model_path)
        print("3. raw_like_excel:", raw_like_path)

    else:
        print("\n当前 save_excel=False，所以没有保存文件。")
        print("你可以直接使用 model_df / matrix_df / excel_like_raw_df。")

    return matrix_df, model_df, excel_like_raw_df, window_df


# ============================================================
# 8. 调用示例：先不保存，只拉数据打印出来
# ============================================================

start_time = "2026-02-01 00:00:00"
end_time = "2026-03-01 23:59:59"

matrix_df, model_df, excel_like_raw_df, window_df = fetch_prediction_data_excel_like(
    start_time=start_time,
    end_time=end_time,
    save_excel=False,              # 先不保存
    output_dir=None,
    include_last_partial=False,    # 最后一个未闭合炉次先不要
    show_detail=True
)


# ============================================================
# 9. 如果格式确认没问题，可以直接接 prediction
# ============================================================

target_col = "熔炼炉B当前批次总气耗_PLC"

feature_cols = [
    "10#熔炼炉固体料重量比例",
    "10#熔炼炉总投料重量(kg)",
    "熔炼炉B当前批次炉门打开时长_PLC",
    "熔炼炉B当前批次炉门打开次数_PLC",
    "熔炼炉B当前批次等待时长_PLC",
    "熔炼炉B当前批次熔炼时间_PLC",
]

df_for_prediction = model_df.copy()

X = df_for_prediction[feature_cols]
y = df_for_prediction[target_col]

print("=" * 80)
print("Prediction input X:")
print("=" * 80)
display(X.head())

print("=" * 80)
print("Prediction target y:")
print("=" * 80)
display(y.head())


#————————————————————————————————————————————————————————model predictionpart————————————————————————————————————————————
# ============================================================
# 简化版 Metris prediction workflow
# Data: model_df
# Logic:
#   1. 使用当前 Metris 拉下来的 model_df
#   2. 只保留当前需要的 X 和 y
#   3. 检查缺失值
#   4. 用 IQR 去除 y outlier
#   5. 训练 3 个模型:
#        - LightGBM
#        - Linear Regression
#        - Random Forest
#   6. 打印模型表现
#   7. 打印 Linear Regression 公式
#   8. 不保存任何文件
# ============================================================

import warnings
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LightGBM 如果环境里没有，就自动跳过
try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("[WARN] 当前环境没有 lightgbm，后面会跳过 LightGBM。")


# ------------------------------------------------------------
# 0. Warning control
# ------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*X does not have valid feature names.*"
)


# ============================================================
# 1. Use current dataset
# ============================================================

# 这里默认你前面已经跑完：
# matrix_df, model_df, excel_like_raw_df, window_df = fetch_prediction_data_excel_like(...)

df = model_df.copy()

print("当前 df shape:", df.shape)
display(df.head())


# ============================================================
# 2. Define target and feature columns
# ============================================================

target_col = "熔炼炉B当前批次总气耗_PLC"

feature_cols = [
    "10#熔炼炉固体料重量比例",
    "10#熔炼炉总投料重量(kg)",
    "熔炼炉B当前批次熔炼时间_PLC",
    "熔炼炉B当前批次炉门打开时长_PLC",
    "熔炼炉B当前批次炉门打开次数_PLC",
    "熔炼炉B当前批次等待时长_PLC",
]

required_cols = ["batch_id"] + feature_cols + [target_col]

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if len(missing_cols) > 0:
    raise ValueError(f"以下列在当前 df/model_df 中不存在，请检查前面是否已经提取这些变量: {missing_cols}")

print("\n当前 target:")
print(" -", target_col)

print("\n当前 features:")
for col in feature_cols:
    print(" -", col)


# ============================================================
# 3. Keep only modeling columns and convert numeric
# ============================================================

model_data = df[required_cols].copy()

# X + y 全部强制转成数值
for col in feature_cols + [target_col]:
    model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

print("\n建模数据 model_data shape:", model_data.shape)
display(model_data.head())


# ============================================================
# 4. Missing value check
# ============================================================

print("\n================ 缺失值检查 ================")

missing_summary = model_data.isna().sum().reset_index()
missing_summary.columns = ["column", "missing_count"]
missing_summary["missing_ratio"] = missing_summary["missing_count"] / len(model_data)

display(missing_summary)


# ============================================================
# 5. Basic describe
# ============================================================

print("\n================ 数值分布 describe ================")

describe_summary = model_data[feature_cols + [target_col]].describe().T
display(describe_summary)


# ============================================================
# 6. Drop rows with missing y
#    X 缺失值后面用 SimpleImputer 补
#    y 缺失不能训练，所以先删掉
# ============================================================

before_drop_y = len(model_data)

model_data = model_data.dropna(subset=[target_col]).reset_index(drop=True)

after_drop_y = len(model_data)

print("\n去掉 y 缺失前样本数:", before_drop_y)
print("去掉 y 缺失后样本数:", after_drop_y)

if len(model_data) < 5:
    raise ValueError(
        f"当前可训练样本数太少: {len(model_data)}。建议拉更长时间范围的数据后再训练。"
    )


# ============================================================
# 7. Analyze y distribution and outliers
# ============================================================

y = model_data[target_col]

print("\n================ y describe ================")
display(y.describe())

# IQR 方法找 y 异常点
q1 = y.quantile(0.25)
q3 = y.quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_mask = (y < lower_bound) | (y > upper_bound)

outlier_df = model_data.loc[
    outlier_mask,
    ["batch_id", target_col] + feature_cols
].copy()

print("y IQR lower bound:", lower_bound)
print("y IQR upper bound:", upper_bound)
print("y 异常点数量:", int(outlier_mask.sum()))

display(outlier_df)


# ============================================================
# 8. Prepare no-y-outlier data
# ============================================================

df_no_outlier = model_data.loc[~outlier_mask].copy().reset_index(drop=True)

print("\n全部可用样本数量:", len(model_data))
print("y outlier 数量:", int(outlier_mask.sum()))
print("去掉 y outlier 后样本数量:", len(df_no_outlier))

if len(df_no_outlier) < 5:
    raise ValueError(
        f"去掉 y outlier 后样本数太少: {len(df_no_outlier)}。"
        "建议先不要去 outlier，或者拉更长时间范围。"
    )


# ============================================================
# 9. Define X and y
# ============================================================

X = df_no_outlier[feature_cols]
y = df_no_outlier[target_col]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

display(X.head())
display(y.head())


# ============================================================
# 10. Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 保证列顺序一致
X_train = X_train[feature_cols]
X_test = X_test[feature_cols]

print("\n训练集样本数:", len(X_train))
print("测试集样本数:", len(X_test))


# ============================================================
# 11. Define models
# ============================================================

models = {}

if HAS_LIGHTGBM:
    models["LightGBM"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ))
    ])

models["Linear Regression"] = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

models["Random Forest"] = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])

print("\n当前参与比较的模型:")
for model_name in models:
    print(" -", model_name)


# ============================================================
# 12. Train and evaluate models
# ============================================================

trained_models = {}
model_summary_rows = []
prediction_detail_rows = []

for model_name, model in models.items():

    print("\n================================================")
    print("正在训练模型:", model_name)
    print("================================================")

    final_model = clone(model)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    trained_models[model_name] = final_model

    model_summary_rows.append({
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "RMSE_pct_of_y_test_mean": rmse / y_test.mean() if y_test.mean() != 0 else np.nan,
        "R2": r2,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
        "features": ", ".join(feature_cols)
    })

    pred_detail = pd.DataFrame({
        "batch_id": df_no_outlier.loc[y_test.index, "batch_id"].values,
        "model": model_name,
        "y_true": y_test.values,
        "y_pred": y_pred,
        "error": y_pred - y_test.values,
        "abs_error": np.abs(y_pred - y_test.values)
    })

    prediction_detail_rows.append(pred_detail)

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("RMSE / y_test_mean:", rmse / y_test.mean() if y_test.mean() != 0 else np.nan)
    print("R2:", r2)


# ============================================================
# 12.1 Save all trained models as individual pickle files
#
# 保存的是完整 sklearn Pipeline，因此会同时保存：
#   - 缺失值填充器 SimpleImputer
#   - StandardScaler（Linear Regression 使用）
#   - 已训练模型及其参数
# 预测时直接 pickle.load(...) 后调用 predict() 即可。
# ============================================================

MODEL_OUTPUT_DIR = Path("./SavedModels")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_file_names = {
    "LightGBM": f"LightGBM_{model_timestamp}.pickle",
    "Linear Regression": f"LinearRegression_{model_timestamp}.pickle",
    "Random Forest": f"RandomForest_{model_timestamp}.pickle",
}

saved_model_paths = {}

print("\n================ 保存已训练模型 ================")
for model_name, file_name in model_file_names.items():
    if model_name not in trained_models:
        print(f"[WARN] {model_name} 未训练，因此没有保存。")
        continue

    model_path = MODEL_OUTPUT_DIR / file_name
    with model_path.open("wb") as model_file:
        pickle.dump(
            trained_models[model_name],
            model_file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    saved_model_paths[model_name] = model_path
    print(f"[OK] {model_name} 已保存到: {model_path.resolve()}")

expected_models = set(model_file_names)
saved_models = set(saved_model_paths)
missing_saved_models = expected_models - saved_models

if missing_saved_models:
    print(
        "[WARN] 以下模型没有生成 pickle: "
        + ", ".join(sorted(missing_saved_models))
    )
else:
    print("三个模型均已成功保存为独立 pickle 文件。")

# ============================================================
# 12.2 Reload models from pickle for all following logic
#
# 关键点：从这里开始，后续的公式提取、最优模型选择和推荐系统
# 不再使用刚刚 fit 后仍留在内存中的模型，而是统一使用 pickle.load
# 重新读回来的模型。
# ============================================================

loaded_models = {}

for model_name, model_path in saved_model_paths.items():
    with model_path.open("rb") as model_file:
        loaded_model = pickle.load(model_file)

    if not hasattr(loaded_model, "predict"):
        raise TypeError(
            f"重新加载后的 {model_name} 没有 predict 方法: {model_path}"
        )

    loaded_models[model_name] = loaded_model
    print(f"[OK] 推荐模型已从 pickle 加载: {model_name} <- {model_path.resolve()}")

if not loaded_models:
    raise RuntimeError("没有从 pickle 成功加载任何模型，无法执行后续推荐。")

# 用 pickle 中重新加载的模型覆盖原 trained_models。
# 后面的推荐代码仍可保持 trained_models[...] 的原写法，
# 但其中的对象现在全部来自 pickle 文件。
trained_models = loaded_models

print("后续公式提取、模型选择和推荐将统一使用 pickle 中加载的模型。")


model_summary_df = pd.DataFrame(model_summary_rows).sort_values(
    ["RMSE", "MAE"]
).reset_index(drop=True)

prediction_detail_df = pd.concat(
    prediction_detail_rows,
    ignore_index=True
)

print("\n================ 模型表现对比表 ================")
display(model_summary_df)

print("\n================ 测试集预测明细 ================")
display(prediction_detail_df)


# ============================================================
# 13. Extract Linear Regression equation
# ============================================================

def extract_linear_equation(
    pipeline,
    feature_cols,
    model_name="Linear Regression"
):
    """
    提取 Linear Regression 的公式参数：
        1. 标准化空间公式
        2. 原始变量单位公式

    pipeline:
        imputer -> scaler -> model
    """

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    coef_scaled = model.coef_
    intercept_scaled = model.intercept_

    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_

    # 原始单位下：
    # y = intercept_scaled + sum(coef_scaled_i * ((x_i - mean_i) / scale_i))
    #   = intercept_original + sum(coef_original_i * x_i)
    coef_original = coef_scaled / scaler_scale
    intercept_original = intercept_scaled - np.sum(
        coef_scaled * scaler_mean / scaler_scale
    )

    equation_df = pd.DataFrame({
        "model": model_name,
        "feature": feature_cols,
        "coef_scaled_space": coef_scaled,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "coef_original_unit": coef_original,
        "imputer_strategy": imputer.strategy,
        "note": "原始单位公式基于 imputer 后再 scaler 的 pipeline 换算得到"
    })

    equation_df.loc[len(equation_df)] = {
        "model": model_name,
        "feature": "intercept",
        "coef_scaled_space": intercept_scaled,
        "scaler_mean": np.nan,
        "scaler_scale": np.nan,
        "coef_original_unit": intercept_original,
        "imputer_strategy": imputer.strategy,
        "note": "截距项"
    }

    print("\n================ Linear Regression Equation ================")

    print("\nA. 标准化变量空间下的公式:")
    print("y = {:.6f}".format(intercept_scaled))

    for feature, coef in zip(feature_cols, coef_scaled):
        print("    + ({:.6f}) * standardized({})".format(coef, feature))

    print("\nB. 原始变量单位下的公式:")
    print("y = {:.6f}".format(intercept_original))

    for feature, coef in zip(feature_cols, coef_original):
        print("    + ({:.6f}) * {}".format(coef, feature))

    return equation_df


if "Linear Regression" in trained_models:
    linear_equation_df = extract_linear_equation(
        pipeline=trained_models["Linear Regression"],
        feature_cols=feature_cols,
        model_name="Linear Regression"
    )

    print("\n================ Linear Regression 参数表 ================")
    display(linear_equation_df)
else:
    linear_equation_df = pd.DataFrame()
    print("[WARN] 没有 Linear Regression 模型，无法提取公式。")


# ============================================================
# 14. Best model summary
# ============================================================

print("\n================ 最优模型 ================")

best_row = model_summary_df.iloc[0]
display(best_row.to_frame().T)

best_model_name = best_row["model"]
best_model = trained_models[best_model_name]

print("best_model_name:", best_model_name)
print("best_model 来自重新加载的 pickle 模型。")
print("trained_models 中的所有模型均已从 pickle 文件重新加载。")

# ============================================================
# Recommendation System V2 - Pickle-loaded model version
#
# 接入当前 Metris 拉下来的 model_df / df_no_outlier，模型来自 pickle 加载后的 trained_models
#
# 从 SavedModels 读取本轮保存的 pickle 模型
# 不保存 Excel
# 只在 notebook 里 display 输出结果
#
# Methods:
#   1. Historical similar low-gas benchmark
#   2. Random Forest constrained optimization
#   3. Linear/Ridge constrained optimization
#   4. LightGBM constrained optimization, if available
#   5. Fused recommendation
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# 0. display fallback
# ============================================================

try:
    display
except NameError:
    def display(x):
        print(x)


# ============================================================
# 1. Define columns
# ============================================================

target_col = "熔炼炉B当前批次总气耗_PLC"

weight_col = "10#熔炼炉总投料重量(kg)"

# 真正推荐给工厂的 4 个可控变量
controllable_cols = [
    "10#熔炼炉固体料重量比例",
    "熔炼炉B当前批次等待时长_PLC",
    "熔炼炉B当前批次炉门打开次数_PLC",
    "熔炼炉B当前批次炉门打开时长_PLC",
]

# 不推荐，但是如果模型需要，会用历史参考值补上
melting_time_col = "熔炼炉B当前批次熔炼时间_PLC"

# 你现在这个数据集里没有生产时间这一列，所以这里仅作为保护项
production_time_col = "熔炼炉B当前批次生产时间_PLC"


# ============================================================
# 2. Check required objects from previous workflow
# ============================================================

if "df_no_outlier" not in globals():
    raise NameError(
        "当前环境里没有 df_no_outlier。"
        "请先运行前面的 prediction training workflow，生成 df_no_outlier。"
    )

if "trained_models" not in globals():
    raise NameError(
        "当前环境里没有 trained_models。"
        "请先运行前面的模型训练代码。"
    )

if "feature_cols" not in globals():
    raise NameError(
        "当前环境里没有 feature_cols。"
        "请先运行前面的模型训练代码。"
    )


df_reference = df_no_outlier.copy()

print("推荐系统使用的数据集 df_reference shape:", df_reference.shape)

print("\n当前 trained_models 里已有模型:")
for m in trained_models.keys():
    print(" -", m)


# ============================================================
# 3. Helper:
#    Get model feature names
# ============================================================

def get_model_feature_names(trained_model, fallback_feature_cols=None):
    """
    自动读取 sklearn Pipeline / model 训练时使用的特征名。

    如果模型里没有 feature_names_in_，则使用 fallback_feature_cols。
    """

    if hasattr(trained_model, "feature_names_in_"):
        return list(trained_model.feature_names_in_)

    if hasattr(trained_model, "named_steps"):
        for step_name, step in trained_model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    if fallback_feature_cols is not None:
        return list(fallback_feature_cols)

    raise ValueError(
        "无法从模型中读取 feature_names_in_，并且没有提供 fallback_feature_cols。"
    )


# ============================================================
# 4. Load models from memory instead of checkpoint
# ============================================================

def get_model_from_memory(
    trained_models,
    preferred_names,
    fallback_feature_cols,
    forbidden_cols=None
):
    """
    从 trained_models 字典里按优先级取模型。

    preferred_names:
        比如 ["Ridge Regression", "Linear Regression"]

    forbidden_cols:
        如果模型特征里包含不希望用于推荐的变量，就跳过。
        例如 production_time_col。
    """

    if forbidden_cols is None:
        forbidden_cols = []

    for model_name in preferred_names:

        if model_name not in trained_models:
            continue

        model = trained_models[model_name]
        model_feature_cols = get_model_feature_names(
            trained_model=model,
            fallback_feature_cols=fallback_feature_cols
        )

        has_forbidden = any(c in model_feature_cols for c in forbidden_cols)

        print("\n候选内存模型:", model_name)
        print("模型特征:")
        for c in model_feature_cols:
            print(" -", c)
        print("是否包含 forbidden cols:", has_forbidden)

        if has_forbidden:
            print("跳过该模型，因为它包含不建议用于推荐的变量。")
            continue

        print("最终选择模型:", model_name)

        return model_name, model, model_feature_cols

    raise ValueError(
        f"在 trained_models 中没有找到可用模型。候选模型: {preferred_names}"
    )


# Random Forest
rf_model_name, rf_model, rf_feature_cols = get_model_from_memory(
    trained_models=trained_models,
    preferred_names=["Random Forest"],
    fallback_feature_cols=feature_cols,
    forbidden_cols=[production_time_col]
)

# Linear model:
# 优先 Ridge Regression，如果你前面训练了 Ridge；
# 如果没有，就使用 Linear Regression。
linear_model_name, linear_model, linear_feature_cols = get_model_from_memory(
    trained_models=trained_models,
    preferred_names=["Ridge Regression", "Linear Regression"],
    fallback_feature_cols=feature_cols,
    forbidden_cols=[production_time_col]
)

# LightGBM:
# 如果前面环境没有 lightgbm，trained_models 里可能没有这个模型。
if "LightGBM" in trained_models:
    lgbm_model_name, lgbm_model, lgbm_feature_cols = get_model_from_memory(
        trained_models=trained_models,
        preferred_names=["LightGBM"],
        fallback_feature_cols=feature_cols,
        forbidden_cols=[production_time_col]
    )
    HAS_RECOMMEND_LGBM = True
else:
    lgbm_model_name = None
    lgbm_model = None
    lgbm_feature_cols = None
    HAS_RECOMMEND_LGBM = False
    print("\n[WARN] trained_models 中没有 LightGBM，推荐系统会跳过 LightGBM。")


print("\n================ 推荐系统最终使用的模型 ================")
print("Random Forest:", rf_model_name)
print("Linear model:", linear_model_name)
print("LightGBM:", lgbm_model_name if HAS_RECOMMEND_LGBM else "跳过")


# ============================================================
# 5. Basic column checks
# ============================================================

all_model_feature_cols = []

all_model_feature_cols.extend(rf_feature_cols)
all_model_feature_cols.extend(linear_feature_cols)

if HAS_RECOMMEND_LGBM:
    all_model_feature_cols.extend(lgbm_feature_cols)

required_cols = (
    [target_col, weight_col]
    + controllable_cols
    + list(set(all_model_feature_cols))
)

missing_cols = [
    c for c in required_cols
    if c not in df_reference.columns
]

if len(missing_cols) > 0:
    raise ValueError(
        f"df_reference / df_no_outlier 中缺少以下列，请检查前面数据提取或训练特征: {missing_cols}"
    )

print("\n需要的列都存在，可以继续。")


# ============================================================
# 6. Method 1:
#    Historical similar batch benchmark recommendation
# ============================================================

def historical_benchmark_recommendation(
    df_reference,
    total_weight,
    weight_col,
    target_col,
    controllable_cols,
    tolerance_ratio=0.05,
    low_gas_top_ratio=0.2
):
    """
    给定总投料量:

    1. 找总投料量相似的历史批次
    2. 按真实总气耗从低到高排序
    3. 取低气耗前 low_gas_top_ratio
    4. 输出 4 个可控变量的历史 benchmark 推荐范围
    """

    low_weight = total_weight * (1 - tolerance_ratio)
    high_weight = total_weight * (1 + tolerance_ratio)

    similar_df = df_reference[
        (df_reference[weight_col] >= low_weight)
        & (df_reference[weight_col] <= high_weight)
    ].copy()

    if len(similar_df) == 0:
        raise ValueError(
            f"没有找到总投料量在 {low_weight:.1f} 到 {high_weight:.1f} 之间的历史批次。"
            "可以尝试增大 tolerance_ratio，比如 0.08 或 0.10。"
        )

    similar_df = similar_df.sort_values(target_col, ascending=True)

    n_top = max(1, int(np.ceil(len(similar_df) * low_gas_top_ratio)))

    low_gas_df = similar_df.head(n_top).copy()

    summary_rows = []

    summary_cols = controllable_cols + [target_col]

    for col in summary_cols:
        summary_rows.append({
            "variable": col,
            "method": "historical_benchmark",
            "value_median": low_gas_df[col].median(),
            "value_mean": low_gas_df[col].mean(),
            "value_p10": low_gas_df[col].quantile(0.10),
            "value_p25": low_gas_df[col].quantile(0.25),
            "value_p75": low_gas_df[col].quantile(0.75),
            "value_p90": low_gas_df[col].quantile(0.90),
            "value_min": low_gas_df[col].min(),
            "value_max": low_gas_df[col].max(),
            "n_similar_batches": len(similar_df),
            "n_low_gas_batches": len(low_gas_df),
            "weight_low_bound": low_weight,
            "weight_high_bound": high_weight,
        })

    benchmark_summary = pd.DataFrame(summary_rows)

    return similar_df, low_gas_df, benchmark_summary


# ============================================================
# 7. Method 2:
#    Model-based constrained optimization recommendation
# ============================================================

def model_based_optimization_recommendation(
    trained_model,
    df_reference,
    similar_df,
    low_gas_df,
    total_weight,
    feature_cols,
    weight_col,
    target_col,
    controllable_cols,
    n_candidates=50000,
    top_n=50,
    random_state=42,
    range_low_q=0.05,
    range_high_q=0.95,
    range_source="similar",
    fixed_value_strategy="low_gas_median"
):
    """
    模型约束优化推荐。

    固定:
        - 总投料重量 = 工厂输入 total_weight

    优化:
        - 10#熔炼炉固体料重量比例
        - 熔炼炉B当前批次等待时长_PLC
        - 熔炼炉B当前批次炉门打开次数_PLC
        - 熔炼炉B当前批次炉门打开时长_PLC

    不推荐但模型需要的变量:
        - 比如熔炼时间_PLC
        - 用 low_gas_df 或 similar_df 的 median 自动补上
    """

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------
    # 1. Decide random search range source
    # ------------------------------------------------------------

    if range_source == "similar":
        range_base_df = similar_df.copy()
    elif range_source == "all":
        range_base_df = df_reference.copy()
    elif range_source == "low_gas":
        range_base_df = low_gas_df.copy()
    else:
        raise ValueError("range_source 只能是 'similar', 'all', 或 'low_gas'")

    # ------------------------------------------------------------
    # 2. Define historical search ranges
    # ------------------------------------------------------------

    range_rows = []
    ranges = {}

    for col in controllable_cols:

        low = range_base_df[col].quantile(range_low_q)
        high = range_base_df[col].quantile(range_high_q)

        if pd.isna(low) or pd.isna(high):
            raise ValueError(f"{col} 的搜索范围存在 NaN，请检查该列数据。")

        if low == high:
            print(f"警告: {col} 的 search_low 和 search_high 相同，候选值将固定为 {low}")

        ranges[col] = (low, high)

        range_rows.append({
            "variable": col,
            "range_source": range_source,
            "range_low_q": range_low_q,
            "range_high_q": range_high_q,
            "search_low": low,
            "search_high": high,
        })

    range_df = pd.DataFrame(range_rows)

    # ------------------------------------------------------------
    # 3. Generate candidate combinations
    # ------------------------------------------------------------

    candidates = pd.DataFrame()

    # 固定工厂输入总投料重量
    candidates[weight_col] = np.repeat(total_weight, n_candidates)

    # 只对 4 个可控变量做 random search
    for col in controllable_cols:

        low, high = ranges[col]

        if "次数" in col:
            low_int = int(np.floor(low))
            high_int = int(np.ceil(high))

            if low_int == high_int:
                candidates[col] = np.repeat(low_int, n_candidates)
            else:
                candidates[col] = rng.integers(
                    low=low_int,
                    high=high_int + 1,
                    size=n_candidates
                )
        else:
            if low == high:
                candidates[col] = np.repeat(low, n_candidates)
            else:
                candidates[col] = rng.uniform(
                    low=low,
                    high=high,
                    size=n_candidates
                )

    # ------------------------------------------------------------
    # 4. Fill model-required but not recommended variables
    # ------------------------------------------------------------

    fixed_feature_rows = []

    for col in feature_cols:

        if col in candidates.columns:
            continue

        if fixed_value_strategy == "low_gas_median":
            fixed_value = low_gas_df[col].median()
            fixed_source = "low_gas_median"

        elif fixed_value_strategy == "similar_median":
            fixed_value = similar_df[col].median()
            fixed_source = "similar_median"

        else:
            raise ValueError(
                "fixed_value_strategy 只能是 'low_gas_median' 或 'similar_median'"
            )

        if pd.isna(fixed_value):
            raise ValueError(
                f"模型需要变量 {col}，但是无法计算固定参考值。"
                "请检查该列在 low_gas_df / similar_df 中是否全是 NaN。"
            )

        candidates[col] = np.repeat(fixed_value, n_candidates)

        fixed_feature_rows.append({
            "variable": col,
            "fixed_value": fixed_value,
            "fixed_source": fixed_source,
            "reason": "model_required_but_not_recommended",
        })

    fixed_feature_df = pd.DataFrame(fixed_feature_rows)

    # ------------------------------------------------------------
    # 5. Ensure exact feature names and order
    # ------------------------------------------------------------

    missing_candidate_cols = [
        c for c in feature_cols
        if c not in candidates.columns
    ]

    if len(missing_candidate_cols) > 0:
        raise ValueError(
            f"候选数据中仍然缺少模型需要的特征列: {missing_candidate_cols}"
        )

    candidate_X = candidates[feature_cols].copy()

    # ------------------------------------------------------------
    # 6. Predict gas consumption
    # ------------------------------------------------------------

    candidates["predicted_gas"] = trained_model.predict(candidate_X)

    # ------------------------------------------------------------
    # 7. Select Top N lowest predicted gas candidates
    # ------------------------------------------------------------

    top_candidates = candidates.sort_values(
        "predicted_gas",
        ascending=True
    ).head(top_n).copy()

    # ------------------------------------------------------------
    # 8. Recommendation summary
    # ------------------------------------------------------------

    summary_rows = []

    fixed_cols = [
        c for c in feature_cols
        if c not in controllable_cols and c != weight_col
    ]

    summary_cols = controllable_cols + fixed_cols + ["predicted_gas"]

    for col in summary_cols:
        summary_rows.append({
            "variable": col,
            "method": "model_optimization",
            "value_median": top_candidates[col].median(),
            "value_mean": top_candidates[col].mean(),
            "value_p10": top_candidates[col].quantile(0.10),
            "value_p25": top_candidates[col].quantile(0.25),
            "value_p75": top_candidates[col].quantile(0.75),
            "value_p90": top_candidates[col].quantile(0.90),
            "value_min": top_candidates[col].min(),
            "value_max": top_candidates[col].max(),
            "n_candidates": n_candidates,
            "top_n": top_n,
        })

    recommendation_summary = pd.DataFrame(summary_rows)

    return (
        candidates,
        top_candidates,
        recommendation_summary,
        range_df,
        fixed_feature_df
    )


# ============================================================
# 8. Validate model recommendation against historical benchmark
# ============================================================

def validate_recommendation_against_benchmark(
    model_summary,
    benchmark_summary,
    controllable_cols
):
    """
    检查模型推荐的 4 个可控变量中位数，
    是否落在历史相似低气耗批次 p10~p90 区间内。
    """

    rows = []

    for col in controllable_cols:

        model_row = model_summary[
            model_summary["variable"] == col
        ].iloc[0]

        bench_row = benchmark_summary[
            benchmark_summary["variable"] == col
        ].iloc[0]

        model_median = model_row["value_median"]
        bench_p10 = bench_row["value_p10"]
        bench_p90 = bench_row["value_p90"]
        bench_median = bench_row["value_median"]

        in_benchmark_range = (
            model_median >= bench_p10
            and model_median <= bench_p90
        )

        rows.append({
            "variable": col,
            "model_recommended_median": model_median,
            "benchmark_median": bench_median,
            "benchmark_p10": bench_p10,
            "benchmark_p90": bench_p90,
            "model_median_in_benchmark_p10_p90": in_benchmark_range,
            "difference_vs_benchmark_median": model_median - bench_median,
        })

    return pd.DataFrame(rows)


# ============================================================
# 9. Build factory-facing final recommendation table
# ============================================================

def build_factory_recommendation_table(
    combined_summary,
    controllable_cols
):
    """
    只给工厂展示真正要推荐的 4 个变量。
    不展示熔炼时间作为推荐变量。
    """

    factory_table = combined_summary[
        combined_summary["variable"].isin(controllable_cols)
    ][
        [
            "variable",
            "model",
            "value_median",
            "value_p10",
            "value_p25",
            "value_p75",
            "value_p90",
            "value_min",
            "value_max",
        ]
    ].copy()

    factory_table = factory_table.rename(columns={
        "variable": "推荐变量",
        "model": "推荐方法",
        "value_median": "推荐值_中位数",
        "value_p10": "推荐范围_p10",
        "value_p25": "推荐范围_p25",
        "value_p75": "推荐范围_p75",
        "value_p90": "推荐范围_p90",
        "value_min": "Top候选最小值",
        "value_max": "Top候选最大值",
    })

    return factory_table


# ============================================================
# 10. Fuse available model recommendations
# ============================================================

def build_fused_recommendation_table(
    combined_summary,
    validation_df,
    controllable_cols,
    available_model_names
):
    """
    动态融合推荐逻辑。

    对每个可控变量:
    1. 检查各模型推荐中位数是否落在 historical benchmark p10~p90 内。
    2. 只融合 valid models。
    3. 如果所有模型都 invalid，则使用 Historical Benchmark median。
    4. 最后 clip 到 benchmark p10~p90。
    """

    # 根据你当前推荐系统的逻辑设置基础权重
    # 如果某个模型不存在，会自动忽略并重新归一化
    base_weights = {
        "Random Forest": 0.40,
        "LightGBM": 0.40,
        "Ridge Regression": 0.20,
        "Linear Regression": 0.20,
    }

    rows = []

    for col in controllable_cols:

        hist_row = combined_summary[
            (combined_summary["variable"] == col)
            & (combined_summary["model"] == "Historical Benchmark")
        ].iloc[0]

        hist_value = hist_row["value_median"]
        benchmark_p10 = hist_row["value_p10"]
        benchmark_p90 = hist_row["value_p90"]

        model_values = {}
        model_valid_flags = {}

        for model_name in available_model_names:

            model_row = combined_summary[
                (combined_summary["variable"] == col)
                & (combined_summary["model"] == model_name)
            ].iloc[0]

            model_value = model_row["value_median"]
            model_values[model_name] = model_value

            valid_flag = validation_df[
                (validation_df["variable"] == col)
                & (validation_df["model"] == model_name)
            ]["model_median_in_benchmark_p10_p90"].iloc[0]

            model_valid_flags[model_name] = valid_flag

        valid_models = [
            model_name for model_name in available_model_names
            if model_valid_flags[model_name]
        ]

        if len(valid_models) > 0:

            weight_sum = sum(
                base_weights.get(m, 1.0)
                for m in valid_models
            )

            normalized_weights = {
                m: base_weights.get(m, 1.0) / weight_sum
                for m in valid_models
            }

            fused_value_raw = sum(
                normalized_weights[m] * model_values[m]
                for m in valid_models
            )

            fusion_rule = (
                "Weighted average of valid models: "
                + ", ".join([
                    f"{m} weight={normalized_weights[m]:.2f}"
                    for m in valid_models
                ])
            )

        else:
            fused_value_raw = hist_value
            fusion_rule = (
                "All models outside benchmark range, "
                "use Historical Benchmark median"
            )

        fused_value_clipped = np.clip(
            fused_value_raw,
            benchmark_p10,
            benchmark_p90
        )

        was_clipped = fused_value_raw != fused_value_clipped

        row = {
            "推荐变量": col,
            "Historical_Benchmark_Median": hist_value,
            "Benchmark_p10": benchmark_p10,
            "Benchmark_p90": benchmark_p90,
            "Fusion_Rule": fusion_rule,
            "Fused_Recommendation_Raw": fused_value_raw,
            "Fused_Recommendation_Final": fused_value_clipped,
            "was_clipped": was_clipped,
        }

        for model_name in available_model_names:
            safe_name = model_name.replace(" ", "_")
            row[f"{safe_name}_Median"] = model_values[model_name]
            row[f"{safe_name}_in_Benchmark_Range"] = model_valid_flags[model_name]

        rows.append(row)

    fused_df = pd.DataFrame(rows)

    return fused_df


# ============================================================
# 11. Main function:
#     Compare benchmark vs RF vs Linear/Ridge vs optional LightGBM
# ============================================================

def run_recommendation_comparison(
    total_weight,
    df_reference,

    rf_model,
    linear_model,
    lgbm_model,

    rf_feature_cols,
    linear_feature_cols,
    lgbm_feature_cols,

    rf_model_name,
    linear_model_name,
    lgbm_model_name,

    weight_col,
    target_col,
    controllable_cols,

    tolerance_ratio=0.05,
    low_gas_top_ratio=0.2,
    n_candidates=50000,
    top_n=50,
    random_state=42,
    range_source="similar",
    fixed_value_strategy="low_gas_median"
):
    """
    对给定总投料量，比较:

    1. Historical Similar All Batches
    2. Historical Benchmark Top Low Gas
    3. Random Forest Optimization
    4. Linear/Ridge Optimization
    5. LightGBM Optimization, if available

    输出:
    - 工厂真正需要看的 4 个变量推荐
    - 各模型推荐是否落在 benchmark p10~p90
    - fixed variables 的使用值
    - 融合推荐
    """

    # ------------------------------------------------------------
    # Method 1: historical benchmark
    # ------------------------------------------------------------

    similar_df, low_gas_df, benchmark_summary = historical_benchmark_recommendation(
        df_reference=df_reference,
        total_weight=total_weight,
        weight_col=weight_col,
        target_col=target_col,
        controllable_cols=controllable_cols,
        tolerance_ratio=tolerance_ratio,
        low_gas_top_ratio=low_gas_top_ratio
    )

    print("\n相似批次数量:", len(similar_df))
    print("低气耗 benchmark 批次数量:", len(low_gas_df))

    # ------------------------------------------------------------
    # Method 2A: Random Forest optimization
    # ------------------------------------------------------------

    (
        rf_candidates,
        rf_top,
        rf_summary,
        search_range_df,
        rf_fixed_feature_df
    ) = model_based_optimization_recommendation(
        trained_model=rf_model,
        df_reference=df_reference,
        similar_df=similar_df,
        low_gas_df=low_gas_df,
        total_weight=total_weight,
        feature_cols=rf_feature_cols,
        weight_col=weight_col,
        target_col=target_col,
        controllable_cols=controllable_cols,
        n_candidates=n_candidates,
        top_n=top_n,
        random_state=random_state,
        range_source=range_source,
        fixed_value_strategy=fixed_value_strategy
    )

    rf_summary["model"] = rf_model_name

    # ------------------------------------------------------------
    # Method 2B: Linear/Ridge optimization
    # ------------------------------------------------------------

    (
        linear_candidates,
        linear_top,
        linear_summary,
        _,
        linear_fixed_feature_df
    ) = model_based_optimization_recommendation(
        trained_model=linear_model,
        df_reference=df_reference,
        similar_df=similar_df,
        low_gas_df=low_gas_df,
        total_weight=total_weight,
        feature_cols=linear_feature_cols,
        weight_col=weight_col,
        target_col=target_col,
        controllable_cols=controllable_cols,
        n_candidates=n_candidates,
        top_n=top_n,
        random_state=random_state + 1,
        range_source=range_source,
        fixed_value_strategy=fixed_value_strategy
    )

    linear_summary["model"] = linear_model_name

    # ------------------------------------------------------------
    # Method 2C: LightGBM optimization, optional
    # ------------------------------------------------------------

    has_lgbm = lgbm_model is not None

    if has_lgbm:
        (
            lgbm_candidates,
            lgbm_top,
            lgbm_summary,
            _,
            lgbm_fixed_feature_df
        ) = model_based_optimization_recommendation(
            trained_model=lgbm_model,
            df_reference=df_reference,
            similar_df=similar_df,
            low_gas_df=low_gas_df,
            total_weight=total_weight,
            feature_cols=lgbm_feature_cols,
            weight_col=weight_col,
            target_col=target_col,
            controllable_cols=controllable_cols,
            n_candidates=n_candidates,
            top_n=top_n,
            random_state=random_state + 2,
            range_source=range_source,
            fixed_value_strategy=fixed_value_strategy
        )

        lgbm_summary["model"] = lgbm_model_name

    else:
        lgbm_candidates = pd.DataFrame()
        lgbm_top = pd.DataFrame()
        lgbm_summary = pd.DataFrame()
        lgbm_fixed_feature_df = pd.DataFrame()

    # ------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------

    benchmark_summary_for_compare = benchmark_summary.copy()
    benchmark_summary_for_compare["model"] = "Historical Benchmark"

    summary_list = [
        benchmark_summary_for_compare,
        rf_summary,
        linear_summary,
    ]

    available_model_names = [
        rf_model_name,
        linear_model_name,
    ]

    if has_lgbm:
        summary_list.append(lgbm_summary)
        available_model_names.append(lgbm_model_name)

    combined_summary = pd.concat(
        summary_list,
        ignore_index=True
    )

    # ------------------------------------------------------------
    # Validation against benchmark
    # ------------------------------------------------------------

    validation_list = []

    rf_validation = validate_recommendation_against_benchmark(
        model_summary=rf_summary,
        benchmark_summary=benchmark_summary,
        controllable_cols=controllable_cols
    )
    rf_validation["model"] = rf_model_name
    validation_list.append(rf_validation)

    linear_validation = validate_recommendation_against_benchmark(
        model_summary=linear_summary,
        benchmark_summary=benchmark_summary,
        controllable_cols=controllable_cols
    )
    linear_validation["model"] = linear_model_name
    validation_list.append(linear_validation)

    if has_lgbm:
        lgbm_validation = validate_recommendation_against_benchmark(
            model_summary=lgbm_summary,
            benchmark_summary=benchmark_summary,
            controllable_cols=controllable_cols
        )
        lgbm_validation["model"] = lgbm_model_name
        validation_list.append(lgbm_validation)

    validation_df = pd.concat(
        validation_list,
        ignore_index=True
    )

    # ------------------------------------------------------------
    # Method-level gas comparison
    # ------------------------------------------------------------

    method_score_rows = []

    method_score_rows.append({
        "method": "Historical Similar All Batches",
        "gas_metric_type": "actual_all_similar_batches",
        "gas_median": similar_df[target_col].median(),
        "gas_mean": similar_df[target_col].mean(),
        "gas_p10": similar_df[target_col].quantile(0.10),
        "gas_p90": similar_df[target_col].quantile(0.90),
        "n_records": len(similar_df),
    })

    method_score_rows.append({
        "method": "Historical Benchmark Top Low Gas",
        "gas_metric_type": "actual_low_gas_batches",
        "gas_median": low_gas_df[target_col].median(),
        "gas_mean": low_gas_df[target_col].mean(),
        "gas_p10": low_gas_df[target_col].quantile(0.10),
        "gas_p90": low_gas_df[target_col].quantile(0.90),
        "n_records": len(low_gas_df),
    })

    method_score_rows.append({
        "method": f"{rf_model_name} Optimization",
        "gas_metric_type": "predicted_top_candidates",
        "gas_median": rf_top["predicted_gas"].median(),
        "gas_mean": rf_top["predicted_gas"].mean(),
        "gas_p10": rf_top["predicted_gas"].quantile(0.10),
        "gas_p90": rf_top["predicted_gas"].quantile(0.90),
        "n_records": len(rf_top),
    })

    method_score_rows.append({
        "method": f"{linear_model_name} Optimization",
        "gas_metric_type": "predicted_top_candidates",
        "gas_median": linear_top["predicted_gas"].median(),
        "gas_mean": linear_top["predicted_gas"].mean(),
        "gas_p10": linear_top["predicted_gas"].quantile(0.10),
        "gas_p90": linear_top["predicted_gas"].quantile(0.90),
        "n_records": len(linear_top),
    })

    if has_lgbm:
        method_score_rows.append({
            "method": f"{lgbm_model_name} Optimization",
            "gas_metric_type": "predicted_top_candidates",
            "gas_median": lgbm_top["predicted_gas"].median(),
            "gas_mean": lgbm_top["predicted_gas"].mean(),
            "gas_p10": lgbm_top["predicted_gas"].quantile(0.10),
            "gas_p90": lgbm_top["predicted_gas"].quantile(0.90),
            "n_records": len(lgbm_top),
        })

    method_score_df = pd.DataFrame(method_score_rows)

    # ------------------------------------------------------------
    # Improvement vs all similar batches
    # ------------------------------------------------------------

    baseline_gas_median = method_score_df.loc[
        method_score_df["method"] == "Historical Similar All Batches",
        "gas_median"
    ].iloc[0]

    baseline_gas_mean = method_score_df.loc[
        method_score_df["method"] == "Historical Similar All Batches",
        "gas_mean"
    ].iloc[0]

    method_score_df["improvement_vs_all_similar_median_%"] = (
        baseline_gas_median - method_score_df["gas_median"]
    ) / baseline_gas_median * 100

    method_score_df["improvement_vs_all_similar_mean_%"] = (
        baseline_gas_mean - method_score_df["gas_mean"]
    ) / baseline_gas_mean * 100

    # ------------------------------------------------------------
    # Factory-facing recommendation table
    # ------------------------------------------------------------

    factory_recommendation_table = build_factory_recommendation_table(
        combined_summary=combined_summary,
        controllable_cols=controllable_cols
    )

    # ------------------------------------------------------------
    # Fused recommendation table
    # ------------------------------------------------------------

    fused_recommendation_table = build_fused_recommendation_table(
        combined_summary=combined_summary,
        validation_df=validation_df,
        controllable_cols=controllable_cols,
        available_model_names=available_model_names
    )

    return {
        "similar_batches": similar_df,
        "low_gas_batches": low_gas_df,
        "benchmark_summary": benchmark_summary,
        "search_range": search_range_df,

        "rf_top_candidates": rf_top,
        "linear_top_candidates": linear_top,
        "lgbm_top_candidates": lgbm_top,

        "combined_summary": combined_summary,
        "validation": validation_df,
        "method_score": method_score_df,
        "factory_recommendation": factory_recommendation_table,
        "fused_recommendation": fused_recommendation_table,

        "rf_fixed_features": rf_fixed_feature_df,
        "linear_fixed_features": linear_fixed_feature_df,
        "lgbm_fixed_features": lgbm_fixed_feature_df,

        "available_model_names": available_model_names,
    }


# ============================================================
# 12. Example call
#     工厂输入一个总投料量，例如 86000 kg
# ============================================================

factory_input_total_weight = 86000

result = run_recommendation_comparison(
    total_weight=factory_input_total_weight,
    df_reference=df_reference,

    rf_model=rf_model,
    linear_model=linear_model,
    lgbm_model=lgbm_model,

    rf_feature_cols=rf_feature_cols,
    linear_feature_cols=linear_feature_cols,
    lgbm_feature_cols=lgbm_feature_cols,

    rf_model_name=rf_model_name,
    linear_model_name=linear_model_name,
    lgbm_model_name=lgbm_model_name,

    weight_col=weight_col,
    target_col=target_col,
    controllable_cols=controllable_cols,

    tolerance_ratio=0.05,
    low_gas_top_ratio=0.2,
    n_candidates=50000,
    top_n=50,
    random_state=42,
    range_source="similar",
    fixed_value_strategy="low_gas_median"
)


# ============================================================
# 13. Display key outputs
# ============================================================

print("\n================ 方法整体气耗对比 ================")
display(result["method_score"])

print("\n================ 给工厂看的推荐结果：只包含 4 个可控变量 ================")
display(result["factory_recommendation"])

print("\n================ 融合后的最终推荐结果 ================")
display(result["fused_recommendation"])

print("\n================ 各方法推荐变量汇总 ================")
display(
    result["combined_summary"][
        result["combined_summary"]["variable"].isin(
            controllable_cols + ["predicted_gas", target_col]
        )
    ].sort_values(["variable", "model"])
)

print("\n================ 模型推荐是否落在历史低气耗范围内 ================")
display(result["validation"])

print("\n================ Random Forest 自动固定的非推荐变量 ================")
display(result["rf_fixed_features"])

print(f"\n================ {linear_model_name} 自动固定的非推荐变量 ================")
display(result["linear_fixed_features"])

if HAS_RECOMMEND_LGBM:
    print("\n================ LightGBM 自动固定的非推荐变量 ================")
    display(result["lgbm_fixed_features"])

print("\n================ Random Forest Top Candidates ================")
display(result["rf_top_candidates"].head(10))

print(f"\n================ {linear_model_name} Top Candidates ================")
display(result["linear_top_candidates"].head(10))

if HAS_RECOMMEND_LGBM:
    print("\n================ LightGBM Top Candidates ================")
    display(result["lgbm_top_candidates"].head(10))

print("\n================ 历史相似低气耗批次 ================")
display(result["low_gas_batches"].head(10))

print("\n当前没有保存任何 Excel 文件。所有结果都在 result 字典里。")