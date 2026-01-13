import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path
from db import get_connection
from psycopg2.extras import execute_batch
import argparse

# 숫자 자릿수 정제
def clamp_numeric(val, max_abs=9999.99, scale=2):
    if val is None or pd.isna(val):
        return None
    val = round(float(val), scale)
    if val > max_abs:
        return max_abs
    if val < -max_abs:
        return -max_abs
    return val

def parse_args():
    parser = argparse.ArgumentParser(description="AI analysis batch")

    parser.add_argument(
        "--start-date",
        required=False,
        help="analysis start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        required=False,
        help="analysis end date (YYYY-MM-DD)"
    )
    parser.add_argument("--schema", required=False)
    parser.add_argument(
        "--blck-cd",
        required=False,
        help="specific block code"
    )

    return parser.parse_args()

import os
print("DB_HOST =", os.getenv("DB_HOST"))
print("DB_PORT =", os.getenv("DB_PORT"))
print("DB_NAME =", os.getenv("DB_NAME"))
print("DB_USER =", os.getenv("DB_USER"))

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")

IS_BATCH = True

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

INPUT_FILE = INPUT_DIR / "BY7a.csv"
OUTPUT_FILE = OUTPUT_DIR / "BY7-a_result.csv"

np.random.seed(42)
tf.random.set_seed(42)

# 데이터 불러오기
# 기존 엑셀 읽기
#data = pd.read_csv(INPUT_FILE)

def load_source_data(start_dt, end_dt, blck_cd=None):
    sql = f"""
        SELECT
            dt as "time",
            blck_cd,
            flow,
            temp as "temperature",
            rcvr_yn AS "leak_recovery"
        FROM {SCHEMA}.ai_anals_input
        WHERE dt >= %s and dt <= %s
    """

    params = [start_dt, end_dt]

    if blck_cd:
        sql+=" AND blck_cd =  %s"
        params.append(blck_cd)

    sql += " ORDER BY blck_cd, dt"

    start_dt = pd.to_datetime(start_dt)
    end_dt = pd.to_datetime(end_dt)
    conn = get_connection()
    try:
        df = pd.read_sql(sql, conn, params=params)
    finally:
        conn.close()

    return df

def save_results(df):
    sql = f"""
        merge into {SCHEMA}.ai_anals_output t
        using (
            select
                %s::timestamp as time,
                %s::varchar as blck_cd,
                %s::numeric as actual_flow,
                %s::numeric as predicted_flow,
                %s::numeric as weighted_error
        ) s
        on (t.dt = s.time and t.blck_cd = s.blck_cd)
        when matched then
            update set
                actual_flow = s.actual_flow,
                predicted_flow = s.predicted_flow,
                weighted_error = s.weighted_error
        when not matched then
            insert (dt, blck_cd, actual_flow, predicted_flow, weighted_error)
            values (s.time, s.blck_cd, s.actual_flow, s.predicted_flow, s.weighted_error)
    """

    rows = df[['Time', 'blck_cd', 'Actual Flow', 'Predicted Flow', 'Weighted Error']].values.tolist()

    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            execute_batch(cur, sql, rows)

    conn.close()

# db가져오기
args = parse_args()

#START_DATE = args.start_date
#END_DATE = args.end_date
#SCHEMA = args.schema
#BLCK_CD = args.blck_cd

START_DATE = "2024-01-01"
END_DATE = "2025-08-13"
SCHEMA = "by_schema"
BLCK_CD = "BL000045"

data = load_source_data(START_DATE, END_DATE, BLCK_CD)

data = data.rename(columns={
    'leak_recovery': 'leak_recovery'
})

# 데이터 전처리 함수
def preprocess_data(data):
    data['time'] = pd.to_datetime(data['time'])
    data['hour'] = data['time'].dt.hour
    data['day_of_week'] = data['time'].dt.dayofweek
    data['month'] = data['time'].dt.month
    data['date'] = data['time'].dt.date

    data = data[
        ['time', 'blck_cd', 'flow', 'temperature', 'leak_recovery', 'hour', 'day_of_week', 'month', 'date']
    ]

    return data


data = preprocess_data(data)

# 데이터 분할
split_time = pd.to_datetime(END_DATE) - pd.Timedelta(days=1)
train_data = data
test_data = data


# 데이터 스케일링
def scale_features(train_data, test_data):
    scaler = MinMaxScaler()
    scaled_train = train_data.copy()
    scaled_test = test_data.copy()

    scaled_train[['flow', 'temperature', 'leak_recovery']] = scaler.fit_transform(
        train_data[['flow', 'temperature', 'leak_recovery']])
    scaled_test[['flow', 'temperature', 'leak_recovery']] = scaler.transform(
        test_data[['flow', 'temperature', 'leak_recovery']])

    return scaled_train, scaled_test, scaler


scaled_train_data, scaled_test_data, scaler = scale_features(train_data, test_data)


# 시퀀스 생성
def create_sequences(data, sequence_length=24):
    all_sequences = []
    all_targets = []
    all_times = []
    all_blck_cds = []

    for blck_cd, g in data.groupby("blck_cd"):
        g = g.sort_values("time").reset_index(drop=True)

        if len(g) <= sequence_length:
            continue

        for i in range(len(g) - sequence_length):
            seq = g.iloc[i:i + sequence_length]
            target = g.iloc[i + sequence_length]['flow']

            all_sequences.append(
                seq[['flow', 'temperature', 'hour', 'day_of_week', 'month', 'leak_recovery']].values
            )
            all_targets.append(target)
            all_times.append(g.iloc[i + sequence_length]['time'])
            all_blck_cds.append(blck_cd)

    return (
        np.array(all_sequences),
        np.array(all_targets),
        pd.Series(all_times),
        pd.Series(all_blck_cds)
    )

sequence_length = 24
X_train, y_train, times_train, blck_cds_train = create_sequences(scaled_train_data, sequence_length)
X_test, y_test, times_test, blck_cds_test = create_sequences(scaled_test_data, sequence_length)

dates_train = pd.to_datetime(times_train).dt.date
dates_test = pd.to_datetime(times_test).dt.date


print("X_train shape:", X_train.shape)
print(
    data.groupby("blck_cd")
        .size()
        .sort_values()
)

# 누수 복구 데이터 증강
def augment_leak_recovery_data(data, sequence_length):
    augmented_data = data.copy()

    leak_recovery_indices = augmented_data[augmented_data['leak_recovery'] == 1].index

    for idx in leak_recovery_indices:
        start_idx = max(idx - (sequence_length * 24 * 7), 0)
        end_idx = min(idx + (sequence_length * 24 * 7), len(augmented_data))

        pre_recovery_window = augmented_data.iloc[start_idx:idx]
        post_recovery_window = augmented_data.iloc[idx:end_idx]

        pre_recovery_min_avg = pre_recovery_window['flow'].min()
        post_recovery_min_avg = post_recovery_window['flow'].min()

        flow_difference = pre_recovery_min_avg - post_recovery_min_avg
        if flow_difference > 0:
            post_recovery_window['flow'] += flow_difference

        augmented_data = pd.concat([augmented_data, post_recovery_window], ignore_index=True)

    return augmented_data


# 증강된 데이터 생성
augmented_train_data = augment_leak_recovery_data(train_data, sequence_length)

# 증강된 데이터 스케일링 및 시퀀스 생성
augmented_scaled_train_data, _, _ = scale_features(augmented_train_data, test_data)
X_augmented_train, y_augmented_train, _, _ = create_sequences(augmented_scaled_train_data, sequence_length)


# 모델 정의
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mean_squared_error')
    return model


# 모델 학습 설정
model = build_model((sequence_length, X_train.shape[2]))

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 학습
history = model.fit(
    X_augmented_train, y_augmented_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop]
)

# 예측 수행
y_pred = model.predict(X_test)


# 스케일 역변환
def inverse_scale(scaler, scaled_values):
    expanded = np.concatenate([scaled_values, np.zeros((scaled_values.shape[0], 2))], axis=1)
    inversed = scaler.inverse_transform(expanded)[:, 0]
    return inversed


y_train_inversed = inverse_scale(scaler, y_train.reshape(-1, 1))
y_test_inversed = inverse_scale(scaler, y_test.reshape(-1, 1))
y_pred_inversed = inverse_scale(scaler, y_pred)

# 결과 데이터프레임 생성
results = pd.DataFrame({
    'Time': times_test.reset_index(drop=True),
    'Date': dates_test.reset_index(drop=True),
    'blck_cd': blck_cds_test.reset_index(drop=True),
    'Actual Flow': y_test_inversed,
    'Predicted Flow': y_pred_inversed
})


# 날짜별 평가지표 계산
def calculate_datewise_metrics(results):
    datewise_metrics = {}

    for date, group in results.groupby('Date'):
        actuals = group['Actual Flow'].values
        predictions = group['Predicted Flow'].values
        hours = group['Time'].dt.hour.values

        mask = actuals != 0
        if not np.any(mask):
            datewise_metrics[date] = 0.0
            continue

        actuals = actuals[mask]
        predictions = predictions[mask]
        hours = hours[mask]

        percentage_errors = np.abs((actuals - predictions) / actuals) * 100

        # 혹시라도 안전장치(필요하면 유지)
        if percentage_errors.size == 0:
            datewise_metrics[date] = 0.0
            continue

        weights = np.full(percentage_errors.shape, 0.2 / 21, dtype=float)

        peak_hours = [2, 3, 4]
        peak_mask = np.isin(hours, list(peak_hours))

        weights[peak_mask] = 0.8 / 3

        weighted_error = float(np.sum(percentage_errors * weights))
        datewise_metrics[date] = weighted_error

    return pd.DataFrame({
        'Date': list(datewise_metrics.keys()),
        'Weighted Error': list(datewise_metrics.values())
    })


metrics_df = calculate_datewise_metrics(results)

# 결과 저장
final_results = results.merge(metrics_df, on='Date', how='left')

# 자릿수 수정
final_results["Actual Flow"] = final_results["Actual Flow"].apply(lambda v: clamp_numeric(v, scale=1))
final_results["Predicted Flow"] = final_results["Predicted Flow"].apply(lambda v: clamp_numeric(v, scale=1))
final_results["Weighted Error"] = final_results["Weighted Error"].apply(lambda v: clamp_numeric(v, scale=2))

#final_results.to_csv(OUTPUT_FILE, index=False)

#print("예측 결과 및 날짜별 평가지표가 포함된 CSV 파일이 생성되었습니다.")


# 특정 날짜의 실제값과 예측값을 비교하는 함수
def plot_single_day_comparison(results, metrics_df, specific_date):
    specific_date_data = results[results['Date'] == specific_date]
    specific_metrics = metrics_df[metrics_df['Date'] == specific_date]['Weighted Error'].values[0]

    plt.figure(figsize=(10, 6))
    plt.plot(specific_date_data['Time'], specific_date_data['Actual Flow'], label='Actual Flow', marker='o')
    plt.plot(specific_date_data['Time'], specific_date_data['Predicted Flow'], label='Predicted Flow', marker='x')

    plt.title(f'Flow Comparison on {specific_date}')
    plt.xlabel('Time')
    plt.ylabel('Flow')
    plt.legend()

    plt.text(0.95, 0.01, f"Weighted Error: {specific_metrics:.2f}%",
             verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes,
             color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 특정 날짜 포함 7일간의 실제값과 예측값을 비교하는 함수
def plot_seven_days_comparison(results, metrics_df, specific_date):
    start_date = pd.to_datetime(specific_date) - pd.Timedelta(days=7)
    seven_days_data = results[(results['Date'] >= start_date.date()) & (results['Date'] <= specific_date)]

    plt.figure(figsize=(12, 6))
    plt.plot(seven_days_data['Time'], seven_days_data['Actual Flow'], label='Actual Flow', marker='o')
    plt.plot(seven_days_data['Time'], seven_days_data['Predicted Flow'], label='Predicted Flow', marker='x')

    plt.title(f'Flow Comparison from {start_date.date()} to {specific_date}')
    plt.xlabel('Time')
    plt.ylabel('Flow')
    plt.legend()

    unique_dates = seven_days_data['Date'].unique()
    for date in unique_dates:
        error = metrics_df[metrics_df['Date'] == date]['Weighted Error'].values[0]
        date_data = seven_days_data[seven_days_data['Date'] == date]
        middle_time = date_data['Time'].iloc[len(date_data) // 2]
        plt.text(middle_time, plt.gca().get_ylim()[0] + 0.05 * (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]),
                 f"{error:.2f}%", color='red', fontsize=10,
                 verticalalignment='bottom', horizontalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.8))

    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 날짜별 비교 그래프
if not IS_BATCH:
    specific_date = pd.to_datetime('2024-03-25').date()
    plot_single_day_comparison(results, metrics_df, specific_date)
    plot_seven_days_comparison(results, metrics_df, specific_date)

save_results(final_results)
print(f"[RESULT] SUCCESS rows={len(final_results)}")