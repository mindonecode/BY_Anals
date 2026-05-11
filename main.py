import os
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import holidays
import joblib

from db import get_connection
from psycopg2.extras import execute_batch


# =========================
# 기본 설정
# =========================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"


# =========================
# Argument
# =========================

def parse_args():

    parser = argparse.ArgumentParser(
        description="AI Leak Analysis Prediction"
    )

    parser.add_argument(
        "--schema",
        required=True,
        help="database schema"
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="analysis start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        required=True,
        help="analysis end date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--blck-cd",
        required=False,
        help="specific block code"
    )

    return parser.parse_args()


# =========================
# 숫자 자릿수 정제
# =========================

def clamp_numeric(val, max_abs=9999.99, scale=2):

    if val is None or pd.isna(val):
        return None

    val = round(float(val), scale)

    if val > max_abs:
        return max_abs

    if val < -max_abs:
        return -max_abs

    return val


# =========================
# DB Load
# =========================

def load_source_data(schema, start_dt, end_dt, blck_cd=None):

    sql = f"""
        SELECT
            dt as "time",
            blck_cd,
            flow,
            temp as "temperature",
            rcvr_yn AS "leak_recovery"
        FROM {schema}.ai_anals_input
        WHERE dt >= %s
          AND dt <= %s
    """

    params = [start_dt, end_dt]

    if blck_cd:
        sql += """
          AND blck_cd = %s
        """
        params.append(blck_cd)

    sql += """
        ORDER BY blck_cd, dt
    """

    conn = get_connection()

    try:
        df = pd.read_sql(
            sql,
            conn,
            params=params
        )
    finally:
        conn.close()

    return df


# =========================
# 전처리
# =========================

def preprocess_data(data):

    data = data.copy()

    data['time'] = pd.to_datetime(data['time'])

    data['hour'] = data['time'].dt.hour
    data['day_of_week'] = data['time'].dt.dayofweek
    data['month'] = data['time'].dt.month
    data['date'] = data['time'].dt.date

    data['is_weekend'] = (
        data['day_of_week'] >= 5
    ).astype(int)

    kr_holidays = holidays.KR()

    data['is_holiday'] = data['date'].apply(
        lambda x: 1 if x in kr_holidays else 0
    )

    daily_temp = data.groupby(
        ['blck_cd', 'date']
    )['temperature'].agg(
        tmax='max',
        tmin='min'
    ).reset_index()

    data = data.merge(
        daily_temp,
        on=['blck_cd', 'date'],
        how='left'
    )

    data = data[
        [
            'time',
            'blck_cd',
            'flow',
            'temperature',
            'leak_recovery',
            'tmax',
            'tmin',
            'hour',
            'day_of_week',
            'is_weekend',
            'is_holiday',
            'month',
            'date'
        ]
    ]

    return data


# =========================
# 스케일러 적용
# =========================

def apply_scaler(data, scaler):

    scaled_data = data.copy()

    scaled_data[
        ['flow', 'tmax', 'tmin']
    ] = scaler.transform(
        data[
            ['flow', 'tmax', 'tmin']
        ]
    )

    return scaled_data


# =========================
# 직전 7일 시퀀스 생성
# =========================

def create_sequences_prev_7_days(data):

    sequences = []
    targets = []
    times = []
    dates = []
    blck_cds = []

    daily_sequences = {}

    for (blck_cd, current_date), group in data.groupby(
        ['blck_cd', 'date']
    ):

        group = group.sort_values('time').copy()

        group['hour_key'] = group[
            'time'
        ].dt.strftime('%Y-%m-%d %H')

        group = group.drop_duplicates(
            subset=['hour_key']
        )

        group = group.drop(
            columns=['hour_key']
        )

        if len(group) == 24:

            daily_sequences[
                (blck_cd, current_date)
            ] = group[
                [
                    'flow',
                    'tmax',
                    'tmin',
                    'hour',
                    'day_of_week',
                    'is_weekend',
                    'is_holiday',
                    'month'
                ]
            ].values

    for _, row in data.iterrows():

        target_time = row['time']
        target_date = row['date']
        target_hour = row['hour']
        target_flow = row['flow']
        blck_cd = row['blck_cd']

        valid_7_days = True
        prev_7_days_seqs = []

        for d in range(7, 0, -1):

            p_date = target_date - pd.Timedelta(days=d)
            key = (blck_cd, p_date)

            if key in daily_sequences:

                prev_7_days_seqs.append(
                    daily_sequences[key]
                )

            else:

                valid_7_days = False
                break

        if valid_7_days:

            combined_prev_7_days = np.vstack(
                prev_7_days_seqs
            )

            scaled_target_hour = target_hour / 23.0

            target_hour_array = np.full(
                (168, 1),
                scaled_target_hour
            )

            seq_with_target_hint = np.hstack(
                (
                    combined_prev_7_days,
                    target_hour_array
                )
            )

            sequences.append(seq_with_target_hint)
            targets.append(target_flow)
            times.append(target_time)
            dates.append(target_date)
            blck_cds.append(blck_cd)

    return (
        np.array(sequences),
        np.array(targets),
        pd.Series(times).reset_index(drop=True),
        pd.Series(dates).reset_index(drop=True),
        pd.Series(blck_cds).reset_index(drop=True)
    )


# =========================
# 역스케일링
# =========================

def inverse_scale_flow(scaler, scaled_values):

    expanded = np.concatenate(
        [
            scaled_values,
            np.zeros((scaled_values.shape[0], 2))
        ],
        axis=1
    )

    inversed = scaler.inverse_transform(
        expanded
    )[:, 0]

    return inversed


# =========================
# 일자별 가중 오차율 계산
# =========================

def calculate_datewise_metrics(results):

    datewise_metrics = {}

    for (blck_cd, date), group in results.groupby(
        ['blck_cd', 'Date']
    ):

        actuals = group['Actual Flow'].values
        predictions = group['Predicted Flow'].values
        hours = group['Time'].dt.hour.values

        # 고객 원본 기준:
        # 실제 유량이 예측 유량보다 큰 경우만 오차 계산
        mask = actuals > predictions

        if not np.any(mask):
            datewise_metrics[(blck_cd, date)] = 0.0
            continue

        actuals = actuals[mask]
        predictions = predictions[mask]
        hours = hours[mask]

        valid_mask = actuals != 0

        if not np.any(valid_mask):
            datewise_metrics[(blck_cd, date)] = 0.0
            continue

        actuals = actuals[valid_mask]
        predictions = predictions[valid_mask]
        hours = hours[valid_mask]

        percentage_errors = np.abs(
            (actuals - predictions) / actuals
        ) * 100

        if percentage_errors.size == 0:
            datewise_metrics[(blck_cd, date)] = 0.0
            continue

        weights = np.full(
            percentage_errors.shape,
            0.2 / 21,
            dtype=float
        )

        peak_hours = [2, 3, 4]
        peak_mask = np.isin(hours, peak_hours)

        weights[peak_mask] = 0.8 / 3

        weighted_error = float(
            np.sum(percentage_errors * weights)
        )

        datewise_metrics[(blck_cd, date)] = weighted_error

    return pd.DataFrame([
        {
            'blck_cd': k[0],
            'Date': k[1],
            'Weighted Error': v
        }
        for k, v in datewise_metrics.items()
    ])


# =========================
# 누수 가능성 판단
# =========================

def add_leak_probability_with_reasoning(metrics_df):

    metrics_df = metrics_df.sort_values(
        ['blck_cd', 'Date']
    ).reset_index(drop=True)

    metrics_df['Leak Probability'] = '데이터부족'
    metrics_df['Reasoning'] = '과거 데이터 부족으로 분석할 수 없습니다.'
    metrics_df['Upper Bound'] = np.nan
    metrics_df['Is Anomaly'] = False

    for idx, row in metrics_df.iterrows():

        blck_cd = row['blck_cd']
        current_date = row['Date']
        current_error = row['Weighted Error']

        start_date = current_date - datetime.timedelta(days=90)
        end_date = current_date - datetime.timedelta(days=1)

        past_3_months = metrics_df[
            (metrics_df['blck_cd'] == blck_cd)
            & (metrics_df['Date'] >= start_date)
            & (metrics_df['Date'] <= end_date)
        ]

        if len(past_3_months) == 0:
            continue

        errors = past_3_months[
            'Weighted Error'
        ].dropna().values

        if len(errors) == 0:
            continue

        threshold_90th = np.percentile(
            errors,
            90
        )

        filtered_errors = errors[
            errors <= threshold_90th
        ]

        if len(filtered_errors) == 0:
            continue

        upper_bound = float(
            np.mean(filtered_errors)
            + np.std(filtered_errors)
        )

        metrics_df.at[idx, 'Upper Bound'] = upper_bound

        if current_error > upper_bound:

            metrics_df.at[idx, 'Leak Probability'] = '높음'
            metrics_df.at[idx, 'Is Anomaly'] = True
            metrics_df.at[idx, 'Reasoning'] = (
                f"당일 오차율({current_error:.1f}%)이 "
                f"과거 3개월 정상 상한선({upper_bound:.1f}%)을 초과했습니다."
            )

        else:

            metrics_df.at[idx, 'Leak Probability'] = '낮음'
            metrics_df.at[idx, 'Is Anomaly'] = False
            metrics_df.at[idx, 'Reasoning'] = (
                f"당일 오차율({current_error:.1f}%)이 "
                f"정상 범위({upper_bound:.1f}% 이하) 내에 있어 안정적입니다."
            )

    return metrics_df


# =========================
# 결과 저장
# =========================

def save_results(schema, df):

    if df.empty:
        return 0

    sql = f"""
        MERGE INTO {schema}.ai_anals_output t
        USING (
            SELECT
                %s::timestamp AS dt,
                %s::varchar AS blck_cd,
                %s::numeric AS actual_flow,
                %s::numeric AS predicted_flow,
                %s::numeric AS weighted_error,
                %s::varchar AS leak_probability,
                %s::text AS reasoning,
                %s::numeric AS upper_bound,
                %s::boolean AS is_anomaly
        ) s
        ON (
            t.dt = s.dt
            AND t.blck_cd = s.blck_cd
        )
        WHEN MATCHED THEN
            UPDATE SET
                actual_flow = s.actual_flow,
                predicted_flow = s.predicted_flow,
                weighted_error = s.weighted_error,
                leak_probability = s.leak_probability,
                reasoning = s.reasoning,
                upper_bound = s.upper_bound,
                is_anomaly = s.is_anomaly
        WHEN NOT MATCHED THEN
            INSERT (
                dt,
                blck_cd,
                actual_flow,
                predicted_flow,
                weighted_error,
                leak_probability,
                reasoning,
                upper_bound,
                is_anomaly
            )
            VALUES (
                s.dt,
                s.blck_cd,
                s.actual_flow,
                s.predicted_flow,
                s.weighted_error,
                s.leak_probability,
                s.reasoning,
                s.upper_bound,
                s.is_anomaly
            )
    """

    rows = df[
        [
            'Time',
            'blck_cd',
            'Actual Flow',
            'Predicted Flow',
            'Weighted Error',
            'Leak Probability',
            'Reasoning',
            'Upper Bound',
            'Is Anomaly'
        ]
    ].values.tolist()

    conn = get_connection()

    try:
        with conn:
            with conn.cursor() as cur:
                execute_batch(
                    cur,
                    sql,
                    rows,
                    page_size=500
                )
    finally:
        conn.close()

    return len(rows)


# =========================
# 블록별 예측 분석
# =========================

def analyze_block(block_data, blck_cd, save_start_date, save_end_date):

    model_path = MODEL_DIR / f"{blck_cd}.keras"
    scaler_path = MODEL_DIR / f"{blck_cd}_scaler.pkl"

    if not model_path.exists():

        print(
            f"[SKIP] model not found: {blck_cd} / {model_path}"
        )

        return pd.DataFrame()

    if not scaler_path.exists():

        print(
            f"[SKIP] scaler not found: {blck_cd} / {scaler_path}"
        )

        return pd.DataFrame()

    print(f"[LOAD MODEL] {blck_cd}")

    model = tf.keras.models.load_model(
        model_path
    )

    scaler = joblib.load(
        scaler_path
    )

    scaled_data = apply_scaler(
        block_data,
        scaler
    )

    X_test, y_test, times_test, dates_test, blck_cds_test = create_sequences_prev_7_days(
        scaled_data
    )

    if len(X_test) == 0:

        print(
            f"[SKIP] sequence not enough: {blck_cd}"
        )

        return pd.DataFrame()

    y_pred = model.predict(
        X_test,
        verbose=0
    )

    y_test_inversed = inverse_scale_flow(
        scaler,
        y_test.reshape(-1, 1)
    )

    y_pred_inversed = inverse_scale_flow(
        scaler,
        y_pred
    )

    results = pd.DataFrame({
        'Time': times_test,
        'Date': dates_test,
        'blck_cd': blck_cds_test,
        'Actual Flow': y_test_inversed,
        'Predicted Flow': y_pred_inversed
    })

    metrics_df = calculate_datewise_metrics(
        results
    )

    metrics_df = add_leak_probability_with_reasoning(
        metrics_df
    )

    final_results = results.merge(
        metrics_df,
        on=['blck_cd', 'Date'],
        how='left'
    )

    final_results = final_results[
        (final_results['Time'] >= save_start_date)
        & (final_results['Time'] <= save_end_date)
    ].copy()

    if final_results.empty:

        print(
            f"[SKIP] no result in target period: {blck_cd}"
        )

        return pd.DataFrame()

    final_results['Actual Flow'] = final_results[
        'Actual Flow'
    ].apply(
        lambda v: clamp_numeric(v, max_abs=9999.9, scale=1)
    )

    final_results['Predicted Flow'] = final_results[
        'Predicted Flow'
    ].apply(
        lambda v: clamp_numeric(v, max_abs=9999.9, scale=1)
    )

    final_results['Weighted Error'] = final_results[
        'Weighted Error'
    ].apply(
        lambda v: clamp_numeric(v, max_abs=9999.99, scale=2)
    )

    final_results['Upper Bound'] = final_results[
        'Upper Bound'
    ].apply(
        lambda v: clamp_numeric(v, max_abs=9999.99, scale=2)
    )

    return final_results


# =========================
# Main
# =========================

def main():

    args = parse_args()

    schema = args.schema
    start_date = args.start_date
    end_date = args.end_date
    blck_cd = args.blck_cd

    #schema = "by_schema"
    #start_date = "2024-01-01"
    #end_date = "2024-03-17'
    #blck_cd = "BL000073"

    save_start_date = pd.to_datetime(
        start_date
    )

    save_end_date = pd.to_datetime(
        end_date
    ) + pd.Timedelta(
        hours=23,
        minutes=59,
        seconds=59
    )

    # 직전 7일 시퀀스 생성 + 과거 90일 정상상한선 계산을 위해
    # 분석 시작일보다 97일 이전부터 조회
    load_start_date = save_start_date - pd.Timedelta(
        days=97
    )

    print("[START] AI ANALYSIS")
    print(f"[SCHEMA] {schema}")
    print(f"[PERIOD] {start_date} ~ {end_date}")
    print(f"[LOAD_FROM] {load_start_date.date()}")
    print(f"[BLOCK] {blck_cd if blck_cd else 'ALL'}")

    data = load_source_data(
        schema=schema,
        start_dt=load_start_date,
        end_dt=save_end_date,
        blck_cd=blck_cd
    )

    if data.empty:

        print(
            "[RESULT] SUCCESS rows=0 reason=no_source_data"
        )

        return

    data = preprocess_data(
        data
    )

    all_results = []

    for current_blck_cd, block_data in data.groupby(
        'blck_cd'
    ):

        block_data = block_data.sort_values(
            'time'
        ).reset_index(drop=True)

        result_df = analyze_block(
            block_data=block_data,
            blck_cd=current_blck_cd,
            save_start_date=save_start_date,
            save_end_date=save_end_date
        )

        if not result_df.empty:

            all_results.append(
                result_df
            )

    if len(all_results) == 0:

        print(
            "[RESULT] SUCCESS rows=0 reason=no_available_model_or_sequence"
        )

        return

    final_df = pd.concat(
        all_results,
        ignore_index=True
    )

    saved_count = save_results(
        schema,
        final_df
    )

    print(
        f"[RESULT] SUCCESS rows={saved_count}"
    )


if __name__ == "__main__":
    main()