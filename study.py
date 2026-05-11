import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from db import get_connection
import argparse
import holidays
import joblib

from sklearn.preprocessing import RobustScaler

# =========================
# 기본 설정
# =========================

np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# =========================
# Argument
# =========================

def parse_args():

    parser = argparse.ArgumentParser(
        description="AI Model Training"
    )

    parser.add_argument(
        "--schema",
        required=False
    )

    parser.add_argument(
        "--blck-cd",
        required=False
    )

    parser.add_argument(
        "--base-date",
        required=False,
        help="base date (YYYY-MM-DD)"
    )

    return parser.parse_args()

# =========================
# DB Load
# =========================

def load_train_data(schema, blck_cd):

    sql = f"""
        SELECT
            dt as "time",
            blck_cd,
            flow,
            temp as "temperature",
            rcvr_yn AS "leak_recovery"
        FROM {schema}.ai_anals_input
        WHERE blck_cd = %s
        ORDER BY dt
    """

    conn = get_connection()

    try:
        df = pd.read_sql(
            sql,
            conn,
            params=[blck_cd]
        )
    finally:
        conn.close()

    return df

# =========================
# 전처리
# =========================

def preprocess_data(data):

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

    return data

# =========================
# 누수복구일 삭제 대상 계산
# =========================

def calculate_drop_dates(data):

    drop_dates = []

    leak_dates = data[
        data['leak_recovery'] == 1
    ]['date'].unique()

    for l_date in leak_dates:

        pre_start = l_date - datetime.timedelta(days=7)
        post_end = l_date + datetime.timedelta(days=7)

        pre_data = data[
            (data['date'] >= pre_start)
            & (data['date'] < l_date)
            & (data['hour'].isin([2, 3, 4]))
        ]

        post_data = data[
            (data['date'] > l_date)
            & (data['date'] <= post_end)
            & (data['hour'].isin([2, 3, 4]))
        ]

        if not pre_data.empty and not post_data.empty:

            pre_mnf = pre_data['flow'].min()
            post_mnf = post_data['flow'].min()

            if (pre_mnf - post_mnf) > 0.5:
                drop_dates.append(l_date)

        else:

            drop_dates.append(l_date)

    return drop_dates

# =========================
# 스케일링
# =========================

def scale_features(train_data):

    scaler = RobustScaler()

    scaled_train = train_data.copy()
    scaled_train[
        ['flow', 'tmax', 'tmin']
    ] = scaler.fit_transform(
        train_data[
            ['flow', 'tmax', 'tmin']
        ]
    )

    return scaled_train, scaler

# =========================
# 시퀀스 생성
# =========================

def create_sequences_prev_7_days(data):

    sequences = []
    targets = []

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

        target_date = row['date']
        target_hour = row['hour']
        target_flow = row['flow']
        blck_cd = row['blck_cd']

        valid_7_days = True

        prev_7_days_seqs = []

        for d in range(7, 0, -1):

            p_date = (
                target_date
                - pd.Timedelta(days=d)
            )

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

            scaled_target_hour = (
                target_hour / 23.0
            )

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

            sequences.append(
                seq_with_target_hint
            )

            targets.append(
                target_flow
            )

    return (
        np.array(sequences),
        np.array(targets)
    )

# =========================
# 모델 생성
# =========================

def build_model(input_shape):

    model = tf.keras.Sequential([

        tf.keras.layers.LSTM(
            64,
            return_sequences=True,
            input_shape=input_shape
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(32),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            16,
            activation='relu'
        ),

        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss='mean_squared_error'
    )

    return model

# =========================
# Main
# =========================

args = parse_args()

SCHEMA = args.schema
BLCK_CD = args.blck_cd
BASE_DATE = args.base_date

#SCHEMA = "by_schema"
#BLCK_CD = "BL000074"
#BASE_DATE = "2025-08-01"

print(f"[START] TRAIN MODEL : {BLCK_CD}")

# =========================
# 데이터 로드
# =========================

data = load_train_data(
    SCHEMA,
    BLCK_CD
)

if data.empty:

    raise Exception(
        f"데이터 없음 : {BLCK_CD}"
    )

# =========================
# 학습 종료일
# 직전년도 말
# =========================

if BASE_DATE:
    TODAY = pd.to_datetime(BASE_DATE)
else:
    TODAY = pd.Timestamp.today()

TRAIN_END_DATE = pd.Timestamp(
    year=TODAY.year - 1,
    month=12,
    day=31,
    hour=23,
    minute=59,
    second=59
)

data = preprocess_data(data)

drop_dates = calculate_drop_dates(data)

train_data = data[
    data['time'] <= TRAIN_END_DATE
].copy()

train_data = train_data[
    ~train_data['date'].isin(drop_dates)
]

if train_data.empty:

    raise Exception(
        f"학습 데이터 없음 : {BLCK_CD}"
    )

print(
    f"[TRAIN DATA] {len(train_data)}"
)

# =========================
# 스케일링
# =========================

scaled_train_data, scaler = scale_features(
    train_data
)

# =========================
# 시퀀스 생성
# =========================

X_train, y_train = create_sequences_prev_7_days(
    scaled_train_data
)

if len(X_train) == 0:

    raise Exception(
        f"시퀀스 부족 : {BLCK_CD}"
    )

print(
    f"[X_TRAIN] {X_train.shape}"
)

# =========================
# 모델 생성
# =========================

model = build_model(
    (
        X_train.shape[1],
        X_train.shape[2]
    )
)

validation_split = (
    0.05 if len(X_train) >= 100 else 0
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(

    X_train,
    y_train,

    epochs=50,

    batch_size=32,

    validation_split=validation_split,

    callbacks=[
        early_stop
    ] if validation_split > 0 else [],

    verbose=1
)

# =========================
# 모델 저장
# =========================

model_path = (
    MODEL_DIR
    / f"{BLCK_CD}.keras"
)

scaler_path = (
    MODEL_DIR
    / f"{BLCK_CD}_scaler.pkl"
)

model.save(model_path)

joblib.dump(
    scaler,
    scaler_path
)

print(f"[MODEL SAVED] {model_path}")
print(f"[SCALER SAVED] {scaler_path}")

print(f"[RESULT] Study SUCCESS about {BLCK_CD}")