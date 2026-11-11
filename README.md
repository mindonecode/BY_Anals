# BY_anals - AI 누수 분석 (Python)

블록별 시간 단위 유량(flow)을 LSTM 모델로 예측하고, 실제 유량과의 가중 오차율로 일자별 누수 가능성을 판정해 DB 에 저장하는 배치 프로그램.
(확인 대상: `main.py`, `study.py`, `db.py`, `run.sh`, `models/`. `venv`, `.env` 는 열람하지 않음.)

## 1. 파일 구성
| 파일 | 역할 |
|---|---|
| `main.py` | 예측/분석 실행. 모델·스케일러 로드 → 예측 → 오차율/누수가능성 계산 → `ai_anals_output` 저장 |
| `study.py` | 블록 1개 모델 학습 후 `models/` 에 저장 (모듈 최상위 코드로 바로 실행, `main()` 함수 없음) |
| `db.py` | `get_connection()` : psycopg2 로 PostgreSQL 접속. `python-dotenv`(`load_dotenv()`)로 `.env` 읽음 |
| `run.sh` | `venv/bin/activate` 후 `python study.py` 실행(인자 없음) |
| `main.py.back` | 이전 버전으로 보이는 백업 파일(내용 미분석, 확인 필요) |
| `models/` | 블록별 `<blck_cd>.keras`, `<blck_cd>_scaler.pkl` |

Python 패키지(import 기준, requirements 파일 없음): numpy, pandas, tensorflow(keras), holidays, joblib, scikit-learn(RobustScaler), psycopg2, python-dotenv.

## 2. 실행 방법

### 예측 (`main.py`)
```
python main.py --schema <스키마> --start-date YYYY-MM-DD --end-date YYYY-MM-DD [--blck-cd <블록코드>]
```
| 인자 | 필수 | 설명 |
|---|---|---|
| `--schema` | O | DB 스키마명 (SQL 에 `{schema}.테이블` 로 직접 삽입) |
| `--start-date` | O | 결과 저장 시작일 |
| `--end-date` | O | 결과 저장 종료일(당일 23:59:59 까지 포함) |
| `--blck-cd` | X | 특정 블록만. 없으면 조회된 모든 블록 |

- 종료 시 표준출력에 `[RESULT] SUCCESS rows=<저장건수>` 형태 출력(소스 없음/모델 없음 시 `rows=0 reason=...`). 예외 시 비정상 종료.

### 학습 (`study.py`)
```
python study.py --schema <스키마> --blck-cd <블록코드> [--base-date YYYY-MM-DD]
```
| 인자 | 필수 | 설명 |
|---|---|---|
| `--schema` | O | DB 스키마명 |
| `--blck-cd` | 코드상 X | 코드상 required=False 이나 `None` 이면 쿼리 파라미터가 `NULL` 이 되어 데이터 없음 예외가 발생하므로 사실상 필요 |
| `--base-date` | X | 학습 기준일. 없으면 오늘 날짜 |

- 종료 시 `[RESULT] Study SUCCESS about <블록코드>` 출력. 데이터/학습데이터/시퀀스가 없으면 예외로 종료.
- `run.sh` 는 `study.py` 를 인자 없이 실행하므로 `--schema` 필수 조건에 의해 그대로는 argparse 오류가 난다(스크립트 내용 기준).

## 3. 처리 파이프라인

### 3.1 데이터 조회 (입력)
`{schema}.ai_anals_input` 에서 조회(예측/학습 공통 컬럼):
```sql
SELECT dt as "time", blck_cd, flow, tmax, tmin, rcvr_yn AS "leak_recovery"
FROM {schema}.ai_anals_input
```
- 예측(`main.py`): `dt >= load_start AND dt <= end_date 23:59:59` (+ `blck_cd = %s` 선택), `ORDER BY blck_cd, dt`. `load_start = start_date - 97일` (직전 7일 시퀀스 + 과거 90일 정상 상한선 계산용).
- 학습(`study.py`): `WHERE blck_cd = %s ORDER BY dt` (기간 제한 없이 전체 조회 후 코드에서 학습 종료일로 필터).
- `ai_anals_input` 은 백엔드가 채운다(4장 참조). 컬럼: `dt`(시간 단위 timestamp), `blck_cd`, `flow`, `tmax`, `tmin`, `rcvr_yn`.

### 3.2 피처 / 전처리
- `time` 에서 파생: `hour`, `day_of_week`, `month`, `date`, `is_weekend`(요일>=5), `is_holiday`(`holidays.KR()`).
- 시퀀스 피처 9개/시점: `flow, tmax, tmin, hour, day_of_week, is_weekend, is_holiday, month` + 예측 대상 시각 힌트(`target_hour / 23.0`).
- 하루가 시간 키(`YYYY-MM-DD HH`) 중복 제거 후 **정확히 24행**인 날만 일 시퀀스로 사용.
- 예측 대상 1행(시각)마다 **직전 7일(각 24시간 = 168행)** 이 모두 존재할 때만 샘플 생성 → 입력 shape `(샘플, 168, 9)`, 타깃은 해당 시각의 `flow`. 직전 7일 중 하루라도 없으면 샘플 제외.

### 3.3 스케일러
- `sklearn.preprocessing.RobustScaler`, 블록별 1개. 대상 컬럼은 `flow, tmax, tmin` 만 (나머지 피처는 스케일하지 않음).
- 학습: 학습 데이터로 `fit_transform` 후 `joblib.dump`. 예측: `joblib.load` 후 `transform`.
- 예측값 역변환(`inverse_scale_flow`): 예측값 뒤에 0 두 열을 붙여 `inverse_transform` 하고 첫 열(flow)만 사용.

### 3.4 모델 구조 (`study.py` `build_model`)
```
LSTM(64, return_sequences=True, input_shape=(168, 9))
Dropout(0.2)
LSTM(32)
Dropout(0.2)
Dense(16, relu)
Dense(1)
```
- 컴파일: `RMSprop`, 손실 `mean_squared_error`.
- 학습: `epochs=40`, `batch_size=32`, `validation_split = 0.05`(샘플 100개 이상) 아니면 0, `EarlyStopping(monitor=val_loss, patience=5, restore_best_weights=True)`(검증셋 있을 때만). 시드 `numpy/tf = 42`.

### 3.5 학습 데이터 선정 (`study.py`)
1. 학습 종료일 = (`--base-date` 또는 오늘)의 **전년도 12/31 23:59:59**. `time <= 종료일` 데이터만 사용.
2. 누수복구일(`leak_recovery == 1`) 중 제외할 날짜(`calculate_drop_dates`):
   - 복구일 전 7일 / 후 7일의 02~04시 데이터가 모두 있으면 각 구간 `flow` 최소값(MNF) 차이 `pre - post > 0.5` 일 때 해당 복구일 제외.
   - 전/후 구간 중 하나라도 비어 있으면 해당 복구일 제외.
3. 제외 날짜를 뺀 데이터로 스케일러 fit → 시퀀스 생성 → 학습.

### 3.6 예측 및 누수 판정 (`main.py`)
1. 블록별로 `models/<blck_cd>.keras`, `<blck_cd>_scaler.pkl` 존재 확인(없으면 `[SKIP]` 로그 후 해당 블록 건너뜀).
2. 시퀀스 생성 → `model.predict` → 실제/예측 flow 역스케일.
3. 일자별 가중 오차율(`calculate_datewise_metrics`): 실제 > 예측 인 시각만, 실제 != 0 만 대상. 오차율 `|실제-예측|/실제*100`. 가중치: 02/03/04시 각 `0.8/3`, 그 외 각 `0.2/21`. 가중합이 일자별 `Weighted Error`(대상 시각이 없으면 0).
4. 누수 가능성(`add_leak_probability_with_reasoning`): 블록별 각 일자의 직전 90일(당일 제외) `Weighted Error` 목록에서 90퍼센타일 이하 값만 취해 `상한 = 평균 + 표준편차`. 당일 오차율 > 상한 이면 `높음`(`is_anomaly=True`), 아니면 `낮음`. 과거 데이터가 없으면 `데이터부족` + 사유 문구(상한 NULL).
5. `start_date ~ end_date 23:59:59` 범위의 시각 행만 남김. 수치 정제(`clamp_numeric`): 유량류 소수 1자리/절대값 9999.9 상한, 오차율·상한 소수 2자리/9999.99 상한.

### 3.7 결과 저장 (`{schema}.ai_anals_output`)
`MERGE INTO ... ON (dt, blck_cd)` (일치 시 UPDATE, 없으면 INSERT), `execute_batch(page_size=500)`.

| 컬럼 | 값 |
|---|---|
| `dt` | 예측 시각 (timestamp) |
| `blck_cd` | 블록코드 |
| `actual_flow` | 실제 유량 |
| `predicted_flow` | 예측 유량 |
| `weighted_error` | 해당 일자 가중 오차율(시간 행마다 동일 값 반복) |
| `leak_probability` | `높음` / `낮음` / `데이터부족` |
| `reasoning` | 판정 사유 문구 |
| `upper_bound` | 정상 상한선 |
| `is_anomaly` | boolean |

## 4. 학습 / 예측 차이
| 구분 | 학습 (`study.py`) | 예측 (`main.py`) |
|---|---|---|
| 필수 인자 | `--schema` (`--blck-cd`, `--base-date` 선택) | `--schema`, `--start-date`, `--end-date` (`--blck-cd` 선택) |
| 조회 범위 | 블록 전체 이력 후 `전년도 12/31` 까지 사용 | `start_date - 97일 ~ end_date 23:59:59` |
| 블록 처리 | 인자로 받은 블록 1개 | 인자 없으면 조회된 전체 블록 순회 |
| 누수복구일 | 학습 제외일 계산에 사용 | 입력 컬럼(`leak_recovery`)으로 읽기만 하고 판정에는 미사용 |
| 스케일러 | `RobustScaler.fit_transform` 후 저장 | 저장된 스케일러 `transform` |
| 모델 | 생성/학습/저장 | 로드해서 `predict` |
| DB 쓰기 | 없음 | `ai_anals_output` MERGE |
| 모델 파일 | 기존 파일이 있으면 삭제 후 저장 | 읽기만 |

## 5. 모델 파일 규칙
- 경로: `BY_anals/models/` (`MODEL_DIR = <스크립트 폴더>/models`, 학습 시 폴더 자동 생성).
- 파일명: 모델 `<blck_cd>.keras`, 스케일러 `<blck_cd>_scaler.pkl` (예: 현재 `BL000045.keras`, `BL000045_scaler.pkl` 1세트 존재).
- 예측 시 둘 중 하나라도 없으면 해당 블록은 건너뜀. 확장자는 `.keras` (루트 사본은 `.h5`, 8장 참조).

## 6. 환경변수 (`.env`, 값은 기재하지 않음)
`db.py` 가 `load_dotenv()` 후 사용: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`.
그 외 스크립트에서 설정하는 값: `TF_CPP_MIN_LOG_LEVEL=3`.
(백엔드가 프로세스 환경에 `PYTHONIOENCODING=utf-8`, `PYTHONUTF8=1` 을 추가)

## 7. 백엔드(BY_back) 연동
패키지 `org.example.by_back.scheduler.dataAiAnals` :
| 클래스 | 역할 |
|---|---|
| `DataAiAnalsScheduler` | `@Scheduled(cron = "0 0 4 * * *")` 매일 04:00. 어제(`sDate`)~오늘(`eDate`) 로 `saveAiLeakInput(sDate, eDate, "")` 로 입력 테이블 적재 후 `executePredict` 로 `main.py` 실행 |
| `AiAnalsStudyScheduler` | `@Scheduled(cron = "0 0 1 10 1 *")` 매년 1월 10일 01:00 학습. `sDate="2024-01-01"`, `eDate=<올해>-01-01`, `base_year=<작년>` 으로 `study.py` 실행 |
| `PythonMain` | `ProcessBuilder` 실행기. 명령: `<python 실행파일> <스크립트> [--start-date .. --end-date ..] [--base-year ..] --schema <app.database.schema> --blck-cd <블록>`. 블록 인자가 없으면 `blckService.getRealAllBlock()` 의 각 `blck_cd` 별로 **블록마다 별도 프로세스** 실행. 작업 디렉터리=`python.project.path`, stderr 를 stdout 에 합쳐 로그(`[PYTHON]`), 출력 중 `[RESULT]` 포함 마지막 줄을 결과로 사용, 종료코드 0 이 아니면 `[RESULT] FAIL` |
| `SchedulerTestController` | 수동 실행 API(POST): `/ai-study/{startDate}/{endDate}/{baseYear}/{blckCd}`, `/ai-leaknalsTotal/{sDate}/{eDate}/{baseYear}/{blckCd}`(입력 적재 + 예측), `/ai-leak-inputData/{sDate}/{eDate}/{blckCd}`(입력 적재만), `/ai-leak-initPython/{sDate}/{eDate}/{baseYear}/{blckCd}`(예측만). `baseYear`, `blckCd` 는 쿼리 파라미터(선택) |

- 경로 설정: 우선 DB `python_config` 테이블(`select path from python_config where key = #{key}`; 키 `python.exec.path`, `python.script.path`, `python.study.path`)을 조회하고, 값이 없으면 `application.properties` 의 같은 이름 프로퍼티를 사용. 추가로 `python.project.path`(작업 디렉터리), `app.database.schema`(`--schema` 값).
- 입력 적재(`DataAnlsMapper.xml` `saveAiLeakInput`, `merge into ai_anals_input`): 지정 기간 시간 단위 `flow`(계측 `msrmt` 의 보정값 `crct_vl` 우선·없으면 `msrmt_vl`, 블록-계측기-태그 매핑 `msrmt_istr`/`msrmt_istr_tag_mpng` 의 `cal_type` +/- 합산, 계측기 유형 `01`), `rcvr_yn`(누수복구대장 `wtlkg_rstr_ldgr` 의 `cstrn_ymd` 일치 건수 > 0 → 1), `tmax`/`tmin`(`temper` 의 최근 `high_temp`/`low_temp`)을 `(dt, blck_cd)` 기준 UPSERT. 블록코드 `BL000000`, `BLAll` 가상 블록도 포함.
- 결과 사용: 대시보드 API `dayLeakBoard/getAiLeakAnalsInfo/{dt}/{blckCd}` → `DayLeakBoardMapper.xml` 의 `ai_anals_output` 조회(일별/주간).

## 8. 차이 요약 (루트 `main.py`, `study.py` vs `BY_anals/`)
`PROJECT_GUIDE.md` 기준 실제 사용본은 `BY_anals/`. 루트 사본은 `diff` 로 확인한 결과 다른 알고리즘 버전이다.

### main.py
| 항목 | 루트 `main.py` | `BY_anals/main.py` |
|---|---|---|
| 인자 | `--schema/--start-date/--end-date` 가 `required=False`, 그러나 `main()` 에서 `args.*` 사용 줄이 주석이고 스키마·기간·블록이 코드에 **하드코딩** | 3개 필수(`required=True`), `args.*` 사용 |
| 데이터 조회 | `load_source_data(schema, blck_cd)`: 기간 조건 없이 전체 조회(`WHERE 1 = 1`) | `load_source_data(schema, start_dt, end_dt, blck_cd)`: `start-97일 ~ end` |
| 전처리 컬럼 | `year` 컬럼 추가 | `year` 없음 |
| 시퀀스 shape | `(샘플, 7, 24, 9)` 4차원, 직전 7일에 최근일 가중치 `linspace(0.5,1.0,7)` 곱함 | `(샘플, 168, 9)` 3차원, 가중치 없음 |
| 모델 파일 | `models/<blck>.h5` | `models/<blck>.keras` |
| 오차율 계산 | 가중치 루프 방식, `errors` NaN 제거 없음, 빈 배열 가드 없음 | 벡터화(`np.isin`), 오차 배열 빈 경우 0 처리, `dropna()` 및 빈 배열 가드 추가 |
| 시작 로그 | `[START] AI HIERARCHICAL ANALYSIS` | `[START] AI ANALYSIS`, `[LOAD_FROM]` 출력 |
| 로그 | `[MODEL PATH]`, `[SCALER PATH]`, `[X_ALL]` 출력 | 해당 출력 없음 |
| 실행 구조 | `main()` + `if __name__ == "__main__": main()` (동일) | 동일 |

### study.py
| 항목 | 루트 `study.py` | `BY_anals/study.py` |
|---|---|---|
| 설명/구조 | "AI Hierarchical Model Training", `main()` 함수 | "AI Model Training", 모듈 최상위 실행 |
| 인자 | `--schema`, `--blck-cd` 모두 `required=False`, `main()` 에서 스키마/블록/기준일 **하드코딩** | `--schema` 필수, 인자 값 사용 |
| 기준 인자 | `--base-date`(사용 줄은 주석) | `--base-date` 사용 |
| 학습 데이터 | `year < base_year`(기준일 없으면 데이터 최대 연도) | `time <= (기준일 연도-1)-12-31 23:59:59` |
| 제외일 계산 | 복구일 판정 후 복구 전 누수 시작일을 역추적(전일부터 MNF 차 > 0.5 인 동안)해 기간 전체를 제외 | 복구일 단일 날짜만 제외 |
| 시퀀스 | `(샘플, 7, 24, 9)` + 최근일 가중치 | `(샘플, 168, 9)` |
| 모델 | `TimeDistributed(LSTM(32))` → `LSTM(32)` → `Dense(16)` → `Dense(1)` (Dropout 없음) | `LSTM(64)` → `Dropout` → `LSTM(32)` → `Dropout` → `Dense(16)` → `Dense(1)` |
| 학습 옵션 | `patience=10`, `validation_split=0.1` 고정 | `patience=5`, `validation_split` 0.05(샘플>=100)/0 |
| 저장 | `models/<blck>.h5` | `models/<blck>.keras` |

### 공통/그 외
- 루트의 `runAnals.bat` 은 `.venv` 활성화 후 `main.py`(루트 폴더 기준) 를 인자 없이 실행한다(하드코딩 값 사용 구조와 대응).
- `db.py`, `models/`, `run.sh` 는 `BY_anals/` 에만 존재(루트 사본은 `main.py`, `study.py` 만 확인).

## 9. 확인 필요 항목
- 백엔드가 `study.py` 에 `--start-date`, `--end-date`, `--base-year` 를 전달하고 `main.py` 에도 `--base-year` 를 (값이 있을 때) 전달하는데, `BY_anals` 의 argparse 는 이 인자들을 정의하지 않는다(`study.py`: `--schema`, `--blck-cd`, `--base-date` / `main.py`: `--schema`, `--start-date`, `--end-date`, `--blck-cd`). 운영 서버에서 사용되는 스크립트 버전 확인 필요(스케줄러 호출 시 argparse 가 미정의 인자를 거부하는지는 실행 기준으로 확인 필요).
- `python_config` 테이블의 실제 키/값(운영 경로).
- `ai_anals_input`/`ai_anals_output` DDL(컬럼 타입, 인덱스): 소스에서는 SQL 캐스트와 MERGE 키(`dt`, `blck_cd`)만 확인.
- `main.py.back` 의 용도/버전.
- `run.sh` 의 인자 없는 `study.py` 호출 사용 여부.
