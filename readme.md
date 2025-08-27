# AI Sound & Vibration Manufacture

실시간 AI 음성 및 진동 탐지 시스템으로, 음성 파일과 진동 데이터를 자동으로 분석하여 양품/불량품을 판별하는 시스템입니다.

## 새로운 기능: 진동 모델 추가

이제 소리 데이터뿐만 아니라 진동 데이터도 분석할 수 있습니다!

### 주요 개선사항:
- **진동 모델 추가**: 3축 진동 데이터 (Acc1, Acc2, Acc3) 분석
- **멀티프로세스 지원**: 소리와 진동 모델이 동시에 실행
- **스펙트로그램 기반**: STFT를 이용한 시간-주파수 분석
- **AutoEncoder + DTW**: CNN AutoEncoder와 2D DTW 결합

## 새로운 기능: 클라우드 메모리 처리

이제 클라우드 스토리지의 파일을 로컬에 다운로드하지 않고 메모리에서 직접 처리할 수 있습니다!

### 주요 개선사항:
- **임시 파일 없음**: 클라우드 파일을 로컬에 다운로드하지 않음
- **메모리 효율성**: 스트림 방식으로 메모리 사용량 최적화
- **빠른 처리**: 디스크 I/O 없이 직접 처리로 성능 향상
- **자동 정리**: 메모리 스트림 자동 정리로 메모리 누수 방지

## 주요 변경사항 (노트북 방식 반영)

### 1. 스펙트로그램 생성 방식 개선
- **기존**: PIL Image 직접 변환
- **개선**: matplotlib → PIL 변환 (노트북 방식)
- **장점**: 더 정확한 시각적 표현과 모델 성능 향상

### 2. DTW 계산 방식 변경
- **기존**: 스펙트로그램 flatten 기반 fastdtw
- **개선**: MFCC 13차원 특징 벡터 기반 librosa.sequence.dtw
- **장점**: 소리 패턴의 더 정확한 유사도 측정

### 3. 동적 임계값 설정
- **기존**: 설정 파일 고정 임계값
- **개선**: 각 파일별 동적 임계값 (percentile 기반)
- **장점**: 데이터 특성에 맞는 적응적 판단

### 4. 3초 미만 클립 처리
- **기존**: 3초 미만 클립 제외
- **개선**: 3초 미만 클립도 포함 (패딩 옵션)
- **장점**: 더 많은 데이터 활용

## 프로젝트 구조

```
ai_sound_manufacture/
├── data/
│   ├── raw_data/          # 원시 파일 입력 폴더
│   │   ├── sound/         # 소리 파일들 (.wav)
│   │   └── vibration/     # 진동 파일들 (.csv)
│   ├── sound_reserved/    # 소리 데이터 결과 저장 폴더
│   │   ├── normal/        # 소리 양품 파일
│   │   └── abnormal/      # 소리 불량품 파일
│   ├── vib_reserved/      # 진동 데이터 결과 저장 폴더
│   │   ├── normal/        # 진동 양품 파일
│   │   └── abnormal/      # 진동 불량품 파일
│   ├── images/            # 스펙트로그램 이미지 저장 폴더
│   └── segments/          # 세그먼트 파일 저장 폴더
├── src/
│   ├── ai/                # AI 탐지 모듈
│   │   ├── sound_detect_if.py   # 소리 AI 탐지 인터페이스
│   │   ├── sound_detect_proc.py # 소리 AI 탐지 프로세스
│   │   ├── vib_detect_if.py     # 진동 AI 탐지 인터페이스
│   │   └── vib_detect_proc.py   # 진동 AI 탐지 프로세스
│   ├── db/                # 데이터 저장 모듈
│   │   ├── db_if.py       # 저장 처리 인터페이스
│   │   └── db_proc.py     # 저장 처리 프로세스
│   ├── evaluation/        # 파일 처리 모듈
│   │   ├── evaluation_if.py     # 소리/진동 평가 인터페이스
│   │   └── evaluation_proc.py   # 소리/진동 평가 프로세스
│   ├── preprocess/        # 전처리 모듈
│   │   ├── file_check_if.py     # 파일 체크 인터페이스
│   │   └── file_check_proc.py   # 파일 체크 프로세스
│   └── util/              # 유틸리티 모듈
├── logs/                  # 로그 파일 저장 폴더
├── config.json            # 설정 파일
├── main.py               # 메인 실행 파일
└── requirements.txt      # Python 패키지 의존성
```

### 패키지 설치
pip install --break-system-packages -r requirements.txt

### mysql 관련 (msqlclient)
sudo apt update
sudo apt install pkg-config build-essential python3-dev default-libmysqlclient-dev
pip install --break-system-packages -r requirements.txt

### cpu / memory / 프로세스 관련
pip install --break-system-packages psutil 또는 `psutil>=5.9.0`

### cuda toolkit
# NVIDIA 공식 키 추가
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# CUDA Toolkit 12.2 설치
sudo apt install -y cuda-toolkit-12-2
# path 추가
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 설치 확인
nvcc --version

# cdDNN (GPU 가속화)
sudo apt install -y libcudnn8 libcudnn8-dev
# 설치 확인
dpkg -l | grep cudnn


## 시스템 아키텍처

이 시스템은 6개의 독립적인 프로세스로 구성되어 있으며, 큐를 통해 데이터를 전달합니다:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   파일 체크     │───▶│   AI 탐지       │───▶│   파일 처리     │
│   프로세스      │    │   프로세스      │    │   프로세스      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   DB 처리       │    │   성능 모니터링 │
                       │   프로세스      │    │   프로세스      │
                       └─────────────────┘    └─────────────────┘
```

1. **파일 체크 프로세스**: `raw_data` 폴더를 모니터링하여 새로운 파일을 감지
2. **소리 AI 탐지 프로세스**: 음성 파일을 분석하여 스펙트로그램 생성, DTW 및 AE Loss 계산
3. **진동 AI 탐지 프로세스**: 진동 CSV 파일을 분석하여 스펙트로그램 생성, DTW 및 AE Loss 계산
4. **저장 처리 프로세스**: 탐지 결과를 Google Cloud Storage에 JSON 파일로 저장
5. **소리 평가 프로세스**: 소리 탐지 결과를 평가하고 파일을 적절한 폴더로 이동
6. **진동 평가 프로세스**: 진동 탐지 결과를 평가하고 파일을 적절한 폴더로 이동

## 핵심 기능

### 1. 실시간 오디오 처리
- **3초 단위 분할**: 오디오를 3초 단위로 분할하여 처리
- **패딩 지원**: 3초 미만 클립도 zero/repeat/edge 패딩으로 처리
- **멀티프로세스**: 큐 기반 비동기 처리로 실시간 성능 보장

### 2. 하이브리드 탐지 알고리즘
- **AutoEncoder**: 스펙트로그램 복원 손실 기반 이상 탐지
- **DTW**: MFCC 기반 시계열 패턴 유사도 측정
- **하이브리드 점수**: AE Loss와 DTW 거리의 가중 평균

### 3. 동적 임계값 설정
```python
# 각 파일별 동적 임계값 계산
ae_threshold_dynamic = np.percentile(ae_losses_temp, 90)
dtw_threshold_dynamic = np.percentile(dtw_scores_temp, 90)
```

### 4. 성능 평가 시스템
- **세그먼트 단위**: 개별 3초 클립의 정확도
- **파일 단위**: 전체 파일의 최종 판정 정확도
- **실시간 지표**: accuracy, precision, recall, f1-score

## 성능 지표

### 세그먼트 단위 지표
- **clip_accuracy**: 개별 세그먼트 정확도
- **clip_precision**: 세그먼트 단위 정밀도
- **clip_recall**: 세그먼트 단위 재현율
- **clip_f1**: 세그먼트 단위 F1 점수

### 파일 단위 지표
- **file_accuracy**: 파일 전체 정확도
- **file_precision**: 파일 단위 정밀도
- **file_recall**: 파일 단위 재현율
- **file_f1_score**: 파일 단위 F1 점수

## 설치 및 설정

### 1. 시스템 요구사항
- Python 3.8 이상
- Windows 10/11 또는 Linux
- 최소 4GB RAM
- 인터넷 연결 (Google Cloud Storage 접근용)

### 2. Python 가상환경 생성 및 활성화
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. Google Cloud 설정

#### 4.1 Google Cloud 프로젝트 생성
1. [Google Cloud Console](https://console.cloud.google.com/)에 접속
2. 새 프로젝트 생성 또는 기존 프로젝트 선택
3. Cloud Storage API 활성화

#### 4.2 서비스 계정 키 생성
1. IAM 및 관리 → 서비스 계정으로 이동
2. 새 서비스 계정 생성
3. Cloud Storage 관리자 권한 부여
4. JSON 키 파일 다운로드

#### 4.3 Cloud Storage 버킷 생성
1. Cloud Storage → 버킷으로 이동
2. 새 버킷 생성 (예: `ai-sound-manufacture-results`)
3. 버킷 이름을 기억해두세요

### 5. 설정 파일 수정
`config.json` 파일을 다음과 같이 수정하세요:

```json
{
  "storage": {
    "type": "google_cloud_storage",
    "project_id": "your-actual-project-id",
    "bucket_name": "your-actual-bucket-name",
    "credentials_path": "path/to/your/service-account-key.json",
    "audio_prefix": "audio/",
    "results_prefix": "results/",
    "use_memory_processing": true,
    "max_files_per_day": 30000
  },
  "vibration": {
    "fs": 1000,
    "duration": 3,
    "skiprows": 11,
    "model_path": "models/vib_cnn_ae.keras",
    "threshold_path": "models/vib_cnn_thresh.npy",
    "ae_weight": 0.5,
    "dtw_weight": 0.5,
    "margin": 1.0,
    "progress_step": 5
  }
}
```

#### 진동 모델 설정 옵션:
- `fs`: 샘플링 주파수 (Hz)
- `duration`: 윈도우 길이 (초)
- `skiprows`: CSV 헤더 건너뛸 줄 수
- `model_path`: 학습된 모델 파일 경로
- `threshold_path`: 임계값 파일 경로
- `ae_weight`: AutoEncoder 가중치
- `dtw_weight`: DTW 가중치
- `margin`: 판정 여유도
- `progress_step`: 진행률 표시 간격 (%)

#### 메모리 처리 설정 옵션:
- `use_memory_processing`: `true`로 설정하면 클라우드 파일을 메모리에서 직접 처리 (권장)
- `max_files_per_day`: 일일 최대 처리 파일 수 제한

### 6. 메모리 처리 기능 테스트

새로운 메모리 처리 기능을 테스트하려면:

```bash
python test_memory_processing.py
```

이 테스트는 다음을 확인합니다:
- 클라우드 파일 스트림 가져오기
- 메모리에서 오디오 파일 읽기
- 성능 비교 (스트림 vs 바이트 방식)
- 새 파일 정보 조회

### 7. 필요한 디렉토리 생성
```bash
# Windows
mkdir data\raw_data data\sound_reserved\normal data\sound_reserved\abnormal data\vib_reserved\normal data\vib_reserved\abnormal data\images data\segments logs

# Linux/Mac
mkdir -p data/raw_data/sound data/raw_data/vibration data/sound_reserved/normal data/sound_reserved/abnormal data/vib_reserved/normal data/vib_reserved/abnormal data/images data/segments results logs
```

## 진동 모델 사용법

### 진동 데이터 형식
CSV 파일은 다음 형식이어야 합니다:
- 헤더: 11줄 건너뛰기
- 컬럼: Acc1, Acc2, Acc3, Microphone
- 샘플링 주파수: 1000Hz (기본값)
- 윈도우 길이: 3초

### 모델 학습 및 테스트
진동 모델의 학습과 테스트는 별도 스크립트 없이 직접 Python 코드로 수행할 수 있습니다:

```python
from src.ai.vib_detect_proc import (
    load_and_preprocess_vibration_data,
    train_vibration_cnn_ae_dtw,
    detect_vibration_anomaly_cnn_dtw
)

# 학습
vib_data = load_and_preprocess_vibration_data("학습용_데이터.csv")
model, threshold = train_vibration_cnn_ae_dtw(vib_data)

# 테스트
test_data = load_and_preprocess_vibration_data("테스트용_데이터.csv")
losses, dtws, results = detect_vibration_anomaly_cnn_dtw(test_data, "models/vib_cnn_ae.keras", "models/vib_cnn_thresh.npy")
```

## 실행 방법

### 1. 메모리 처리 기능 확인
먼저 메모리 처리 기능이 정상 작동하는지 테스트하세요:

```bash
python test_memory_processing.py
```

### 2. 기본 실행 (소리 + 진동)
```bash
python main.py
```

### 3. 백그라운드 실행 (Linux/Mac)
```bash
nohup python main.py > output.log 2>&1 &
```

### 4. 실행 확인
시스템이 정상적으로 실행되면 다음과 같은 메시지가 출력됩니다:
```
main process 실행
프로세스 큐 생성 완료
파일 체크 프로세스 시작
소리 AI 탐지 프로세스 시작
진동 AI 탐지 프로세스 시작
DB 처리 프로세스 시작
소리 평가 프로세스 시작
진동 평가 프로세스 시작
모든 프로세스 시작 완료
```

## 테스트 방법

### 1. 테스트 음성 파일 준비
- WAV, MP3, FLAC 등 지원되는 오디오 형식
- 파일 크기: 1MB 이하 권장
- 길이: 3-10초 권장

### 2. 파일 업로드 테스트
1. `data/raw_data/` 폴더에 테스트 음성 파일을 복사
2. 시스템이 자동으로 파일을 감지하고 처리
3. 처리 결과 확인:
   - `data/split/`: 양품으로 분류된 파일
   - `data/reserved/`: 불량품으로 분류된 파일
   - `logs/app.log`: 처리 로그

### 3. 실시간 모니터링
```bash
# 로그 실시간 확인
tail -f logs/app.log

# Windows
Get-Content logs/app.log -Wait
```

## 결과 확인

### 1. 로컬 결과
- **소리 양품**: `data/sound_reserved/normal/` 폴더에 저장
- **소리 불량품**: `data/sound_reserved/abnormal/` 폴더에 저장
- **진동 양품**: `data/vib_reserved/normal/` 폴더에 저장
- **진동 불량품**: `data/vib_reserved/abnormal/` 폴더에 저장
- **로그**: `logs/app.log` 파일에 상세 기록

### 2. Cloud Storage 결과
- Google Cloud Storage의 `detection_results/` 폴더에 JSON 파일로 저장
- 파일명 형식: `{timestamp}_{원본파일명}.json`

### 3. JSON 결과 예시
```json
{
  "file_path": "data/raw_data/test.wav",
  "original_filename": "test.wav",
  "result": "양품",
  "dtw_score": 1234.56,
  "ae_loss": 0.00123,
  "final_score": 0.45,
  "timestamp": "2024-01-01 12:00:00",
  "folder_info": {},
  "created_at": "2024-01-01 12:00:00",
  "storage_path": "detection_results/2024-01-01_12-00-00_test.json"
}
```

## 탐지 과정

### 1. 오디오 전처리
```python
# 3초 단위 분할
segments = split_audio(audio_data, sample_rate, clip_duration=3.0)

# 패딩 처리 (필요시)
if len(segment) < clip_samples:
    segment = pad_to_length(segment, clip_samples, pad_mode)
```

### 2. 스펙트로그램 생성 (노트북 방식)
```python
# Mel 스펙트로그램 → matplotlib → PIL 변환
S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
S_dB = librosa.power_to_db(S, ref=np.max)
# matplotlib으로 이미지 생성 후 PIL로 변환
```

### 3. MFCC 기반 DTW 계산
```python
# MFCC 특징 벡터 계산
mfcc = compute_mfcc(segment, sample_rate, n_mfcc=13)

# DTW 거리 계산
dtw_score = compute_dtw_distance(mfcc, ref_mfcc_list)
```

### 4. AutoEncoder 손실 계산
```python
# 스펙트로그램 복원 손실
ae_loss = calculate_ae_loss(spectrogram, autoencoder)
```

### 5. 하이브리드 점수 계산
```python
# 동적 임계값 기반 최종 점수
final_score = calculate_final_score_hybrid(
    ae_loss, dtw_score, ae_threshold_dynamic, dtw_threshold_dynamic, 
    ae_weight, dtw_weight
)
```

## 문제 해결

### 1. 일반적인 오류

#### ImportError: No module named 'librosa'
```bash
pip install librosa
```

#### Google Cloud 인증 오류
- 서비스 계정 키 파일 경로 확인
- 키 파일에 올바른 권한이 있는지 확인

#### 큐 크기 초과 오류
- `config.json`에서 큐 크기 조정
- 처리 속도 향상을 위해 시스템 리소스 확인

### 2. 로그 확인
```bash
# 최근 로그 확인
tail -n 100 logs/app.log

# 오류 로그만 확인
grep "ERROR" logs/app.log
```

### 3. 프로세스 상태 확인
```bash
# 실행 중인 Python 프로세스 확인
ps aux | grep python

# Windows
tasklist | findstr python
```

### 4. 스펙트로그램 생성 실패
- matplotlib 백엔드 설정 확인
- 메모리 부족 시 이미지 크기 조정

### 5. DTW 계산 오류
- MFCC 차원 일치 확인
- 참조 오디오 품질 검증

### 6. 성능 지표 극단값
- 충분한 데이터 확보
- 임계값 설정 조정

## 성능 최적화

### 1. 시스템 리소스
- CPU: 4코어 이상 권장
- RAM: 8GB 이상 권장
- 디스크: SSD 권장

### 2. 설정 최적화
- `config.json`에서 큐 크기 조정
- 로그 레벨을 INFO에서 WARNING으로 변경하여 성능 향상

### 3. 병렬 처리
- 현재 4개 프로세스가 병렬로 실행
- 각 프로세스는 독립적으로 동작하여 최적 성능 제공

### 4. 동적 임계값
- 각 파일별로 AE Loss와 DTW Score의 90th percentile 계산
- 데이터 특성에 맞는 적응적 판단

### 5. 멀티프로세스 처리
- 파일 체크, AI 탐지, 파일 처리를 별도 프로세스로 분리
- 큐 기반 비동기 처리로 실시간 성능 보장

### 6. 메모리 최적화
- 스펙트로그램을 메모리 버퍼에서 처리
- 불필요한 디스크 I/O 최소화

## 로그 및 모니터링

### 로그 파일
- `logs/app.log`: 시스템 전체 로그
- `logs/missing_files.log`: 누락된 파일 기록

### 성능 모니터링
- `performance/performance_YYYY-MM-DD.csv`: 시스템 성능 지표
- CPU, 메모리, 디스크 사용률 실시간 모니터링

## 프로세스 종료

### 1. 안전한 종료
```bash
# Ctrl+C를 눌러 프로그램 종료
```

### 2. 강제 종료 (비상시)
```bash
# Linux/Mac
pkill -f "python main.py"

# Windows
taskkill /f /im python.exe
```

## 개발자 정보

### 1. 모듈 구조
- 각 모듈은 `_if.py` (인터페이스)와 `_proc.py` (프로세스)로 구성
- 함수 기반 설계로 유지보수성 향상
- 큐를 통한 비동기 통신

### 2. 확장 가능성
- 새로운 AI 모델 추가 가능
- 다른 클라우드 스토리지 서비스 연동 가능
- 웹 인터페이스 추가 가능

### 3. 로깅 시스템
- 각 프로세스별 독립적인 로깅
- 로그 레벨 조정 가능
- 파일 기반 로그 저장
