import time
import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import spectrogram
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from tqdm import tqdm
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
from ..util.util_logger import Logger
from ..util.util_datetime import DateTimeUtil
from queue import Full

def load_label_info(ref_csv_path=None):
    """
    ref 폴더의 CSV 파일에서 파일별 레이블 정보를 로드합니다.
    
    Returns:
        dict: {filename: label} 형태의 딕셔너리, 파일이 없으면 None
    """
    if not ref_csv_path or not os.path.exists(ref_csv_path):
        return None
        
    try:
        df = pd.read_csv(ref_csv_path)
        
        # test 데이터만 필터링 (split == 'test'인 항목들)
        test_data = df[df['split'] == 'test']
        
        # 파일명과 레이블 매핑
        label_mapping = {}
        for _, row in test_data.iterrows():
            csv_filename = row['csv_file']
            label = row['label']
            
            # 레이블을 이진으로 변환 (normal_test=0, abnormal_test=1)
            if 'normal' in label:
                label_mapping[csv_filename] = 0  # 정상
            elif 'abnormal' in label:
                label_mapping[csv_filename] = 1  # 불량
            else:
                label_mapping[csv_filename] = -1  # 알 수 없음
        
        return label_mapping
        
    except Exception as e:
        print(f"⚠ 경고: 레이블 정보 로드 중 오류 발생 - {e}")
        return None

### [1] CSV 파일 하나 로딩 + 표준화
def load_and_standardize_csv(file_path, skiprows=0, header=None, names=['Acc1', 'Acc2', 'Acc3', 'Microphone']):
    """CSV 파일을 로드하고 표준화된 형식으로 변환합니다."""
    # 먼저 헤더를 확인
    df_header = pd.read_csv(file_path, nrows=1)
    
    if 'vibration' in df_header.columns and 'x' in df_header.columns:
        # 헤더가 있는 경우
        df = pd.read_csv(file_path)
        df = df.rename(columns={
            'x': 'Acc1',
            'y': 'Acc2',
            'z': 'Acc3',
            'vibration': 'Vibration',
            'audio': 'Microphone'
        })
    else:
        # 헤더가 없는 경우
        df = pd.read_csv(file_path, skiprows=skiprows, header=header, names=names)
        if len(df.columns) == 4:
            df.columns = ['Acc1', 'Acc2', 'Acc3', 'Microphone']
        elif len(df.columns) >= 3:
            df = df.iloc[:, :3]
            df.columns = ['Acc1', 'Acc2', 'Acc3']
        else:
            raise ValueError(f"{file_path} ➤ 알 수 없는 형식의 CSV입니다.")

    # 숫자형으로 변환하기 전에 문자열 값 제거
    for col in ['Acc1', 'Acc2', 'Acc3']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df[['Acc1', 'Acc2', 'Acc3']]

### [2] 시계열 슬라이싱
def split_into_windows(series, window_size, step):
    """시계열 데이터를 윈도우로 분할합니다."""
    return [series[i:i+window_size] for i in range(0, len(series) - window_size + 1, step)]

### 탐지용 전처리
def load_multiple_csvs_from_folder(folder_path, fs=1000, duration=3):
    """
    탐지용 - 폴더 내 CSV들을 개별 처리해 (filename, vib_xyz_dict) 리스트로 반환
    """
    results = []
    win_len = fs * duration
    step = win_len

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            try:
                df = load_and_standardize_csv(filepath)
                vib_xyz_dict = {
                    "Acc1": split_into_windows(df["Acc1"].values, win_len, step),
                    "Acc2": split_into_windows(df["Acc2"].values, win_len, step),
                    "Acc3": split_into_windows(df["Acc3"].values, win_len, step),
                }
                results.append((filename, vib_xyz_dict))
            except Exception as e:
                print(f"⚠ 무시된 파일: {filename} ({e})")
    return results

### 학습용 여러 csv 통합 전처리
def load_multiple_csvs_for_training(folder_path, fs=1000, duration=3):
    """
    학습용 - 폴더 내 모든 CSV를 하나로 병합된 vib_xyz_dict로 반환
    """
    acc1_all, acc2_all, acc3_all = [], [], []
    win_len = fs * duration
    step = win_len

    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".csv"):
            path = os.path.join(folder_path, fname)
            try:
                df = load_and_standardize_csv(path)
                acc1 = split_into_windows(df["Acc1"].values, win_len, step)
                acc2 = split_into_windows(df["Acc2"].values, win_len, step)
                acc3 = split_into_windows(df["Acc3"].values, win_len, step)

                acc1_all.extend(acc1)
                acc2_all.extend(acc2)
                acc3_all.extend(acc3)
            except Exception as e:
                print(f"⚠ 무시된 학습 파일: {fname} ({e})")

    vib_xyz_dict = {
        "Acc1": acc1_all,
        "Acc2": acc2_all,
        "Acc3": acc3_all
    }
    return vib_xyz_dict

def load_and_preprocess_vibration_data_from_folder(folder_path, fs=16, duration=3):
    """
    폴더 내의 모든 CSV 파일을 로드하고 각 파일을 개별적으로 처리합니다.
    각 파일이 이미 3초 데이터라면 파일별로 하나의 윈도우로 처리합니다.

    Args:
        folder_path (str): CSV 파일들이 있는 폴더 경로
        fs (int): 샘플링 주파수 (Hz) - 기본값을 16Hz로 변경
        duration (int): 윈도우 길이 (초)

    Returns:
        dict: {"Acc1": [...], "Acc2": [...], "Acc3": [...], "filenames": [...]}
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    csv_files.sort()  # 파일명 순서대로 정렬
    
    print(f"▶ {len(csv_files)}개의 CSV 파일을 찾았습니다.")
    
    all_acc1, all_acc2, all_acc3 = [], [], []
    all_filenames = []  # 파일명 추적용
    
    for csv_file in tqdm(csv_files, desc="CSV 파일 로딩 중"):
        try:
            # CSV 파일 로드 (헤더가 있는 경우)
            df = pd.read_csv(csv_file)
            
            # 컬럼명 확인 및 매핑
            if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                # 새로운 형식: x, y, z 컬럼
                acc1 = df['x'].values.astype(float)
                acc2 = df['y'].values.astype(float)
                acc3 = df['z'].values.astype(float)
            elif 'Acc1' in df.columns and 'Acc2' in df.columns and 'Acc3' in df.columns:
                # 기존 형식: Acc1, Acc2, Acc3 컬럼
                acc1 = df['Acc1'].values.astype(float)
                acc2 = df['Acc2'].values.astype(float)
                acc3 = df['Acc3'].values.astype(float)
            else:
                print(f"⚠ 경고: {csv_file}에서 적절한 컬럼을 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # 파일별로 개별 처리
            filename = os.path.basename(csv_file)
            
            # 각 파일을 하나의 윈도우로 처리
            all_acc1.append(acc1)
            all_acc2.append(acc2)
            all_acc3.append(acc3)
            all_filenames.append(filename)
            
        except Exception as e:
            print(f"⚠ 오류: {csv_file} 로딩 중 오류 발생 - {e}")
            continue
    
    if not all_acc1:
        raise ValueError("로드된 데이터가 없습니다. 폴더 경로와 CSV 파일 형식을 확인해주세요.")
    
    print(f"▶ 총 {len(all_acc1)}개 파일 로드 완료")
    
    # 각 파일의 데이터 길이 확인
    file_lengths = [len(acc1) for acc1 in all_acc1]
    print(f"▶ 파일별 데이터 길이: 최소 {min(file_lengths)}, 최대 {max(file_lengths)}")
    
    # 윈도우 크기 확인 (실제 데이터에 맞게 조정)
    expected_win_len = fs * duration  # 16Hz × 3초 = 48개 샘플
    actual_min_len = min(file_lengths)
    
    # 실제 데이터 길이에 맞게 윈도우 크기 조정
    if actual_min_len < expected_win_len:
        print(f"⚠ 실제 데이터 길이({actual_min_len})가 예상 길이({expected_win_len})보다 짧습니다.")
        print(f"⚠ 윈도우 크기를 {actual_min_len}개 샘플로 조정합니다.")
        win_len = actual_min_len
    else:
        win_len = expected_win_len
    
    print(f"▶ 최종 윈도우 크기: {win_len}개 샘플")
    
    # 데이터 길이가 윈도우 크기와 다른 경우 처리
    processed_acc1, processed_acc2, processed_acc3, processed_filenames = [], [], [], []
    
    for i, (acc1, acc2, acc3, filename) in enumerate(zip(all_acc1, all_acc2, all_acc3, all_filenames)):
        if len(acc1) >= win_len:
            # 윈도우 크기만큼만 사용
            processed_acc1.append(acc1[:win_len])
            processed_acc2.append(acc2[:win_len])
            processed_acc3.append(acc3[:win_len])
            processed_filenames.append(filename)
        else:
            # 윈도우 크기보다 작으면 패딩 또는 건너뛰기
            print(f"⚠ 경고: {filename}의 데이터가 너무 짧습니다 ({len(acc1)} < {win_len}). 건너뜁니다.")
    
    vib_xyz_dict = {
        "Acc1": processed_acc1,
        "Acc2": processed_acc2,
        "Acc3": processed_acc3,
        "filenames": processed_filenames,
    }

    print(f"▶ {len(vib_xyz_dict['Acc1'])}개의 윈도우 생성 완료")
    return vib_xyz_dict

def load_and_preprocess_vibration_data(csv_file, fs=1000, duration=3):
    """개별 CSV 파일을 로드하고 슬라이딩 윈도우로 분할합니다."""
    try:
        df = load_and_standardize_csv(csv_file)
        
        # 슬라이딩 윈도우로 분할
        window_size = fs * duration
        step_size = window_size  # 비중첩
        
        acc1_windows = []
        acc2_windows = []
        acc3_windows = []
        
        # 데이터가 윈도우 크기보다 작으면 전체 데이터를 하나의 윈도우로 사용
        if len(df) <= window_size:
            acc1_windows.append(df['Acc1'].values)
            acc2_windows.append(df['Acc2'].values)
            acc3_windows.append(df['Acc3'].values)
        else:
            for i in range(0, len(df) - window_size + 1, step_size):
                acc1_windows.append(df['Acc1'].values[i:i + window_size])
                acc2_windows.append(df['Acc2'].values[i:i + window_size])
                acc3_windows.append(df['Acc3'].values[i:i + window_size])
        
        return {
            "Acc1": acc1_windows,
            "Acc2": acc2_windows,
            "Acc3": acc3_windows
        }
        
    except Exception as e:
        print(f"⚠ 경고: {csv_file} 처리 중 오류 - {e}")
        return None

# 스펙트로그램 변환
def vibration_to_spectrogram_array(signal, fs=16):
    """
    #### 1D 신호 >>> 2D 이미지(Time-Frequency Spectrogram) 변환
    : 시간-주파수 영역으로 변환 위해 STFT(Short-Time Fouier Transform) 기반 스펙트트럼 사용

    1) 파라미터 설정값 - 데이터 길이에 맞게 동적 조정
    - fs : 실제 샘플링 주파수 (16Hz)
    - nperseg : 데이터 길이에 맞게 조정 (최소 4개 샘플)
    - noverlap : nperseg의 절반 이하로 설정

    ** 제조 설비 진동은 보통 10ms ~ 수백 ms 사이 순간적 이상 이벤트 발생
      0.25초 단위는 한 주기 또는 이상 발생 패턴 하나를 포착하기에 충분한 시간 분해능 가짐
    * 50% overlap(추가 고민 필요) : 고주파 불량 신호가 짧은시간 동안 고에너지를 띌 경우 겹침 처리로 누락 방지
    """
    # 데이터 길이에 맞게 파라미터 동적 조정
    signal_length = len(signal)
    
    # nperseg를 데이터 길이의 절반으로 설정 (최소 4개)
    nperseg = max(4, min(signal_length // 2, 256))
    
    # noverlap을 nperseg의 절반으로 설정 (nperseg보다 작게)
    noverlap = max(1, nperseg // 2)
    
    # 데이터가 너무 짧으면 전체 데이터를 사용
    if signal_length < nperseg:
        nperseg = signal_length
        noverlap = 0
    
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # 중심 주파수 계산 (weighted mean)
    center_freq = np.sum(f[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-8)

    # 대역폭 계산 (variance) : 신호가 퍼진 정도 (주파수 분산)
    #### 정상 신호는 좁은 대역, 이상 신호는 넓은 대역에 퍼짐 
    bandwidth = np.sqrt(np.sum((f[:, None] - center_freq)**2 * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-8))

    # 로그 변환 → 고주파 강조 (dB 단위로 변환)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)

    # 정규화 (전체 기준 or 고정 min/max)
    Sxx_log = (Sxx_log - np.min(Sxx_log)) / (np.max(Sxx_log) - np.min(Sxx_log))

    # (선택) scalogram ridge 검출 및 비교용
    # ridge_curve = np.argmax(Sxx_log, axis=0)

    Sxx_resized = tf.image.resize(Sxx_log[..., np.newaxis], (224, 224)).numpy()

    return Sxx_resized.astype(np.float32)

### 이미지 저장
def generate_images_and_save(vib_xyz_dict, save_path="vib_images.npy", fs=1000):
    """진동 데이터를 스펙트로그램 이미지로 변환하고 저장합니다."""
    total = len(vib_xyz_dict["Acc1"])
    npy_writer = np.lib.format.open_memmap(save_path, mode='w+', dtype=np.float32, shape=(total, 224, 224, 1))
    for i in tqdm(range(total), desc="스펙트로그램 생성 중", ncols=100):
        merged = np.mean([vib_xyz_dict["Acc1"][i], vib_xyz_dict["Acc2"][i], vib_xyz_dict["Acc3"][i]], axis=0)
        npy_writer[i] = vibration_to_spectrogram_array(merged, fs)
        if i % 100 == 0: gc.collect()
    del npy_writer
    gc.collect()

### AutoEncoder 모델 정의
def build_cnn_autoencoder():
    """
    1. 구조
      1) 이미지 크기 : (224 * 224 * 1)
      2) Encoder : Conv(16) > Pool > Conv(8) > upsample > Conv(4) 
      3) Decoder : upsample > Conv(8) > upsample > Conv(1, sigmoid)
      4) Activation : ReLU / 출력 Sigmoid 
      5) Loss : MSE 
    """
    input_img = Input(shape=(224, 224, 1))
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(8, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    encoded = layers.Conv2D(4, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(encoded)
    x = layers.Conv2D(8, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)
    model = models.Model(input_img, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

def compute_2d_dtw(img1, img2):
    """
    두 개의 2D 이미지 (스펙트로그램)를 시간 축 기준 주파수 벡터 시퀀스로 보고 DTW 거리 계산
    img1, img2 shape: (freq, time)
    - 시간 벡터 : 주파수 스펙트럼 
    - 앞선 log 변환 통해 고주파 잔진동 충분히 반영. (DTW 신호의 전체적 모양을 제대로 인식하게)
    

    * 1D 아닌 2D 스펙트로그램 기반 시간별 주파수 벡터 기반 시계열 매칭 
    * 진동 신호 이상은 주로 특정 주파수 대역 에너지 변화로 발생. 
      전체 waveform 모양보다는 시간별 주파수 스펙트럼의 변화가 더 유의미. 
    * AE는 공간적 손실만 평가함. 시간축 왜곡이나 진동 패턴 왜곡까지 평가하기 위해 DTW 보완 

    """
    seq1 = [img1[:, i] for i in range(img1.shape[1])]  # 시간축 따라 자르기
    seq2 = [img2[:, i] for i in range(img2.shape[1])]
    dist, _ = fastdtw(seq1, seq2, dist=euclidean)  # DTW with vector distance
    return dist

### 학습 및 기준 점수 설정
def train_vibration_cnn_ae_dtw(
    vib_xyz_dict,
    model_path="vib_cnn_ae.keras",
    threshold_path="vib_cnn_thresh.npy",
    cache_path="vib_images.npy",
    fs=1000
):
    """진동 데이터로 CNN AutoEncoder + DTW 모델을 학습합니다."""
    # 1. 이미지 캐시 생성
    if not os.path.exists(cache_path):
        generate_images_and_save(vib_xyz_dict, cache_path, fs)

    all_images = np.load(cache_path, mmap_mode='r')
    print(f"▶ 총 {len(all_images)}개 이미지 로드 완료")

    # 2. 데이터 분할
    X_train, X_val, idx_train, idx_val = train_test_split(
        all_images, np.arange(len(all_images)), test_size=0.2, random_state=42
    )

    # 3. 모델 정의 및 학습
    print("▶ AutoEncoder 모델 학습 시작")
    model = build_cnn_autoencoder()
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=300,
        batch_size=8,
        shuffle=True,
        verbose=1,
        callbacks=[early_stop]
    )

    # 4. 모델 저장 (Keras 포맷 사용 권장)
    model.save(model_path)
    print(f"▶ 모델 저장 완료: {model_path}")

    # 5. 복원 예측 (배치 처리)
    print("▶ 복원 예측 시작")
    recon_val = model.predict(X_val, batch_size=8, verbose=1)

    # 6. AE Loss 계산
    ae_losses = np.mean((X_val - recon_val) ** 2, axis=(1, 2, 3))

    # 7. DTW 거리 계산 (mean projection → 1D 비교)
    dtw_distances = []
    for i, idx in enumerate(idx_val):
        # 원본 merged 시그널
        merged = np.mean([
            vib_xyz_dict["Acc1"][idx],
            vib_xyz_dict["Acc2"][idx],
            vib_xyz_dict["Acc3"][idx]
        ], axis=0)

        # 원본 시그널 → 스펙트로그램 이미지
        original_img = vibration_to_spectrogram_array(merged, fs).squeeze()  # shape: (224, 224)
        recon_img = recon_val[i].squeeze()  # shape: (224, 224)

        # 2D DTW 계산
        dtw_score = compute_2d_dtw(original_img, recon_img)
        dtw_distances.append(dtw_score)

    # 8. 정규화 및 threshold 계산
    ae_norm = (ae_losses - np.min(ae_losses)) / (np.max(ae_losses) - np.min(ae_losses) + 1e-8)
    dtw_norm = (np.array(dtw_distances) - np.min(dtw_distances)) / (np.max(dtw_distances) - np.min(dtw_distances) + 1e-8)

    # Final Score 계산 (AE Loss + DTW score)
    final_scores = 0.5 * ae_norm + 0.5 * dtw_norm
    threshold = np.percentile(final_scores, 90)
    np.save(threshold_path, threshold)
    print(f"▶ Threshold 저장 완료: {threshold_path} (90% 기준 = {threshold:.6f})")

    return model, threshold

def detect_vibration_anomaly_cnn_dtw(
    test_folder_path, 
    model_path,
    threshold_path,
    save_csv_path=None,
    label_mapping=None, 
    ae_weight=0.5, 
    dtw_weight=0.5, 
    margin=1.0
):
    """
    CNN AutoEncoder와 DTW를 결합한 진동 이상 탐지를 수행합니다.
    """
    print("🔍 진동 이상 탐지 시작")
    
    # 모델 및 임계값 로드
    try:
        model = tf.keras.models.load_model(model_path)
        threshold = np.load(threshold_path)
        print(f"📁 모델 로드 완료: {model_path}")
        print(f"📁 임계값 로드 완료: {threshold_path} (값: {threshold:.6f})")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None, None, None
    
    # 테스트 데이터 로드
    test_vib_xyz = load_and_preprocess_vibration_data_from_folder(test_folder_path)
    if not test_vib_xyz or len(test_vib_xyz["Acc1"]) == 0:
        print("❌ 테스트 데이터를 찾을 수 없습니다.")
        return None, None, None
    
    print(f"▶ 테스트 데이터: {len(test_vib_xyz['Acc1'])}개 윈도우")
    
    # 1단계: 모든 데이터의 AE Loss와 DTW 거리 계산
    print("1단계: 모든 데이터의 특성 계산 중...")
    all_losses = []
    all_dtws = []
    
    for i in tqdm(range(len(test_vib_xyz["Acc1"])), desc="특성 계산 중"):
        # 가장 큰 진폭을 가진 채널 선택
        acc1_spec = vibration_to_spectrogram_array(test_vib_xyz["Acc1"][i])
        acc2_spec = vibration_to_spectrogram_array(test_vib_xyz["Acc2"][i])
        acc3_spec = vibration_to_spectrogram_array(test_vib_xyz["Acc3"][i])
        
        max_amp_acc1 = np.max(np.abs(test_vib_xyz["Acc1"][i]))
        max_amp_acc2 = np.max(np.abs(test_vib_xyz["Acc2"][i]))
        max_amp_acc3 = np.max(np.abs(test_vib_xyz["Acc3"][i]))
        
        if max_amp_acc1 >= max_amp_acc2 and max_amp_acc1 >= max_amp_acc3:
            spec = acc1_spec
            signal = test_vib_xyz["Acc1"][i]
        elif max_amp_acc2 >= max_amp_acc3:
            spec = acc2_spec
            signal = test_vib_xyz["Acc2"][i]
        else:
            spec = acc3_spec
            signal = test_vib_xyz["Acc3"][i]
            
        if spec is None:
            all_losses.append(np.nan)
            all_dtws.append(np.nan)
            continue
        
        # AutoEncoder Loss 계산
        reconstructed = model.predict(spec[np.newaxis, ...], verbose=0)
        ae_loss = np.mean((spec - reconstructed[0]) ** 2)
        all_losses.append(ae_loss)
        
        # DTW 거리 계산
        avg_dtw = 0
        try:
            segments = np.array_split(signal, 4)
            dtw_distances = []
            
            for j in range(len(segments)-1):
                distance, _ = fastdtw(segments[j], segments[j+1], dist=euclidean)
                dtw_distances.append(distance)
            
            avg_dtw = np.mean(dtw_distances) if dtw_distances else 0
            all_dtws.append(avg_dtw)
        except:
            all_dtws.append(0)
    
    # 2단계: 동적 임계값 계산 (오디오 모델 방식)
    print("2단계: 동적 임계값 계산 중...")
    valid_losses = [l for l in all_losses if not np.isnan(l)]
    valid_dtws = [d for d in all_dtws if not np.isnan(d)]
    
    if len(valid_losses) > 0:
        ae_threshold_dynamic = np.percentile(valid_losses, 90)
        print(f"동적 AE 임계값: {ae_threshold_dynamic:.6f}")
    else:
        ae_threshold_dynamic = threshold
        print(f"기본 AE 임계값 사용: {threshold:.6f}")
    
    if len(valid_dtws) > 0:
        dtw_threshold_dynamic = np.percentile(valid_dtws, 90)
        print(f"동적 DTW 임계값: {dtw_threshold_dynamic:.6f}")
    else:
        dtw_threshold_dynamic = 1000.0
        print(f"기본 DTW 임계값 사용: 1000.0")
    
    # 3단계: 최종 이상 탐지
    print("3단계: 최종 이상 탐지 중...")
    losses = []
    dtws = []
    results = []
    
    for i in tqdm(range(len(test_vib_xyz["Acc1"])), desc="이상 탐지 중"):
        if np.isnan(all_losses[i]) or np.isnan(all_dtws[i]):
            losses.append(np.nan)
            dtws.append(np.nan)
            results.append(-1)  # 오류
            continue
        
        ae_loss = all_losses[i]
        avg_dtw = all_dtws[i]
        
        losses.append(ae_loss)
        dtws.append(avg_dtw)
        
        # 오디오 모델 방식: 정규화된 점수 계산
        ae_ratio = ae_loss / ae_threshold_dynamic if ae_threshold_dynamic > 0 else 0
        dtw_ratio = avg_dtw / dtw_threshold_dynamic if dtw_threshold_dynamic > 0 else 0
        
        # 하이브리드 점수 계산
        score = ae_weight * ae_ratio + dtw_weight * dtw_ratio
        
        # 이상 판정
        result = 1 if score > margin else 0
        results.append(result)
    
    # 결과 저장
    if save_csv_path:
        total = len(results)
        
        # 파일별 레이블 매핑
        file_labels = []
        for i in range(total):
            filename = test_vib_xyz["filenames"][i][0] if i < len(test_vib_xyz["filenames"]) else "unknown"
            if label_mapping and filename in label_mapping:
                file_labels.append(label_mapping[filename])
            else:
                file_labels.append(-1)
        
        # 점수 계산
        scores = []
        for i in range(total):
            if not np.isnan(losses[i]):
                normalized_ae = losses[i] / threshold if threshold > 0 else 0
                normalized_dtw = dtws[i] / 1000 if not np.isnan(dtws[i]) else 0
                score = ae_weight * normalized_ae + dtw_weight * normalized_dtw
                scores.append(score)
            else:
                scores.append(np.nan)
        
        df = pd.DataFrame({
            "index": list(range(total)),
            "filename": [test_vib_xyz["filenames"][i][0] if i < len(test_vib_xyz["filenames"]) else "unknown" for i in range(total)],
            "loss": losses,
            "dtw": dtws,
            "score": scores,
            "anomaly": results,
            "true_label": file_labels
        })
        
        df.to_csv(save_csv_path, index=False)
        print(f"📁 탐지 결과 저장 완료: {save_csv_path}")
        
        # 성능 평가 (레이블이 있는 경우)
        if label_mapping:
            valid_labels = df[df['true_label'] != -1]
            if len(valid_labels) > 0:
                y_true = valid_labels['true_label'].values
                y_pred = valid_labels['anomaly'].values
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                print(f"\n📊 성능 지표:")
                print(f"Accuracy : {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall   : {recall:.4f}")
                print(f"F1-score : {f1:.4f}")

    return losses, dtws, results

def process_vibration_data(file_path, config):
    """진동 데이터 처리 개선"""
    try:
        # 파일 존재 확인
        if not os.path.exists(file_path):
            logger.error(f"파일이 존재하지 않습니다: {file_path}")
            return None
            
        # 파일 크기 확인
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.warning(f"빈 파일입니다: {file_path}")
            return None
            
        # CSV 파일 읽기
        try:
            data = pd.read_csv(file_path, skiprows=config.get_config("vibration.skiprows", 0))
        except Exception as e:
            logger.error(f"CSV 파일 읽기 실패: {file_path}, 오류: {e}")
            return None
            
        # 데이터 유효성 검사
        if data.empty:
            logger.warning(f"빈 데이터: {file_path}")
            return None
            
        # 필요한 컬럼 확인
        required_columns = ['x', 'y', 'z']  # 또는 실제 사용하는 컬럼명
        if not all(col in data.columns for col in required_columns):
            logger.warning(f"필요한 컬럼이 없습니다: {file_path}, 컬럼: {data.columns.tolist()}")
            return None
            
        # 데이터 전처리
        fs = config.get_config("vibration.fs", 16)
        duration = config.get_config("vibration.duration", 3)
        
        # 시계열 데이터로 변환
        vibration_data = data[required_columns].values
        
        # 세그먼트 분할
        segment_length = int(fs * duration)
        segments = []
        
        for i in range(0, len(vibration_data) - segment_length + 1, segment_length):
            segment = vibration_data[i:i + segment_length]
            segments.append(segment)
            
        if not segments:
            logger.warning(f"세그먼트를 생성할 수 없습니다: {file_path}")
            return None
            
        return {
            'file_path': file_path,
            'segments': segments,
            'fs': fs,
            'duration': duration
        }
        
    except Exception as e:
        logger.error(f"진동 데이터 처리 중 오류 발생: {file_path}, 오류: {e}")
        return None

def detect_proc_worker(vib_shared_queue, detect_queue, db_queue, config_info, log_file, log_level, log_format):
    """
    진동 AI 탐지 프로세스 워커 함수
    """
    try:
        from src.util.util_logger import Logger
        from queue import Full
        
        logger = Logger("vib_detector", log_file, log_level, log_format)
        logger.info("vib_detect_proc_worker 시작")

        # 설정을 실시간으로 다시 로드하는 함수
        def get_current_config():
            from src.util.util_config import Config
            return Config("config.json")

        # 진동 모델 설정
        fs = config_info.get_config("vibration.fs", 16)
        duration = config_info.get_config("vibration.duration", 3)
        skiprows = config_info.get_config("vibration.skiprows", 0)
        model_path = config_info.get_config("vibration.model_path", "models/vib_cnn_ae.keras")
        threshold_path = config_info.get_config("vibration.threshold_path", "models/vib_cnn_thresh.npy")
        ae_weight = config_info.get_config("vibration.ae_weight", 0.5)
        dtw_weight = config_info.get_config("vibration.dtw_weight", 0.5)
        progress_step = int(config_info.get_config("vibration.progress_step", 5))
        
        # 학습 코드와 동일한 정규화 파라미터 사용 (config.json에서 로드)
        ae_min = config_info.get_config("vibration.ae_min", 0.0)
        ae_max = config_info.get_config("vibration.ae_max", 1.0)
        dtw_min = config_info.get_config("vibration.dtw_min", 0.0)
        dtw_max = config_info.get_config("vibration.dtw_max", 1.0)
        
        # 학습에서 계산된 최종 임계값 사용
        final_threshold = config_info.get_config("vibration.final_threshold", 0.5)
        
        logger.info(f"진동 설정 - fs: {fs}Hz, duration: {duration}s")
        logger.info(f"정규화 파라미터 - AE: {ae_min:.6f}~{ae_max:.6f}, DTW: {dtw_min:.6f}~{dtw_max:.6f}")
        logger.info(f"최종 임계값: {final_threshold:.6f}")

        # 초기 margin 설정
        margin = float(config_info.get_config("vibration.margin", 1.0))

        # 모델 로드
        try:
            autoencoder = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"진동 모델 로드 완료: {model_path}")
            logger.info(f"진동 임계값 설정: final_threshold={final_threshold}")
        except Exception as e:
            logger.error(f"진동 모델 로드 실패: {e}")
            return

        # 중복 처리 방지를 위한 세트 초기화
        processed_segments = set()

        while True:
            try:
                if not vib_shared_queue.empty():
                    file_info = vib_shared_queue.get()
                    file_path = file_info['file_path']
                    folder_info = file_info.get('folder_info', {})
                    timestamp = file_info['timestamp']

                    logger.info(f"진동 데이터 파일 처리 시작: {file_path}")
                    
                    # 진동 데이터 로드 및 전처리
                    vib_xyz_dict = load_and_preprocess_vibration_data_from_folder(
                        folder_path=os.path.dirname(file_path),
                        fs=fs,
                        duration=duration
                    )
                
                    if vib_xyz_dict is None or len(vib_xyz_dict["Acc1"]) == 0:
                        logger.warning("진동 데이터가 비어 있음")
                        continue

                    total_segs = len(vib_xyz_dict["Acc1"])
                    logger.info(f"{total_segs}개의 진동 세그먼트로 분할됨")

                    # 시작 진행률(0%) 알림
                    try:
                        detect_queue.put({
                            'unit': 'progress',
                            'timestamp': timestamp,
                            'file_path': file_path,
                            'segment_done': 0,
                            'segment_total': total_segs,
                            'progress_pct': 0
                        }, block=False)
                    except Full:
                        pass
                    
                    # 파일명 표시
                    display_name = os.path.basename(file_path)
                    logger.info(f"[진행률] 0% (0/{total_segs}) - {display_name}")

                    # 노트북 방식: 3초 단위로 실시간 처리
                    any_bad = False
                    last_logged_pct = -1

                    # 각 세그먼트를 개별적으로 처리
                    for idx in range(total_segs):
                        # 중복 처리 방지
                        segment_key = f"{file_path}_{idx}"
                        if segment_key in processed_segments:
                            logger.info(f"중복 진동 세그먼트 건너뛰기: {segment_key}")
                            continue
                        
                        processed_segments.add(segment_key)
                        try:
                            # 3축 진동 데이터 병합
                            merged = np.mean([
                                vib_xyz_dict["Acc1"][idx],
                                vib_xyz_dict["Acc2"][idx],
                                vib_xyz_dict["Acc3"][idx]
                            ], axis=0)

                            # 스펙트로그램 생성
                            img = vibration_to_spectrogram_array(merged, fs)
                            if img is None:
                                logger.warning(f"스펙트로그램 생성 실패 - 클립 {idx}")
                                continue

                            # AutoEncoder 복원
                            recon = autoencoder.predict(np.expand_dims(img, axis=0), verbose=0)[0]

                            # AE Loss 계산
                            ae_loss = np.mean((img - recon) ** 2)

                            # DTW 거리 계산
                            dtw_dist = compute_2d_dtw(img.squeeze(), recon.squeeze())

                            # 학습 코드와 동일한 정규화 방식 적용
                            ae_norm = (ae_loss - ae_min) / (ae_max - ae_min + 1e-8)
                            dtw_norm = (dtw_dist - dtw_min) / (dtw_max - dtw_min + 1e-8)
                            
                            # 학습 코드와 동일한 최종 점수 계산
                            final_score = ae_weight * ae_norm + dtw_weight * dtw_norm

                            # 학습에서 계산된 임계값으로 이상 탐지
                            is_anomaly = final_score > final_threshold
                            
                            logger.debug(f"세그먼트 {idx} - AE: {ae_loss:.6f}->{ae_norm:.6f}, DTW: {dtw_dist:.6f}->{dtw_norm:.6f}, 최종: {final_score:.6f}, 임계값: {final_threshold:.6f}")

                            if is_anomaly:
                                any_bad = True

                            # 세그먼트 결과를 큐에 전송
                            try:
                                detect_queue.put({
                                    'unit': 'segment',
                                    'timestamp': timestamp,
                                    'file_path': file_path,
                                    'folder_info': folder_info,
                                    'segment_index': idx,
                                    'segment_total': total_segs,
                                    'result': "불량품" if is_anomaly else "정상품",
                                    'ae_loss': float(ae_loss),
                                    'dtw_score': float(dtw_dist),
                                    'final_score': float(final_score),
                                    'duration': duration,
                                    'sample_rate': fs
                                }, block=False)
                                logger.info(f"진동 세그먼트 결과 큐 push: idx={idx}, result={'불량품' if is_anomaly else '정상품'}, final={final_score:.4f}")
                            except Full:
                                logger.warning(f"진동 세그먼트 결과 큐 전송 실패: {idx}")
                                pass

                            # 진행률 업데이트
                            current_pct = int((idx + 1) / total_segs * 100)
                            if current_pct >= last_logged_pct + progress_step:
                                last_logged_pct = current_pct
                                logger.info(f"[진행률] {current_pct}% ({idx + 1}/{total_segs}) - {display_name}")

                                # 진행률 큐에 전송
                                try:
                                    detect_queue.put({
                                        'unit': 'progress',
                                        'timestamp': timestamp,
                                        'file_path': file_path,
                                        'segment_done': idx + 1,
                                        'segment_total': total_segs,
                                        'progress_pct': current_pct
                                    }, block=False)
                                except Full:
                                    pass

                        except Exception as e:
                            logger.error(f"세그먼트 {idx} 처리 중 오류: {e}")
                            continue

                    # 최종 결과 전송
                    try:
                        detect_queue.put({
                            'unit': 'vibration',
                            'timestamp': timestamp,
                            'file_path': file_path,
                            'folder_info': folder_info,
                            'segment_total': total_segs,
                            'anomaly_detected': any_bad,
                            'margin': margin
                        }, block=False)
                    except Full:
                        pass

                    # DB 큐에도 전송
                    try:
                        db_queue.put({
                            'unit': 'vibration',
                            'timestamp': timestamp,
                            'file_path': file_path,
                            'folder_info': folder_info,
                            'segment_total': total_segs,
                            'anomaly_detected': any_bad,
                            'margin': margin
                        }, block=False)
                    except Full:
                        pass

                    logger.info(f"진동 데이터 처리 완료: {file_path} (이상: {'감지됨' if any_bad else '없음'})")

            except Exception as e:
                logger.error(f"진동 데이터 처리 중 오류: {e}")
                continue
                
    except Exception as e:
        print(f"진동 탐지 워커 프로세스 오류: {e}")
        if 'logger' in locals():
            logger.error(f"진동 탐지 워커 프로세스 오류: {e}")