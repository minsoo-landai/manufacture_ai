import time
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PIL import Image 
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import tensorflow as tf
import io
from ..util.util_logger import Logger
from ..util.util_datetime import DateTimeUtil
from ..util.util_file import read_audio_file
from queue import Full


def split_audio(audio_data, sample_rate, clip_duration=3.0, drop_last=True):
    clip_samples = int(clip_duration * sample_rate)
    total_samples = len(audio_data)
    segments = []
    for i in range(0, total_samples, clip_samples):
        segment = audio_data[i:i + clip_samples]
        #if len(segment) >= clip_samples:   # 3초 이상
        #if len(segment) > 0:               # 3초 미만이어도 
        # 학습 파이프라인과 동일하게 처리
        if len(segment) < clip_samples and drop_last:
            continue
        segments.append(segment)
    return segments


def pad_to_length(x, target_len, mode="zero"):
    n = target_len - len(x)
    if n <= 0:
        return x
    if mode == "repeat":
        if len(x) == 0:
            return np.zeros(target_len, dtype=np.float32)
        reps = int(np.ceil(n / len(x)))
        return np.concatenate([x, np.tile(x, reps)[:n]])
    elif mode == "edge":
        val = x[-1] if len(x) > 0 else 0.0
        return np.concatenate([x, np.full(n, val, dtype=x.dtype if len(x)>0 else np.float32)])
    else:
        return np.pad(x, (0, n), mode="constant")


def detect_proc_worker(shared_queue, detect_queue, db_queue, config_info, log_file, log_level, log_format):
    logger = Logger("detector", log_file, log_level, log_format)
    logger.info("detect_proc_worker 시작")

    # 설정을 실시간으로 다시 로드하는 함수
    def get_current_config():
        from ..util.util_config import Config
        return Config("config.json")

    sample_rate   = config_info.get_config("ai.sample_rate")
    n_mels        = config_info.get_config("ai.n_mels")
    hop_length    = config_info.get_config("ai.hop_length")
    ae_weight     = config_info.get_config("detection.ae_weight")
    dtw_weight    = config_info.get_config("detection.dtw_weight")
    ae_threshold  = config_info.get_config("detection.ae_threshold")
    dtw_threshold = config_info.get_config("detection.dtw_threshold")
    model_path    = config_info.get_config("ai.model_path")
    reference_path = config_info.get_config("data.reference_dir")
    clip_duration  = config_info.get_config("detection.clip_duration", 3.0)
    progress_step  = int(config_info.get_config("detection.progress_step", 5))  # % 단위
    n_mfcc        = config_info.get_config("detection.n_mfcc", 13)

    # 초기 margin 설정
    margin = float(config_info.get_config("detection.margin", 1.0))

    # 3초 미만 조각 패딩 처리시 
    pad_short_last = config_info.get_config("detection.pad_short_last", True)
    pad_mode = str(config_info.get_config("detection.pad_mode","zero")).lower() # zero / repeat / edge
    clip_samples = int(round(clip_duration*sample_rate))

    autoencoder = tf.keras.models.load_model(model_path)
    
    # 참조 오디오 로드 및 MFCC 계산
    ref_audio = read_audio_file(reference_path, sample_rate)
    ref_clips = split_audio(ref_audio, sample_rate, clip_duration)
    ref_mfcc_list = []
    for clip in ref_clips:
        if len(clip) >= sample_rate:  # 최소 1초 이상
            mfcc = compute_mfcc(clip, sample_rate, n_mfcc)
            ref_mfcc_list.append(mfcc)
    
    if not ref_mfcc_list:
        logger.error("참조 MFCC 생성 실패")
        return
    
    logger.info(f"참조 MFCC {len(ref_mfcc_list)}개 생성 완료")

    # 파일별 처리 추적 (중복 파일 처리 방지)
    processed_files = set()

    while True:
        try:
            if not shared_queue.empty():
                file_info  = shared_queue.get()
                file_path  = file_info['file_path']
                folder_info = file_info.get('folder_info', {})
                timestamp  = file_info['timestamp']

                # 중복 파일 처리 방지
                if file_path in processed_files:
                    logger.info(f"중복 파일 건너뛰기: {file_path}")
                    continue
                
                processed_files.add(file_path)
                logger.info(f"raw_data 파일 처리 시작: {file_path}")
                
                audio_data = read_audio_file(file_path, sample_rate)
                if audio_data is None or len(audio_data) == 0:
                    logger.warning("오디오 데이터가 비어 있음")
                    continue

                segments = split_audio(audio_data, sample_rate, clip_duration)
                total_segs = len(segments)
                logger.info(f"{total_segs}개의 클립으로 분할됨")

                if total_segs == 0:
                    logger.warning("분할된 세그먼트가 없음")
                    continue

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

                # 각 세그먼트를 개별적으로 처리 (노트북 동일 방식)
                for idx, segment in enumerate(segments):
                    # 세그먼트 패딩
                    if len(segment) < clip_samples and pad_short_last:
                        segment = pad_to_length(segment, clip_samples, pad_mode)

                    # 스펙트로그램 생성
                    spectrogram = audio_to_spectrogram_image(segment, sample_rate, n_mels, hop_length)
                    if spectrogram is None:
                        logger.warning(f"스펙트로그램 생성 실패 - 클립 {idx}")
                        continue

                    # MFCC 기반 DTW 거리 계산
                    mfcc = compute_mfcc(segment, sample_rate, n_mfcc)
                    dtw_score = compute_dtw_distance(mfcc, ref_mfcc_list)
                    
                    # AutoEncoder 복원 손실 계산
                    ae_loss = calculate_ae_loss(spectrogram, autoencoder)
                    
                    # config에서 가져온 임계값 사용
                    # ae_threshold와 dtw_threshold는 이미 위에서 config에서 가져왔음
                    
                    # 노트북 방식: 최종 점수 계산 (AE/임계값 * 가중치 + DTW/임계값 * 가중치)
                    score = (ae_loss / ae_threshold) * ae_weight + (dtw_score / dtw_threshold) * dtw_weight
                    
                    # 실시간으로 최신 margin 값 가져오기
                    current_config = get_current_config()
                    current_margin = float(current_config.get_config("detection.margin", 1.05))
                    
                    clip_result = "불량품" if score > current_margin else "양품"
                    any_bad = any_bad or (clip_result == "불량품")

                    seg_start = round(idx * clip_duration, 3)
                    seg_end   = round(min((idx+1)*clip_duration, len(audio_data)/sample_rate), 3)

                    # 파일명 결정
                    original_filename = os.path.basename(file_path)
                    
                    segment_data = {
                        'unit': 'segment',
                        'timestamp': timestamp,
                        'file_path': file_path,
                        'original_filename': original_filename,
                        'parent_file_name': original_filename,
                        'folder_info': folder_info,
                        'segment_index': idx,
                        'segment_total': total_segs,
                        'segment_start': seg_start,
                        'segment_end': seg_end,
                        'segment_duration': round(seg_end - seg_start, 3),
                        'result': clip_result,
                        'dtw_score': dtw_score,
                        'ae_loss': ae_loss,
                        'final_score': score,
                        'duration': len(audio_data) / sample_rate,
                        'sample_rate': sample_rate
                    }
                    
                    # 평가 큐에만 전송 (DB는 파일 완료 시에만 처리)
                    detect_queue.put(segment_data)
                    logger.info(f"세그먼트 결과 큐 push: idx={idx}, result={clip_result}, final={score:.4f}")

                    # 진행률 계산/로그/전송 (progress_step% 단위로만)
                    pct = int(((idx + 1) * 100) / total_segs)
                    should_log = (pct // progress_step) > (max(last_logged_pct, 0) // progress_step) or (idx + 1 == total_segs)
                    if should_log:
                        last_logged_pct = pct
                        # 파일명 표시
                        display_name = os.path.basename(file_path)
                        msg = f"[진행률] {pct}% ({idx+1}/{total_segs}) - {display_name}"
                        logger.info(msg)
                        try:
                            detect_queue.put({
                                'unit': 'progress',
                                'timestamp': timestamp,
                                'file_path': file_path,
                                'segment_done': idx + 1,
                                'segment_total': total_segs,
                                'progress_pct': pct
                            }, block=False)
                        except Full:
                            pass

                # 파일명 결정
                original_filename = os.path.basename(file_path)
                
                # 파일 완료 신호
                db_queue.put({
                    'unit': 'file_done',
                    'timestamp': timestamp,
                    'file_path': file_path,
                    'original_filename': original_filename,
                    'folder_info': folder_info,
                    'file_result': "불량품" if any_bad else "양품",
                    'segment_total': total_segs,
                    'duration': len(audio_data) / sample_rate,
                    'sample_rate': sample_rate
                })

            time.sleep(0.1)

        except Exception as e:
            logger.error(f"탐지 프로세스 오류: {e}")
            time.sleep(1)

def audio_to_spectrogram_image(audio_data, sample_rate, n_mels, hop_length):
    """Mel 스펙트로그램을 이미지로 변환"""
    try:
        # 기본 파라미터 사용
        S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
        librosa.display.specshow(S_dB, sr=sample_rate)
        plt.axis('off')
        
        # 메모리 버퍼로 저장
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        # PIL 로딩
        img = tf.keras.preprocessing.image.load_img(buf, target_size=(224, 224), color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        
        return img_array
    except Exception as e:
        print(f"스펙트로그램 생성 실패: {e}")
        return None

def compute_mfcc(audio_data, sample_rate, n_mfcc=13):
    """MFCC 특징 벡터 계산"""
    try:
        return librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    except Exception as e:
        print(f"MFCC 계산 실패: {e}")
        return None

def compute_dtw_distance(mfcc, ref_mfcc_list):
    """MFCC 특징 벡터 간의 최소 DTW 거리 계산"""
    try:
        if mfcc is None:
            return float('inf')
        
        min_distance = float('inf')
        for ref_mfcc in ref_mfcc_list:
            if ref_mfcc is not None:
                # librosa의 DTW 알고리즘을 사용하여 유클리드 거리 계산
                D, _ = librosa.sequence.dtw(X=mfcc, Y=ref_mfcc, metric='euclidean')
                distance = D[-1, -1]
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 0.0
    except Exception as e:
        print(f"DTW 계산 실패: {e}")
        return float('inf')

def calculate_ae_loss(spectrogram, autoencoder):
    """AutoEncoder의 복원 손실을 계산하여 이상치 탐지 점수 생성"""
    try:
        if spectrogram is None:
            return float('inf')
        
        # 입력 텐서 형태를 AutoEncoder 모델에 맞게 조정
        if len(spectrogram.shape) == 3:
            # (224, 224, 1) 형태인 경우 배치 차원만 추가
            input_tensor = np.expand_dims(spectrogram, axis=0)
        else:
            # (224, 224) 형태인 경우 배치와 채널 차원 추가
            input_tensor = np.expand_dims(spectrogram, axis=(0, -1))
        
        # AutoEncoder를 통한 복원 수행
        reconstructed = autoencoder.predict(input_tensor, verbose=0)
        reconstructed = np.squeeze(reconstructed)
        
        # 원본과 복원된 이미지 간의 평균 제곱 오차 계산
        original = np.squeeze(spectrogram)
        ae_loss = np.mean((original - reconstructed) ** 2)
        return ae_loss
    except Exception as e:
        print(f"AE Loss 계산 실패: {e}")
        return float('inf')

def calculate_final_score_hybrid(ae_loss, dtw_score, ae_threshold, dtw_threshold, ae_weight, dtw_weight):
    """
    AE Loss와 DTW Score를 가중 평균하여 최종 이상치 탐지 점수 계산
    
    #### 기존 코드 (변경 전)
    try:
        # 각 점수를 해당 임계값으로 정규화하여 0-1 범위의 비율로 변환
        ae_ratio = ae_loss / ae_threshold if ae_threshold > 0 else 0
        dtw_ratio = dtw_score / dtw_threshold if dtw_threshold > 0 else 0
        
        # 가중 평균을 사용하여 최종 이상치 탐지 점수 계산
        final_score = ae_weight * ae_ratio + dtw_weight * dtw_ratio
        return final_score
    except Exception as e:
        print(f"최종 점수 계산 실패: {e}")
        return 1.0
    """
    # 임계갑 비교 (학습 파이프라인과 동일화)
    ae_flag = 1 if ae_loss > ae_threshold else 0
    dtw_flag = 1 if dtw_score > dtw_threshold else 0

    # 둘 중 하나라도 넓으면 불량
    final_flag = 1 if (ae_flag or dtw_flag) else 0

    # raw값 기반 점수 
    final_score = ae_weight * ae_loss + dtw_weight * dtw_score 
    return final_flag, final_score 


# 기존 함수들 (호환성을 위해 유지)
def create_spectrogram(audio_data, sample_rate, n_mels, hop_length):
    """기존 스펙트로그램 생성 함수 (호환성 유지)"""
    try:
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length
        )

        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_db -= np.nanmin(mel_spectrogram_db)

        denom = np.nanmax(mel_spectrogram_db)
        if not np.isfinite(denom) or denom <= 0:
            mel_spectrogram_db = np.zeros_like(mel_spectrogram_db, dtype=np.float32)
        else:
            mel_spectrogram_db = mel_spectrogram_db / denom

        image = Image.fromarray((mel_spectrogram_db * 255).astype(np.uint8))
        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32) / 255.0
        return image_array
    except Exception as e:
        print(f"스펙트로그램 생성 실패: {e}")
        return None

def calculate_dtw(spectrogram, reference_spectrogram):
    """기존 DTW 계산 함수 (호환성 유지)"""
    try:
        if spectrogram is None or reference_spectrogram is None:
            return float('inf')
        dist, _ = fastdtw(spectrogram.flatten(), reference_spectrogram.flatten())
        return dist
    except Exception as e:
        print(f"DTW 계산 실패: {e}")
        return float('inf')

def calculate_final_score(ae_loss, dtw_score, ae_threshold, dtw_threshold, ae_weight, dtw_weight):
    """기존 최종 점수 계산 함수 (호환성 유지)"""
    try:
        ae_ratio = ae_loss / ae_threshold if ae_threshold > 0 else 0
        dtw_ratio = dtw_score / dtw_threshold if dtw_threshold > 0 else 0
        final_score = ae_weight * ae_ratio + dtw_weight * dtw_ratio
        return final_score
    except Exception as e:
        print(f"최종 점수 계산 실패: {e}")
        return 1.0