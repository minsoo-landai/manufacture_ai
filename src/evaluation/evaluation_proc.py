import time
import os
import shutil
import csv
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ..util.util_logger import Logger
from ..util.util_datetime import DateTimeUtil
import pandas as pd
import os
from datetime import datetime
from ..util.util_datetime import DateTimeUtil
from ..util.util_logger import Logger



def evaluation_proc_worker(detect_queue, config_info, log_file, log_level, log_format):
    """
    파일 처리 및 성능 평가 프로세스 워커
    - 'segment' : 행 단위 저장 (클립 지표 + 누적 지표 + 예측값)
    - 'file_done': 파일 이동/정리
    """
    logger = Logger("sound_file_processor", log_file, log_level, log_format)
    logger.info("=== sound_evaluation_proc_worker 시작 ===")
    logger.info(f"detect_queue 크기: {detect_queue.qsize()}")
    logger.info(f"save_evaluation_results 설정: {config_info.get_config('evaluation.save_evaluation_results', True)}")

    raw_data_dir = config_info.get_config("data.raw_data_dir")
    #split_dir = config_info.get_config("data.split_dir")
    reserved_dir = config_info.get_config("data.sound_reserved_dir")
    save_evaluation_results = config_info.get_config("evaluation.save_evaluation_results", True)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 헤더: 클립 지표 + 누적 지표 + 정답/예측 표시
    clip_csv_headers = [
        'unit','timestamp','file_path','original_filename','parent_file_name','folder_info',
        'segment_index','segment_total','segment_start','segment_end','segment_duration',
        'result','dtw_score','ae_loss','final_score','duration','sample_rate',
        # 행별 정답/예측
        'true_label','pred_label','is_correct',
        # 클립(개별) 지표 - 개별 세그먼트 지표
        'clip_accuracy','clip_precision','clip_recall','clip_f1',
        # 누적 지표 - 현재 파일 내에서 계산
        'cumu_accuracy','cumu_precision','cumu_recall','cumu_f1'
    ]

    today = DateTimeUtil.get_current_date()
    clip_csv = os.path.join(results_dir, f"evaluation_clip_results_{today}.csv")
    
    # CSV 파일이 없으면 헤더와 함께 생성, 있으면 기존 파일 유지
    if not os.path.exists(clip_csv):
        try:
            # results 디렉토리가 없으면 생성
            os.makedirs(results_dir, exist_ok=True)
            
            with open(clip_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=clip_csv_headers)
                w.writeheader()
            logger.info(f"새로운 클립 결과 CSV 파일 생성: {clip_csv}")
            logger.info(f"CSV 헤더 작성 완료: {len(clip_csv_headers)}개 컬럼")
            logger.info(f"CSV 헤더: {clip_csv_headers}")
        except Exception as e:
            logger.error(f"CSV 파일 생성 실패: {e}")
            logger.error(f"생성하려던 경로: {clip_csv}")
            logger.error(f"results_dir 존재 여부: {os.path.exists(results_dir)}")
    else:
        logger.info(f"기존 클립 결과 CSV 파일 사용: {clip_csv}")
        logger.info(f"기존 파일 크기: {os.path.getsize(clip_csv)} bytes")

    # 누적 지표(세그먼트 기준) - 전체 세션 동안 유지
    y_true_seg, y_pred_seg = [], []
    
    # 파일 단위 성능 추적
    file_performance_data = []
    current_file_segments = []
    current_file_name = None
    
    # 중복 처리 방지를 위한 세그먼트 추적
    processed_segments = set()
    
    # 성능 지표 초기화 로깅
    logger.info("성능 지표 추적 시작 - 누적 지표가 실시간으로 계산됩니다")

    while True:
        try:
            if detect_queue.empty():
                time.sleep(0.1)
                continue

            info = detect_queue.get()
            logger.info(f"큐에서 데이터 수신: {info.get('unit', 'unknown')}")
            unit = info.get('unit', 'segment')

            # 진행률 메시지 처리 (터미널 로그)
            if unit == 'progress':
                pct = info.get('progress_pct')
                done = info.get('segment_done')
                total = info.get('segment_total')
                fname = os.path.basename(info.get('file_path', ''))
                logger.info(f"[진행률] {pct}% ({done}/{total}) - {fname}")
                continue

            if unit == 'segment':
                start_t = time.time()
                fp = info['file_path']
                segment_index = info.get('segment_index', 0)
                
                # 중복 처리 방지 (더 강력한 키 사용)
                timestamp = info.get('timestamp', 0)
                segment_key = f"{fp}_{segment_index}_{timestamp}"
                
                # 디버깅을 위한 로그
                logger.debug(f"세그먼트 키 생성: {segment_key}")
                logger.debug(f"이미 처리된 세그먼트 수: {len(processed_segments)}")
                
                if segment_key in processed_segments:
                    logger.info(f"중복 세그먼트 건너뛰기: {segment_key}")
                    continue
                
                processed_segments.add(segment_key)
                logger.info(f"세그먼트 데이터 처리 시작: {fp}, 세그먼트 인덱스: {segment_index}")
                
                # 파일 경로 검증 - raw_data에서 처리하므로 원본 경로 확인
                file_exists = os.path.exists(fp)
                if not file_exists:
                    # raw_data 내에서 파일 찾기
                    filename = os.path.basename(fp)
                    raw_data_path = os.path.join(raw_data_dir, filename)
                    
                    if os.path.exists(raw_data_path):
                        fp = raw_data_path
                        file_exists = True
                        logger.info(f"raw_data에서 파일 발견: {raw_data_path}")
                
                if not file_exists:
                    logger.warning(f"파일 없음(세그먼트): {fp}")
                    create_missing_file_log(fp, logger)
                    continue

                # 파일 변경 감지
                if current_file_name != fp:
                    # 이전 파일의 성능 계산 및 저장
                    if current_file_segments:
                        file_perf = calculate_file_performance(current_file_segments)
                        file_performance_data.append(file_perf)
                    
                    current_file_name = fp
                    current_file_segments = []
                    
                    # 새로운 파일 시작 시 누적 리스트 초기화
                    y_true_seg.clear()
                    y_pred_seg.clear()
                    logger.info(f"새 파일 처리 시작: {fp}")

                # 실제 라벨 추출 (파일 경로 기반)
                t = estimate_true_label(fp)
                
                # 탐지 결과에서 예측 라벨 가져오기 (탐지 프로세스에서 계산된 값 사용)
                result = info.get('result', '양품')
                p = 1 if result == "불량품" else 0
                
                # DTW 점수와 AE Loss 로깅 (디버깅용)
                dtw_score = info.get('dtw_score', 0)
                ae_loss = info.get('ae_loss', 0)
                logger.info(f"세그먼트 {segment_index}: DTW={dtw_score:.2f}, AE={ae_loss:.6f}, Result={result}, True={t}, Pred={p}")
                
                y_true_seg.append(t); y_pred_seg.append(p)
                
                # 현재 파일의 세그먼트 정보 저장
                current_file_segments.append({
                    'true_label': t,
                    'pred_label': p,
                    'segment_index': info.get('segment_index', 0),
                    'result': info.get('result')
                })

                # 클립(해당 행) 지표 - 개별 세그먼트 지표 계산
                clip_acc = 1.0 if t == p else 0.0
                # 개별 세그먼트의 precision, recall, f1 계산
                if t == 1:  # 실제 불량품인 경우
                    clip_prec = 1.0 if p == 1 else 0.0
                    clip_rec = 1.0 if p == 1 else 0.0
                    clip_f1 = 1.0 if p == 1 else 0.0
                else:  # 실제 정상품인 경우
                    clip_prec = 1.0 if p == 0 else 0.0
                    clip_rec = 1.0 if p == 0 else 0.0
                    clip_f1 = 1.0 if p == 0 else 0.0

                # 누적 지표 (현재 파일 내에서만 계산)
                if len(y_true_seg) >= 1:
                    acc = accuracy_score(y_true_seg, y_pred_seg)
                    # precision, recall, f1은 불량품(1)이 있을 때만 계산
                    if 1 in y_true_seg and 0 in y_true_seg:  # 정상과 불량이 모두 있을 때
                        prec = precision_score(y_true_seg, y_pred_seg, zero_division=0)
                        rec  = recall_score(y_true_seg, y_pred_seg, zero_division=0)
                        f1   = f1_score(y_true_seg, y_pred_seg, zero_division=0)
                    elif 1 in y_true_seg:  # 불량만 있는 경우
                        prec = 1.0 if any(y_pred_seg) else 0.0
                        rec = 1.0 if any(y_pred_seg) else 0.0
                        f1 = 1.0 if any(y_pred_seg) else 0.0
                    else:  # 정상만 있는 경우
                        prec = 1.0 if not any(y_pred_seg) else 0.0
                        rec = 1.0 if not any(y_pred_seg) else 0.0
                        f1 = 1.0 if not any(y_pred_seg) else 0.0

                processing_time = time.time() - start_t
                perf_payload = {
                    'true_label': t, 'pred_label': p, 'is_correct': int(t == p),
                    'clip_accuracy': clip_acc, 'clip_precision': clip_prec,
                    'clip_recall': clip_rec, 'clip_f1': clip_f1,
                    'cumu_accuracy': acc, 'cumu_precision': prec,
                    'cumu_recall': rec, 'cumu_f1': f1
                }

                row = create_clip_evaluation_data(info, processing_time, fp, perf_payload)
                logger.info(f"평가 데이터 생성 완료: true_label={t}, pred_label={p}, accuracy={acc:.2f}")
                logger.info(f"CSV 행 키 개수: {len(row.keys())}")
                logger.info(f"CSV 행 키: {list(row.keys())}")

                if save_evaluation_results:
                    logger.info(f"CSV 파일에 세그먼트 결과 저장 중: {clip_csv}")
                    logger.info(f"저장할 행 데이터 키: {list(row.keys())}")
                    logger.info(f"저장할 행 데이터 샘플: {dict(list(row.items())[:5])}")
                    try:
                        with open(clip_csv, 'a', newline='', encoding='utf-8') as f:
                            w = csv.DictWriter(f, fieldnames=clip_csv_headers)
                            w.writerow(row)
                        logger.info(f"세그먼트 결과 저장 완료: {info.get('segment_index')} - 정확도: {acc:.2f}%, 정밀도: {prec:.2f}%, 재현율: {rec:.2f}%, F1: {f1:.2f}%")
                    except Exception as csv_error:
                        logger.error(f"CSV 저장 중 오류: {csv_error}")
                        logger.error(f"저장하려던 데이터: {row}")
                        # 파일이 손상되었을 수 있으므로 백업 시도
                        try:
                            backup_csv = f"{clip_csv}.backup"
                            shutil.copy2(clip_csv, backup_csv)
                            logger.info(f"CSV 파일 백업 생성: {backup_csv}")
                        except Exception as backup_error:
                            logger.error(f"백업 생성 실패: {backup_error}")
                else:
                    logger.warning("save_evaluation_results가 False로 설정되어 있어 CSV 저장을 건너뜁니다.")
                continue

            if unit == 'file_done':
                fp = info['file_path']
                
                # 현재 파일의 성능 계산 및 저장
                if current_file_segments:
                    file_perf = calculate_file_performance(current_file_segments)
                    file_perf['file_path'] = fp
                    file_perf['original_filename'] = info.get('original_filename', '')
                    file_perf['file_result'] = info.get('file_result', '양품')
                    file_perf['segment_total'] = info.get('segment_total', 0)
                    file_perf['duration'] = info.get('duration', 0.0)
                    file_performance_data.append(file_perf)
                    
                    # 개별 파일 성능 CSV 저장 (중복 방지)
                    if save_evaluation_results:
                        save_single_file_performance_csv(file_perf, results_dir, today, logger)
                
                # raw_data 파일 이동 처리
                if os.path.exists(fp):
                    move_file_to_destination(fp, info.get('file_result', '양품'),
                                             reserved_dir, logger)
                    delete_original_folder(fp, raw_data_dir, logger)
                else:
                    create_missing_file_log(fp, logger)
                
                continue

        except Exception as e:
            logger.error(f"소리 파일 처리 오류: {e}")
            time.sleep(1)


def calculate_file_performance(segments):
    """파일 단위 성능 지표 계산"""
    if not segments:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    y_true = [seg['true_label'] for seg in segments]
    y_pred = [seg['pred_label'] for seg in segments]
    
    # 파일 단위 최종 판정 (하나라도 불량이면 불량)
    file_true = 1 if any(y_true) else 0
    file_pred = 1 if any(y_pred) else 0
    
    # 세그먼트 단위 성능
    seg_acc = accuracy_score(y_true, y_pred)
    
    # 파일 단위 성능
    file_acc = 1.0 if file_true == file_pred else 0.0
    
    # 세그먼트 단위 precision, recall, f1 계산
    if 1 in y_true and 0 in y_true:  # 정상과 불량이 모두 있는 경우
        seg_prec = precision_score(y_true, y_pred, zero_division=0)
        seg_rec = recall_score(y_true, y_pred, zero_division=0)
        seg_f1 = f1_score(y_true, y_pred, zero_division=0)
    elif 1 in y_true:  # 불량만 있는 경우
        seg_prec = 1.0 if any(y_pred) else 0.0
        seg_rec = 1.0 if any(y_pred) else 0.0
        seg_f1 = 1.0 if any(y_pred) else 0.0
    else:  # 정상만 있는 경우
        seg_prec = 1.0 if not any(y_pred) else 0.0
        seg_rec = 1.0 if not any(y_pred) else 0.0
        seg_f1 = 1.0 if not any(y_pred) else 0.0
    
    # 파일 단위 precision, recall, f1 (파일을 하나의 샘플로 간주)
    if file_true == 1:  # 실제 불량 파일
        file_prec = 1.0 if file_pred == 1 else 0.0
        file_rec = 1.0 if file_pred == 1 else 0.0
        file_f1 = 1.0 if file_pred == 1 else 0.0
    else:  # 실제 정상 파일
        file_prec = 1.0 if file_pred == 0 else 0.0
        file_rec = 1.0 if file_pred == 0 else 0.0
        file_f1 = 1.0 if file_pred == 0 else 0.0
    
    return {
        'file_accuracy': round(file_acc * 100, 2),
        'file_precision': round(file_prec * 100, 2),
        'file_recall': round(file_rec * 100, 2),
        'file_f1_score': round(file_f1 * 100, 2),
        'segment_accuracy': round(seg_acc * 100, 2),
        'segment_precision': round(seg_prec * 100, 2),
        'segment_recall': round(seg_rec * 100, 2),
        'segment_f1_score': round(seg_f1 * 100, 2),
        'true_label': file_true,
        'pred_label': file_pred
    }


def save_single_file_performance_csv(file_perf, results_dir, today, logger):
    """개별 파일 성능 CSV 저장 (중복 방지)"""
    try:
        csv_path = os.path.join(results_dir, f"detection_results_{today}.csv")
        
        headers = [
            'file_path', 'original_filename', 'file_result', 'segment_total', 'duration',
            'file_accuracy', 'file_precision', 'file_recall', 'file_f1_score',
            'segment_accuracy', 'segment_precision', 'segment_recall', 'segment_f1_score',
            'true_label', 'pred_label'
        ]
        
        write_header = not os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                w.writeheader()
            
            w.writerow(file_perf)
        
        logger.info(f"파일 단위 성능 결과 저장됨: {csv_path}")
        
    except Exception as e:
        logger.error(f"파일 성능 CSV 저장 실패: {e}")


def save_file_performance_csv(file_performance_data, results_dir, today, logger):
    """파일 단위 성능 CSV 저장"""
    try:
        csv_path = os.path.join(results_dir, f"detection_results_{today}.csv")
        
        headers = [
            'file_path', 'original_filename', 'file_result', 'segment_total', 'duration',
            'file_accuracy', 'file_precision', 'file_recall', 'file_f1_score',
            'segment_accuracy', 'segment_precision', 'segment_recall', 'segment_f1_score',
            'true_label', 'pred_label'
        ]
        
        write_header = not os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                w.writeheader()
            
            for data in file_performance_data:
                w.writerow(data)
        
        logger.info(f"파일 단위 성능 결과 저장됨: {csv_path}")
        
    except Exception as e:
        logger.error(f"파일 성능 CSV 저장 실패: {e}")


def estimate_true_label(filepath_or_filename):
    """파일명에서 라벨 추정 (노트북 방식)"""
    path_str = str(filepath_or_filename).lower().replace('\\', '/')
    path_parts = path_str.split('/')
    filename = path_parts[-1] if path_parts else str(filepath_or_filename)

    # 폴더 경로에서 abnormal 키워드 확인
    if 'sound_abnormal' in path_str or 'vib_abnormal' in path_str:
        return 1  # 불량품
    
    # 폴더 경로에서 normal 키워드 확인
    if 'sound' in path_str and 'abnormal' not in path_str:
        return 0  # 정상품 (sound 폴더, abnormal이 아닌 경우)
    if 'vib_normal' in path_str:
        return 0  # 정상품

    # 노트북과 동일한 키워드 사용 (파일명 기반)
    defect_keywords = ['불량', 'bad', 'defect', 'fault', 'ng', '무나사']
    normal_keywords = ['정상', 'good', '양품', 'normal', 'ok', 'sound_segment']

    # 파일명에서 키워드 확인 (노트북 방식)
    for keyword in defect_keywords:
        if keyword in filename:
            return 1  # 불량품
    
    for keyword in normal_keywords:
        if keyword in filename:
            return 0  # 정상품

    # 기본값 (노트북에서는 경고 후 0 반환)
    return 0


def calculate_performance_metrics_per_file(true_label, predicted_label):
    """개별 파일 단위 정확도만 계산"""
    try:
        acc = 1.0 if true_label == predicted_label else 0.0
        return {
            'accuracy': round(acc * 100, 2),
            'precision': None,
            'recall': None,
            'f1_score': None
        }
    except Exception as e:
        print(f"[오류] 지표 계산 실패: {e}")
        return dict.fromkeys(['accuracy', 'precision', 'recall', 'f1_score'], None)



def create_evaluation_data(result_info, processing_time, file_path, performance_metrics):
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)

        return {
            'timestamp': result_info['timestamp'],
            'file_path': result_info['file_path'],
            'original_filename': result_info['original_filename'],
            'parent_file_name': result_info.get('parent_file_name', result_info['original_filename']),
            'segment_index': result_info.get('segment_index'),
            'segment_total': result_info.get('segment_total'),
            'result': result_info['result'],
            'dtw_score': float(result_info['dtw_score']),
            'ae_loss': float(result_info['ae_loss']),
            'final_score': float(result_info['final_score']),
            'processing_time_seconds': round(processing_time, 3),
            'file_size_mb': round(size_mb, 2),
            'sample_rate': result_info.get('sample_rate', 16000),
            'duration_seconds': round(result_info.get('duration', 0.0), 2),
            'folder_info': str(result_info.get('folder_info', {})),
            **performance_metrics
        }

    except Exception as e:
        print(f"[오류] 평가 데이터 생성 실패: {e}")
        return {}

def create_clip_evaluation_data(info, processing_time, file_path, performance_metrics):
    """세그먼트 단위 CSV 행 생성 (클립 지표 + 누적 지표 + 정답/예측)"""
    def pct(x):
        try:
            if x is None:
                return 0.0  # None 값을 0.0으로 처리
            return round(float(x) * 100, 2)
        except Exception:
            return 0.0

    def safe_float(x, default=0.0):
        try:
            return float(x)
        except (ValueError, TypeError):
            return default

    def safe_int(x, default=0):
        try:
            return int(x)
        except (ValueError, TypeError):
            return default

    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
    except Exception:
        size_mb = None

    true_label = safe_int(performance_metrics.get('true_label', 0))
    pred_label = safe_int(performance_metrics.get('pred_label', 0))

    return {
        'unit': info.get('unit', 'segment'),
        'timestamp': info.get('timestamp', 0.0),
        'file_path': info.get('file_path', ''),
        'original_filename': info.get('original_filename', ''),
        'parent_file_name': info.get('parent_file_name', info.get('original_filename', '')),
        'folder_info': info.get('folder_info', 'raw_data'),
        'segment_index': safe_int(info.get('segment_index', 0)),
        'segment_total': safe_int(info.get('segment_total', 0)),
        'segment_start': safe_float(info.get('segment_start', 0.0)),
        'segment_end': safe_float(info.get('segment_end', 0.0)),
        'segment_duration': safe_float(info.get('segment_duration', 0.0)),
        'result': info.get('result', '양품'),
        'dtw_score': safe_float(info.get('dtw_score', 0.0)),
        'ae_loss': safe_float(info.get('ae_loss', 0.0)),
        'final_score': safe_float(info.get('final_score', 0.0)),
        'duration': safe_float(info.get('duration', 0.0)),
        'sample_rate': safe_int(info.get('sample_rate', 16000)),
        # 정답/예측
        'true_label': true_label,
        'pred_label': pred_label,
        'is_correct': int(true_label == pred_label),
        # 클립 지표
        'clip_accuracy': pct(performance_metrics.get('clip_accuracy')),
        'clip_precision': pct(performance_metrics.get('clip_precision')),
        'clip_recall': pct(performance_metrics.get('clip_recall')),
        'clip_f1': pct(performance_metrics.get('clip_f1')),
        # 누적 지표
        'cumu_accuracy': pct(performance_metrics.get('cumu_accuracy')),
        'cumu_precision': pct(performance_metrics.get('cumu_precision')),
        'cumu_recall': pct(performance_metrics.get('cumu_recall')),
        'cumu_f1': pct(performance_metrics.get('cumu_f1'))
    }


def move_file_to_destination(file_path, result, reserved_dir, logger):
    """파일을 결과에 따라 적절한 위치로 이동 (소리 데이터용) - 주석처리됨"""
    # 파일 이동 기능 주석처리
    logger.info(f"소리 파일 이동 기능이 비활성화됨: {file_path}")
    return
    
    # try:
    #     if not os.path.exists(file_path):
    #         logger.warning(f"이동할 소리 파일이 없음: {file_path}")
    #         return

    #     filename = os.path.basename(file_path)
    #     base_name, ext = os.path.splitext(filename)
        
    #     if result == "불량품":
    #         # 소리 불량품은 sound_reserved/abnormal로 이동
    #         dest_path = os.path.join(reserved_dir, "abnormal", f"{base_name}_sound_reserved{ext}")
    #     else:
    #         # 소리 양품은 sound_reserved/normal로 이동
    #         dest_path = os.path.join(reserved_dir, "normal", f"{base_name}_sound_split{ext}")

    #     os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    #     shutil.move(file_path, dest_path)
    #     logger.info(f"소리 탐지 완료된 파일 이동됨 → {dest_path}")

    # except Exception as e:
    #     logger.error(f"소리 파일 이동 실패: {e}")


def delete_original_folder(file_path, raw_data_dir, logger):
    """원본 폴더 삭제 - 주석처리됨"""
    # 폴더 삭제 기능 주석처리
    logger.info(f"폴더 삭제 기능이 비활성화됨: {file_path}")
    return
    
    # try:
    #     file_dir = os.path.dirname(file_path)
    #     if file_dir != raw_data_dir and os.path.exists(file_dir):
    #         shutil.rmtree(file_dir)
    #         logger.info(f"원본 폴더 삭제됨: {file_dir}")
    # except Exception as e:
    #     logger.error(f"폴더 삭제 실패: {e}")


def create_missing_file_log(file_path, logger):
    """파일 없음 로그 생성"""
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        missing_log = os.path.join(log_dir, "missing_files.log")
        
        with open(missing_log, 'a', encoding='utf-8') as f:
            f.write(f"{DateTimeUtil.get_current_datetime()} - {file_path}\n")
        
        logger.warning(f"파일 없음 로그 기록됨: {file_path}")
    except Exception as e:
        logger.error(f"파일 없음 로그 생성 실패: {e}")


# ==================== 진동 평가 함수들 ====================

def vib_evaluation_proc_worker(detect_queue, config_info, log_file, log_level, log_format):
    """
    진동 파일 처리 및 성능 평가 프로세스 워커
    - 'segment' : 행 단위 저장 (클립 지표 + 누적 지표 + 예측값)
    - 'file_done': 파일 이동/정리
    """
    logger = Logger("vib_file_processor", log_file, log_level, log_format)
    raw_data_dir = config_info.get_config("data.raw_data_dir")
    reserved_dir = config_info.get_config("data.vib_reserved_dir")
    save_evaluation_results = config_info.get_config("evaluation.save_evaluation_results", True)
    
    logger.info("=== vib_evaluation_proc_worker 시작 ===")
    logger.info(f"진동 탐지 결과 큐 크기: {detect_queue.qsize()}")
    logger.info(f"save_evaluation_results 설정: {save_evaluation_results}")
    logger.info(f"raw_data_dir: {raw_data_dir}")
    logger.info(f"reserved_dir: {reserved_dir}")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 헤더: 클립 지표 + 누적 지표 + 정답/예측 표시
    clip_csv_headers = [
        'unit','timestamp','file_path','original_filename','parent_file_name','folder_info',
        'segment_index','segment_total','segment_start','segment_end','segment_duration',
        'result','dtw_score','ae_loss','final_score','duration','sample_rate',
        # 행별 정답/예측
        'true_label','pred_label','is_correct',
        # 클립(개별) 지표 - 개별 세그먼트 지표
        'clip_accuracy','clip_precision','clip_recall','clip_f1',
        # 누적 지표 - 현재 파일 내에서 계산
        'cumu_accuracy','cumu_precision','cumu_recall','cumu_f1'
    ]

    today = DateTimeUtil.get_current_date()
    clip_csv = os.path.join(results_dir, f"vib_evaluation_clip_results_{today}.csv")
    
    # CSV 파일이 없으면 헤더와 함께 생성, 있으면 기존 파일 유지
    if not os.path.exists(clip_csv):
        try:
            # results 디렉토리가 없으면 생성
            os.makedirs(results_dir, exist_ok=True)
            
            with open(clip_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=clip_csv_headers)
                w.writeheader()
            logger.info(f"새로운 진동 클립 결과 CSV 파일 생성: {clip_csv}")
            logger.info(f"CSV 헤더 작성 완료: {len(clip_csv_headers)}개 컬럼")
            logger.info(f"CSV 헤더: {clip_csv_headers}")
        except Exception as e:
            logger.error(f"CSV 파일 생성 실패: {e}")
            logger.error(f"생성하려던 경로: {clip_csv}")
            logger.error(f"results_dir 존재 여부: {os.path.exists(results_dir)}")
    else:
        logger.info(f"기존 진동 클립 결과 CSV 파일 사용: {clip_csv}")
        logger.info(f"기존 파일 크기: {os.path.getsize(clip_csv)} bytes")

    # 누적 지표(세그먼트 기준) - 전체 세션 동안 유지
    y_true_seg, y_pred_seg = [], []
    
    # 파일 단위 성능 추적
    file_performance_data = []
    current_file_segments = []
    current_file_name = None
    
    # 중복 처리 방지를 위한 세그먼트 추적
    processed_segments = set()
    
    # 성능 지표 초기화 로깅
    logger.info("진동 성능 지표 추적 시작 - 누적 지표가 실시간으로 계산됩니다")

    while True:
        try:
            if detect_queue.empty():
                time.sleep(0.1)
                continue

            info = detect_queue.get()
            logger.info(f"큐에서 진동 데이터 수신: {info.get('unit', 'unknown')}")
            unit = info.get('unit', 'segment')

            # 진행률 메시지 처리 (터미널 로그)
            if unit == 'progress':
                pct = info.get('progress_pct')
                done = info.get('segment_done')
                total = info.get('segment_total')
                fname = os.path.basename(info.get('file_path', ''))
                logger.info(f"[진동 진행률] {pct}% ({done}/{total}) - {fname}")
                continue

            if unit == 'segment':
                start_t = time.time()
                fp = info['file_path']
                segment_index = info.get('segment_index', 0)
                
                # 중복 처리 방지
                segment_key = f"{fp}_{segment_index}"
                if segment_key in processed_segments:
                    logger.info(f"중복 세그먼트 건너뛰기: {segment_key}")
                    continue
                
                processed_segments.add(segment_key)
                logger.info(f"=== 진동 세그먼트 데이터 처리 시작 ===")
                logger.info(f"파일 경로: {fp}")
                logger.info(f"세그먼트 인덱스: {segment_index}")
                logger.info(f"세그먼트 키: {segment_key}")
                
                # 파일 경로 검증 - raw_data에서 처리하므로 원본 경로 확인
                file_exists = os.path.exists(fp)
                if not file_exists:
                    # raw_data 내에서 파일 찾기
                    filename = os.path.basename(fp)
                    raw_data_path = os.path.join(raw_data_dir, filename)
                    
                    if os.path.exists(raw_data_path):
                        fp = raw_data_path
                        file_exists = True
                        logger.info(f"raw_data에서 진동 파일 발견: {raw_data_path}")
                
                if not file_exists:
                    logger.warning(f"진동 파일 없음(세그먼트): {fp}")
                    create_missing_file_log(fp, logger)
                    continue

                # 파일 변경 감지
                if current_file_name != fp:
                    # 이전 파일의 성능 계산 및 저장
                    if current_file_segments:
                        file_perf = calculate_file_performance(current_file_segments)
                        file_performance_data.append(file_perf)
                    
                    current_file_name = fp
                    current_file_segments = []
                    
                    # 새로운 파일 시작 시 누적 리스트 초기화
                    y_true_seg.clear()
                    y_pred_seg.clear()
                    logger.info(f"새 진동 파일 처리 시작: {fp}")

                # 실제 라벨 추출 (파일 경로 기반)
                t = estimate_true_label(fp)
                
                # 탐지 결과에서 예측 라벨 가져오기 (탐지 프로세스에서 계산된 값 사용)
                result = info.get('result', '양품')
                p = 1 if result == "불량품" else 0
                
                # DTW 점수와 AE Loss 로깅 (디버깅용)
                dtw_score = info.get('dtw_score', 0)
                ae_loss = info.get('ae_loss', 0)
                logger.info(f"진동 세그먼트 {segment_index}: DTW={dtw_score:.2f}, AE={ae_loss:.6f}, Result={result}, True={t}, Pred={p}")
                
                y_true_seg.append(t); y_pred_seg.append(p)
                
                # 현재 파일의 세그먼트 정보 저장
                current_file_segments.append({
                    'true_label': t,
                    'pred_label': p,
                    'segment_index': info.get('segment_index', 0),
                    'result': info.get('result')
                })

                # 클립(해당 행) 지표 - 개별 세그먼트 지표 계산
                clip_acc = 1.0 if t == p else 0.0
                # 개별 세그먼트의 precision, recall, f1 계산
                if t == 1:  # 실제 불량품인 경우
                    clip_prec = 1.0 if p == 1 else 0.0
                    clip_rec = 1.0 if p == 1 else 0.0
                    clip_f1 = 1.0 if p == 1 else 0.0
                else:  # 실제 정상품인 경우
                    clip_prec = 1.0 if p == 0 else 0.0
                    clip_rec = 1.0 if p == 0 else 0.0
                    clip_f1 = 1.0 if p == 0 else 0.0

                # 누적 지표 (현재 파일 내에서만 계산)
                if len(y_true_seg) >= 1:
                    acc = accuracy_score(y_true_seg, y_pred_seg)
                    # precision, recall, f1은 불량품(1)이 있을 때만 계산
                    if 1 in y_true_seg and 0 in y_true_seg:  # 정상과 불량이 모두 있을 때
                        prec = precision_score(y_true_seg, y_pred_seg, zero_division=0)
                        rec  = recall_score(y_true_seg, y_pred_seg, zero_division=0)
                        f1   = f1_score(y_true_seg, y_pred_seg, zero_division=0)
                    elif 1 in y_true_seg:  # 불량만 있는 경우
                        prec = 1.0 if any(y_pred_seg) else 0.0
                        rec = 1.0 if any(y_pred_seg) else 0.0
                        f1 = 1.0 if any(y_pred_seg) else 0.0
                    else:  # 정상만 있는 경우
                        prec = 1.0 if not any(y_pred_seg) else 0.0
                        rec = 1.0 if not any(y_pred_seg) else 0.0
                        f1 = 1.0 if not any(y_pred_seg) else 0.0

                processing_time = time.time() - start_t
                perf_payload = {
                    'true_label': t, 'pred_label': p, 'is_correct': int(t == p),
                    'clip_accuracy': clip_acc, 'clip_precision': clip_prec,
                    'clip_recall': clip_rec, 'clip_f1': clip_f1,
                    'cumu_accuracy': acc, 'cumu_precision': prec,
                    'cumu_recall': rec, 'cumu_f1': f1
                }

                row = create_clip_evaluation_data(info, processing_time, fp, perf_payload)
                logger.info(f"진동 평가 데이터 생성 완료: true_label={t}, pred_label={p}, accuracy={acc:.2f}")
                logger.info(f"CSV 행 키 개수: {len(row.keys())}")
                logger.info(f"CSV 행 키: {list(row.keys())}")

                if save_evaluation_results:
                    logger.info(f"CSV 파일에 진동 세그먼트 결과 저장 중: {clip_csv}")
                    logger.info(f"저장할 행 데이터 키: {list(row.keys())}")
                    logger.info(f"저장할 행 데이터 샘플: {dict(list(row.items())[:5])}")
                    try:
                        with open(clip_csv, 'a', newline='', encoding='utf-8') as f:
                            w = csv.DictWriter(f, fieldnames=clip_csv_headers)
                            w.writerow(row)
                        logger.info(f"=== 진동 세그먼트 결과 저장 완료 ===")
                        logger.info(f"세그먼트 인덱스: {info.get('segment_index')}")
                        logger.info(f"정확도: {acc:.2f}%")
                        logger.info(f"정밀도: {prec:.2f}%")
                        logger.info(f"재현율: {rec:.2f}%")
                        logger.info(f"F1: {f1:.2f}%")
                    except Exception as csv_error:
                        logger.error(f"CSV 저장 중 오류: {csv_error}")
                        logger.error(f"저장하려던 데이터: {row}")
                        # 파일이 손상되었을 수 있으므로 백업 시도
                        try:
                            backup_csv = f"{clip_csv}.backup"
                            shutil.copy2(clip_csv, backup_csv)
                            logger.info(f"CSV 파일 백업 생성: {backup_csv}")
                        except Exception as backup_error:
                            logger.error(f"백업 생성 실패: {backup_error}")
                else:
                    logger.warning("save_evaluation_results가 False로 설정되어 있어 CSV 저장을 건너뜁니다.")
                continue

            if unit == 'file_done':
                logger.info(f"=== 진동 파일 완료 처리 시작 ===")
                fp = info['file_path']
                logger.info(f"파일 경로: {fp}")
                logger.info(f"파일 결과: {info.get('file_result', '양품')}")
                logger.info(f"세그먼트 총 수: {info.get('segment_total', 0)}")
                
                # 현재 파일의 성능 계산 및 저장
                if current_file_segments:
                    file_perf = calculate_file_performance(current_file_segments)
                    file_perf['file_path'] = fp
                    file_perf['original_filename'] = info.get('original_filename', '')
                    file_perf['file_result'] = info.get('file_result', '양품')
                    file_perf['segment_total'] = info.get('segment_total', 0)
                    file_perf['duration'] = info.get('duration', 0.0)
                    file_performance_data.append(file_perf)
                    
                    # 개별 파일 성능 CSV 저장 (중복 방지)
                    if save_evaluation_results:
                        save_single_file_performance_csv(file_perf, results_dir, today, logger)
                else:
                    logger.warning(f"진동 파일 완료 처리 중 current_file_segments가 비어있음: {fp}")
                    # 세그먼트 데이터가 없어도 기본 정보만 저장
                    if save_evaluation_results:
                        basic_file_perf = {
                            'file_path': fp,
                            'original_filename': info.get('original_filename', ''),
                            'file_result': info.get('file_result', '양품'),
                            'segment_total': info.get('segment_total', 0),
                            'duration': info.get('duration', 0.0),
                            'file_accuracy': 0.0,
                            'file_precision': 0.0,
                            'file_recall': 0.0,
                            'file_f1_score': 0.0,
                            'segment_accuracy': 0.0,
                            'segment_precision': 0.0,
                            'segment_recall': 0.0,
                            'segment_f1_score': 0.0,
                            'true_label': 0,
                            'pred_label': 0
                        }
                        save_single_file_performance_csv(basic_file_perf, results_dir, today, logger)
                
                # raw_data 파일 이동 처리 (진동 파일용)
                if os.path.exists(fp):
                    move_file_to_destination(fp, info.get('file_result', '양품'),
                                             reserved_dir, logger)
                    delete_original_folder(fp, raw_data_dir, logger)
                else:
                    create_missing_file_log(fp, logger)
                
                continue

        except Exception as e:
            logger.error(f"진동 파일 처리 오류: {e}")
            time.sleep(1)


def save_single_file_performance_csv(file_perf, results_dir, today, logger):
    """개별 파일 성능 CSV 저장 (중복 방지)"""
    try:
        csv_path = os.path.join(results_dir, f"vib_detection_results_{today}.csv")
        
        headers = [
            'file_path', 'original_filename', 'file_result', 'segment_total', 'duration',
            'file_accuracy', 'file_precision', 'file_recall', 'file_f1_score',
            'segment_accuracy', 'segment_precision', 'segment_recall', 'segment_f1_score',
            'true_label', 'pred_label'
        ]
        
        write_header = not os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                w.writeheader()
            
            w.writerow(file_perf)
        
        logger.info(f"진동 파일 단위 성능 결과 저장됨: {csv_path}")
        
    except Exception as e:
        logger.error(f"진동 파일 성능 CSV 저장 실패: {e}")


def save_file_performance_csv(file_performance_data, results_dir, today, logger):
    """파일 단위 성능 CSV 저장"""
    try:
        csv_path = os.path.join(results_dir, f"vib_detection_results_{today}.csv")
        
        headers = [
            'file_path', 'original_filename', 'file_result', 'segment_total', 'duration',
            'file_accuracy', 'file_precision', 'file_recall', 'file_f1_score',
            'segment_accuracy', 'segment_precision', 'segment_recall', 'segment_f1_score',
            'true_label', 'pred_label'
        ]
        
        write_header = not os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                w.writeheader()
            
            for data in file_performance_data:
                w.writerow(data)
        
        logger.info(f"진동 파일 단위 성능 결과 저장됨: {csv_path}")
        
    except Exception as e:
        logger.error(f"진동 파일 성능 CSV 저장 실패: {e}")


def create_evaluation_data(result_info, processing_time, file_path, performance_metrics):
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)

        return {
            'timestamp': result_info['timestamp'],
            'file_path': result_info['file_path'],
            'original_filename': result_info['original_filename'],
            'parent_file_name': result_info.get('parent_file_name', result_info['original_filename']),
            'segment_index': result_info.get('segment_index'),
            'segment_total': result_info.get('segment_total'),
            'result': result_info['result'],
            'dtw_score': float(result_info['dtw_score']),
            'ae_loss': float(result_info['ae_loss']),
            'final_score': float(result_info['final_score']),
            'processing_time_seconds': round(processing_time, 3),
            'file_size_mb': round(size_mb, 2),
            'sample_rate': result_info.get('sample_rate', 1000),  # 진동은 1000Hz
            'duration_seconds': round(result_info.get('duration', 0.0), 2),
            'folder_info': str(result_info.get('folder_info', {})),
            **performance_metrics
        }

    except Exception as e:
        print(f"[오류] 진동 평가 데이터 생성 실패: {e}")
        return {}


def get_true_label_from_path(file_path):
    """파일 경로를 기반으로 실제 라벨을 추출"""
    if 'sound_normal' in file_path or 'vib_normal' in file_path:
        return 0  # 정상
    elif 'sound_abnormal' in file_path or 'vib_abnormal' in file_path:
        return 1  # 비정상
    else:
        # 파일명 패턴으로 판단
        filename = os.path.basename(file_path)
        if 'abnormal' in filename.lower() or 'defect' in filename.lower():
            return 1
        elif 'normal' in filename.lower() or 'good' in filename.lower():
            return 0
        else:
            # 기본값은 정상으로 설정
            return 0

def process_evaluation_results(evaluation_data, config):
    """평가 결과 처리 및 라벨링 개선"""
    processed_results = []
    
    for result in evaluation_data:
        # 파일 경로에서 실제 라벨 추출
        true_label = get_true_label_from_path(result['file_path'])
        
        # 예측 라벨 계산 (임계값 기반)
        dtw_score = result.get('dtw_score', 0)
        ae_loss = result.get('ae_loss', 0)
        
        dtw_threshold = config.get_config("detection.dtw_threshold", 50000.0)
        ae_threshold = config.get_config("detection.ae_threshold", 0.01)
        
        # 예측 라벨 결정
        if dtw_score > dtw_threshold or ae_loss > ae_threshold:
            pred_label = 1  # 비정상
        else:
            pred_label = 0  # 정상
        
        # 정확도 계산
        is_correct = 1 if true_label == pred_label else 0
        
        # 결과 업데이트
        result['true_label'] = true_label
        result['pred_label'] = pred_label
        result['is_correct'] = is_correct
        
        processed_results.append(result)
    
    return processed_results




def move_file_to_destination(file_path, result, reserved_dir, logger):
    """파일을 결과에 따라 적절한 위치로 이동 (진동 데이터용) - 주석처리됨"""
    # 파일 이동 기능 주석처리
    logger.info(f"진동 파일 이동 기능이 비활성화됨: {file_path}")
    return
    
    # try:
    #     if not os.path.exists(file_path):
    #         logger.warning(f"이동할 진동 파일이 없음: {file_path}")
    #         return

    #     filename = os.path.basename(file_path)
    #     base_name, ext = os.path.splitext(filename)
        
    #     if result == "불량품":
    #         # 진동 불량품은 vib_reserved/abnormal로 이동
    #         dest_path = os.path.join(reserved_dir, "abnormal", f"{base_name}_vib_reserved{ext}")
    #     else:
    #         # 진동 양품은 vib_reserved/normal로 이동
    #         dest_path = os.path.join(reserved_dir, "normal", f"{base_name}_vib_split{ext}")

    #     os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    #     shutil.move(file_path, dest_path)
    #     logger.info(f"진동 탐지 완료된 파일 이동됨 → {dest_path}")

    # except Exception as e:
    #     logger.error(f"진동 파일 이동 실패: {e}")


def delete_original_folder(file_path, raw_data_dir, logger):
    """원본 폴더 삭제 - 주석처리됨"""
    # 폴더 삭제 기능 주석처리
    logger.info(f"진동 폴더 삭제 기능이 비활성화됨: {file_path}")
    return
    
    # try:
    #     file_dir = os.path.dirname(file_path)
    #     if file_dir != raw_data_dir and os.path.exists(file_dir):
    #         shutil.rmtree(file_dir)
    #         logger.info(f"원본 진동 폴더 삭제됨: {file_dir}")
    # except Exception as e:
    #     logger.error(f"진동 폴더 삭제 실패: {e}")


def create_missing_file_log(file_path, logger):
    """파일 없음 로그 생성"""
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        missing_log = os.path.join(log_dir, "missing_vib_files.log")
        
        with open(missing_log, 'a', encoding='utf-8') as f:
            f.write(f"{DateTimeUtil.get_current_datetime()} - {file_path}\n")
        
        logger.warning(f"진동 파일 없음 로그 기록됨: {file_path}")
    except Exception as e:
        logger.error(f"진동 파일 없음 로그 생성 실패: {e}")