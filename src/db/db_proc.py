import pandas as pd
import os
from datetime import datetime
from ..util.util_datetime import DateTimeUtil
from ..util.util_logger import Logger
#from .db_if import db_if_mysql_get_conn
from .db_mysql import (
    mysql_insert_row, mysql_insert_many,
    mysql_connection, ensure_bolt_inspection_table, mysql_is_alive
)

def db_proc_worker(shared_queue, evaluation_queue, config_info, log_file, log_level, log_format, mysql_conn):
    """DB 처리 프로세스 워커"""
    logger = Logger("db_processor", log_file, log_level, log_format)
    logger.info("db_proc_worker 시작")
    
    # 결과 저장 디렉토리 생성
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 결과 데이터 저장용 리스트
    # === 1) 자식 프로세스에서 직접 MySQL 연결 확보 ===
    if (mysql_conn is None) or (not mysql_is_alive(mysql_conn)):
        mysql_conn = mysql_connection(
            host=config_info.get_config("db.mysql.host", "localhost"),
            user=config_info.get_config("db.mysql.user", "root"),
            password=config_info.get_config("db.mysql.password", "your_password"),
            database=config_info.get_config("db.mysql.database", "test_db"),
        )
        if mysql_conn:
            logger.info("자식 프로세스에서 MySQL 연결 확보")
        else:
            logger.error("MySQL 연결 실패 → DB 저장 없이 진행")

    # === 2) 테이블 스키마 보증 (여러 번 호출돼도 안전) ===
    if mysql_conn:
        ensure_bolt_inspection_table(mysql_conn)

    # 결과 데이터 저장용 리스트
    segment_results = []
    file_results = []

    while True:
        try:
            # 큐에서 데이터 가져오기
            data = shared_queue.get()

            unit = data.get('unit')
            if unit == 'progress':
                # 진행률 정보는 로그만 출력
                logger.info(f"진행률: {data.get('progress_pct', 0)}% "
                            f"({data.get('segment_done', 0)}/{data.get('segment_total', 0)})")

            elif unit == 'segment':
                # 세그먼트 결과 처리
                segment_results.append(data)
                logger.info(f"세그먼트 결과 저장: {data.get('original_filename', 'unknown')} - "
                            f"{data.get('result', 'unknown')}")

                # (선택) 세그먼트 단위 DB 저장을 원하면 주석 해제
                """
                try:
                    if mysql_conn:
                        row = {
                            "capture_datetime": data.get("timestamp") or DateTimeUtil.get_current_datetime(),
                            "raw_vibration_data": None,
                            "raw_audio_data": None,
                            "raw_audio_path": data.get("file_path"),
                            "dtw_score": float(data.get("dtw_score", 0.0)),
                            "ae_loss": float(data.get("ae_loss", 0.0)),
                            "final_score": float(data.get("final_score", 0.0)),
                            "sample_rate": int(data.get("sample_rate", 16000)),
                            "bolt_status": _normalize_status(data.get("result", "양품")),
                        }
                        mysql_insert_row(mysql_conn, "bolt_inspection", row)
                except Exception as ie:
                    logger.error(f"[segment] MySQL INSERT 실패: {ie}")
                """

            elif unit == 'file_done':
                # 파일 완료 신호 처리
                file_results.append(data)
                logger.info(f"파일 완료 신호 수신: {data.get('original_filename', 'unknown')}")

                # === 3) 파일 단위 요약을 DB에 저장 (권장 위치)
                try:
                    if mysql_conn:
                        row = {
                            "capture_datetime": data.get("timestamp") or DateTimeUtil.get_current_datetime(),
                            "raw_vibration_data": None,   # 원본 BLOB은 운영상 비권장 → 경로만 저장 권장
                            "raw_audio_data": None,
                            "raw_audio_path": data.get("file_path"),
                            "dtw_score": float(data.get("dtw_score", 0.0)),
                            "ae_loss": float(data.get("ae_loss", 0.0)),
                            "final_score": float(data.get("final_score", 0.0)),
                            "sample_rate": int(data.get("sample_rate", 16000)),
                            "bolt_status": _normalize_status(data.get("file_result", data.get("result", "양품"))),
                        }
                        mysql_insert_row(mysql_conn, "bolt_inspection", row)
                except Exception as ie:
                    logger.error(f"[file_done] MySQL INSERT 실패: {ie}")

                # 평가 큐로 전달
                evaluation_queue.put(data)

                # 결과를 CSV 파일로 저장
                _save_results_to_csv(segment_results, file_results, results_dir, logger)
                segment_results = []
                file_results = []

                # 세그먼트 결과 초기화 (파일별로 분리)
                segment_results = []

            #### 예제
            '''
            # mysql DB에 데이터 저장
            conn = db_if_mysql_get_conn()
            # 1) INSERT (단건)
            user_id = mysql_insert_row(conn, "users", {"name": "홍길동", "email": "hong@example.com"})
            print("inserted id:", user_id)

            # 2) INSERT (다건)
            count = mysql_insert_many(conn, "users", [
                {"name": "김철수", "email": "kim@example.com"},
                {"name": "이영희", "email": "lee@example.com"},
            ])
            print("bulk inserted:", count)
            '''
                
        except Exception as e:
            logger.error(f"DB 프로세스 오류: {e}")

def _save_results_to_csv(segment_results, file_results, results_dir, logger):
    """결과를 CSV 파일로 저장 (소리 데이터만)"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d')
        
        # 소리 데이터만 필터링 (진동 데이터는 진동 평가 프로세스에서 별도 저장)
        sound_segment_results = [r for r in segment_results if r.get('folder_info') == 'sound']
        sound_file_results = [r for r in file_results if r.get('folder_info') == 'sound']
        
        # 세그먼트 결과 저장 (소리만)
        if sound_segment_results:
            segment_df = pd.DataFrame(sound_segment_results)
            segment_csv_path = os.path.join(results_dir, f"evaluation_clip_results_{timestamp}.csv")
            
            # 기존 파일이 있으면 추가 모드로 저장
            if os.path.exists(segment_csv_path):
                segment_df.to_csv(segment_csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
                logger.info(f"소리 세그먼트 결과 추가 저장 완료: {segment_csv_path}")
            else:
                segment_df.to_csv(segment_csv_path, index=False, encoding='utf-8-sig')
                logger.info(f"소리 세그먼트 결과 새로 저장 완료: {segment_csv_path}")
        
        # 파일 결과 저장 (소리만)
        if sound_file_results:
            file_df = pd.DataFrame(sound_file_results)
            file_csv_path = os.path.join(results_dir, f"detection_results_{timestamp}.csv")
            
            # 기존 파일이 있으면 추가 모드로 저장
            if os.path.exists(file_csv_path):
                file_df.to_csv(file_csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
                logger.info(f"소리 파일 결과 추가 저장 완료: {file_csv_path}")
            else:
                file_df.to_csv(file_csv_path, index=False, encoding='utf-8-sig')
                logger.info(f"소리 파일 결과 새로 저장 완료: {file_csv_path}")
            
    except Exception as e:
        logger.error(f"CSV 저장 실패: {e}")

def create_missing_files_log(config_info, log_file, log_level, log_format):
    """파일 없음 로그 생성"""
    try:
        logger = Logger("missing_files", log_file, log_level, log_format)
        logger.info("파일 없음 로그 생성 시작")
        
        # 로그 파일 경로
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        missing_files_log = os.path.join(log_dir, "missing_files.log")
        
        # 현재 시간으로 로그 생성
        timestamp = DateTimeUtil.get_current_timestamp()
        with open(missing_files_log, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp} - 파일 없음 로그 생성\n")
        
        logger.info(f"파일 없음 로그 생성 완료: {missing_files_log}")
        
    except Exception as e:
        print(f"파일 없음 로그 생성 실패: {e}")

def _normalize_status(value: str) -> str:
    try:
        v = str(value or "").strip().lower()
    except Exception:
        v = ""
    bad_tokens = {"불량", "불량품", "bad", "defect", "ng", "fault"}
    return "불량품" if v in bad_tokens else "양품"

def _f(x, default=None):
    """
    안전 변환 helper 함수 - 키 존재하나 None인 케이스에서 float(None) 방지 
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

# 사용
#"dtw_score": _f(data.get("dtw_score"), 0.0),
#"ae_loss":   _f(data.get("ae_loss"), 0.0),
#"final_score": _f(data.get("final_score"), 0.0),