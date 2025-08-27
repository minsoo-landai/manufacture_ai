import os
import sys
import time
import warnings
import argparse

from multiprocessing import Process, Queue
from src.util.util_config import Config
from src.util.util_logger import Logger
from src.preprocess.file_check_if import file_check_if_run
from src.ai.sound_detect_if import detect_if_run
from src.ai.vib_detect_if import detect_if_run as vib_detect_if_run
from src.db.db_if import db_if_run, db_if_init
from src.evaluation.evaluation_if import evaluation_if_run, vib_evaluation_if_run
from src.evaluation.performance_if import performance_if_run
from src.ai.sound_train_if import sound_train_if_run    # 학습시 진행

# TensorFlow/Protobuf 경고 숨기기
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
# TensorFlow oneDNN 정보 메시지 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    """메인 함수"""
    try:
        # 명령행 인수 파싱
        parser = argparse.ArgumentParser(description='AI 사운드/진동 탐지 시스템')
        # parser.add_argument('--test-vibration', action='store_true', help='진동 파이프라인 테스트 실행')
        args = parser.parse_args()
        
        # 진동 파이프라인 테스트 모드 (제거됨)
        # if args.test_vibration:
        #     test_vibration_pipeline()
        #     return
        
        # 설정 로드
        config = Config("config.json")
        
        # 로깅 설정
        log_level = config.get_config("logging.level", "INFO")
        log_format = config.get_config("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_file = config.get_config("logging.file_path", "logs/app.log")
        
        # 로거 초기화
        logger = Logger("main", log_file, log_level, format=log_format)
        logger.info("AI 사운드/진동 탐지 시스템 시작")
        
        # 탐지 디렉토리 확인 (raw_data에서 처리)
        detect_dir = config.get_config("data.detect_dir")
        logger.info(f"탐지 디렉토리: {detect_dir}")
        logger.info("raw_data 디렉토리 모니터링 모드로 시작")
        
        # 큐 생성
        shared_queue = Queue()          # 소리 파일 → 소리 탐지
        vib_shared_queue = Queue()      # 진동 파일 → 진동 탐지
        detect_queue = Queue()          # 소리 탐지 → 소리 평가
        vib_detect_queue = Queue()      # 진동 탐지 → 진동 평가
        db_queue = Queue()              # 탐지 → DB
        evaluation_queue = Queue()      # DB → 평가
        vib_evaluation_queue = Queue()  # 진동 평가 → 성능 모니터링
        performance_queue = Queue()     # 성능 모니터링

        # db 로그인 초기화 (미사용시 주석)
        '''
        ok = db_if_init()    # mysql 로그인 / 연결
        if not ok:
            logger.error("MySQL 로그인 실패")
        else :
            logger.info("MySQL 초기화 완료")
        '''
        
        # 프로세스 시작
        processes = []
        
        # 파일 체크 프로세스 (소리와 진동 큐를 모두 전달)
        file_check_process = file_check_if_run(shared_queue, vib_shared_queue, detect_dir, log_file, log_level, log_format)
        if file_check_process:
            file_check_process.start()
            processes.append(file_check_process)
            logger.info("파일 체크 프로세스 시작됨")
        
        # 소리 AI 탐지 프로세스
        detect_process = detect_if_run(shared_queue, detect_queue, db_queue, config, log_file, log_level, log_format)
        if detect_process:
            detect_process.start()
            processes.append(detect_process)
            logger.info("소리 AI 탐지 프로세스 시작됨")
        
        # 진동 AI 탐지 프로세스
        vib_detect_process = vib_detect_if_run(vib_shared_queue, vib_detect_queue, db_queue, config, log_file, log_level, log_format)
        if vib_detect_process:
            vib_detect_process.start()
            processes.append(vib_detect_process)
            logger.info("진동 AI 탐지 프로세스 시작됨")
        
        # DB 처리 프로세스
        db_process = db_if_run(db_queue, evaluation_queue, config, log_file, log_level, log_format)
        if db_process:
            db_process.start()
            processes.append(db_process)
            logger.info("DB 처리 프로세스 시작됨")
        
        # 소리 평가 프로세스
        evaluation_process = evaluation_if_run(detect_queue, performance_queue, config, log_file, log_level, log_format)
        if evaluation_process:
            evaluation_process.start()
            processes.append(evaluation_process)
            logger.info("소리 평가 프로세스 시작됨")
        
        # 진동 평가 프로세스
        vib_evaluation_process = vib_evaluation_if_run(vib_detect_queue, performance_queue, config, log_file, log_level, log_format)
        if vib_evaluation_process:
            vib_evaluation_process.start()
            processes.append(vib_evaluation_process)
            logger.info("진동 평가 프로세스 시작됨")
        
        # 성능 모니터링 프로세스
        performance_process = performance_if_run(performance_queue, config, log_file, log_level, log_format)
        if performance_process:
            performance_process.start()
            processes.append(performance_process)
            logger.info("성능 모니터링 프로세스 시작됨")
        
        logger.info(f"총 {len(processes)}개 프로세스가 시작되었습니다")
        
        # 메인 루프
        try:
            while True:
                time.sleep(1)
                # 프로세스 상태 확인
                for i, process in enumerate(processes):
                    if not process.is_alive():
                        logger.error(f"프로세스 {i}가 종료되었습니다")
                        return
        except KeyboardInterrupt:
            logger.info("사용자에 의해 프로그램이 중단되었습니다")
        finally:
            # 프로세스 정리
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            logger.info("모든 프로세스가 종료되었습니다")
    
    except Exception as e:
        print(f"메인 함수 오류: {e}")
        sys.exit(1)

# 진동 파이프라인 테스트 함수 (제거됨)
# def test_vibration_pipeline():
#     """진동 파이프라인 테스트 함수"""
#     pass

if __name__ == "__main__":
    main()