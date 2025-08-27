import time
import os
from ..util.util_file import file_util, FolderMonitor
from ..util.util_datetime import DateTimeUtil
from ..util.util_logger import Logger

def file_check_proc_worker(shared_queue, vib_shared_queue, path, log_file, log_level, log_format):
    """파일 체크 프로세스 워커 함수 - raw_data에서 직접 파일 모니터링"""
    print("file_check_proc_worker 시작")     
    
    # 새로운 로거 (file_check)
    logger = Logger("file_checker",
                    log_file,
                    log_level,
                    log_format)
    logger.info("file_check_proc_worker 시작")
    logger.info(f"모니터링 경로: {path}")
    
    # 소리와 진동 디렉토리 모니터링
    logger.info("소리와 진동 디렉토리 모니터링 모드로 시작")
    sound_monitor = FolderMonitor(os.path.join(path, "sound_normal"))
    sound_abnormal_monitor = FolderMonitor(os.path.join(path, "sound_abnormal"))
    vib_monitor = FolderMonitor(os.path.join(path, "vib_normal"))
    vib_abnormal_monitor = FolderMonitor(os.path.join(path, "vib_abnormal"))
    processed_files = set()
    
    # 기존 파일들을 먼저 처리
    _process_existing_files(shared_queue, vib_shared_queue, path, processed_files, logger)
    
    _run_local_monitoring(shared_queue, vib_shared_queue, sound_monitor, sound_abnormal_monitor, vib_monitor, vib_abnormal_monitor, processed_files, logger)

def _process_existing_files(shared_queue, vib_shared_queue, path, processed_files, logger):
    """기존 파일들을 처리하는 함수"""
    logger.info("기존 파일들 처리 시작")
    
    # 소리 정상 파일들 처리
    sound_normal_path = os.path.join(path, "sound_normal")
    if os.path.exists(sound_normal_path):
        sound_files = file_util.get_files_in_directory(sound_normal_path, "*.wav")
        logger.info(f"기존 소리 정상 파일 수: {len(sound_files)}")
        for file_path in sound_files:
            if file_path not in processed_files:
                try:
                    file_info = {
                        'file_path': file_path,
                        'folder_info': 'sound',
                        'data_type': 'sound',
                        'timestamp': DateTimeUtil.get_current_timestamp()
                    }
                    shared_queue.put(file_info)
                    processed_files.add(file_path)
                    logger.info(f"기존 소리 파일 큐 추가: {file_path}")
                except Exception as e:
                    logger.error(f"기존 소리 파일 큐 등록 중 오류: {file_path} | {e}")
    
    # 소리 불량 파일들 처리
    sound_abnormal_path = os.path.join(path, "sound_abnormal")
    if os.path.exists(sound_abnormal_path):
        sound_abnormal_files = file_util.get_files_in_directory(sound_abnormal_path, "*.wav")
        logger.info(f"기존 소리 불량 파일 수: {len(sound_abnormal_files)}")
        for file_path in sound_abnormal_files:
            if file_path not in processed_files:
                try:
                    file_info = {
                        'file_path': file_path,
                        'folder_info': 'sound_abnormal',
                        'data_type': 'sound',
                        'timestamp': DateTimeUtil.get_current_timestamp()
                    }
                    shared_queue.put(file_info)
                    processed_files.add(file_path)
                    logger.info(f"기존 불량 소리 파일 큐 추가: {file_path}")
                except Exception as e:
                    logger.error(f"기존 불량 소리 파일 큐 등록 중 오류: {file_path} | {e}")
    
    # 진동 정상 파일들 처리
    vib_normal_path = os.path.join(path, "vib_normal")
    if os.path.exists(vib_normal_path):
        vib_files = file_util.get_files_in_directory(vib_normal_path, "*.csv")
        logger.info(f"기존 진동 정상 파일 수: {len(vib_files)}")
        for file_path in vib_files:
            if file_path not in processed_files:
                try:
                    file_info = {
                        'file_path': file_path,
                        'folder_info': 'vibration',
                        'data_type': 'vibration',
                        'timestamp': DateTimeUtil.get_current_timestamp()
                    }
                    vib_shared_queue.put(file_info)
                    processed_files.add(file_path)
                    logger.info(f"기존 진동 파일 큐 추가: {file_path}")
                except Exception as e:
                    logger.error(f"기존 진동 파일 큐 등록 중 오류: {file_path} | {e}")
    
    # 진동 불량 파일들 처리
    vib_abnormal_path = os.path.join(path, "vib_abnormal")
    if os.path.exists(vib_abnormal_path):
        vib_abnormal_files = file_util.get_files_in_directory(vib_abnormal_path, "*.csv")
        logger.info(f"기존 진동 불량 파일 수: {len(vib_abnormal_files)}")
        for file_path in vib_abnormal_files:
            if file_path not in processed_files:
                try:
                    file_info = {
                        'file_path': file_path,
                        'folder_info': 'vib_abnormal',
                        'data_type': 'vibration',
                        'timestamp': DateTimeUtil.get_current_timestamp()
                    }
                    vib_shared_queue.put(file_info)
                    processed_files.add(file_path)
                    logger.info(f"기존 불량 진동 파일 큐 추가: {file_path}")
                except Exception as e:
                    logger.error(f"기존 불량 진동 파일 큐 등록 중 오류: {file_path} | {e}")
    
    logger.info("기존 파일들 처리 완료")

def _run_local_monitoring(shared_queue, vib_shared_queue, sound_monitor, sound_abnormal_monitor, vib_monitor, vib_abnormal_monitor, processed_files, logger):
    """로컬 파일 시스템 모니터링 실행 - 소리와 진동 파일을 구분하여 처리"""
    while True:
        try:
            # 소리 파일 확인 (정상 데이터)
            sound_files = sound_monitor.check_new_files()
            logger.info(f"신규 감지된 소리 파일 수: {len(sound_files)}")
            
            for file_path in sound_files:
                if not file_util.is_audio_file(file_path):
                    logger.debug(f"오디오 파일이 아님, 건너뛰기: {file_path}")
                    continue

                if file_path in processed_files:
                    logger.debug(f"이미 처리된 파일, 건너뛰기: {file_path}")
                    continue

                try:
                    # 소리 파일 정보를 소리 탐지 큐에 전송
                    file_info = {
                        'file_path': file_path,
                        'folder_info': 'sound',
                        'data_type': 'sound',
                        'timestamp': DateTimeUtil.get_current_timestamp()
                    }
                    shared_queue.put(file_info)
                    processed_files.add(file_path)
                    logger.info(f"소리 파일 큐 추가: {file_path}, 현재 소리 큐 크기: {shared_queue.qsize()}")
                except Exception as inner_e:
                    logger.error(f"소리 파일 큐 등록 중 오류: {file_path} | {inner_e}")
            
            # 소리 파일 확인 (불량 데이터)
            sound_abnormal_files = sound_abnormal_monitor.check_new_files()
            logger.info(f"신규 감지된 불량 소리 파일 수: {len(sound_abnormal_files)}")
            
            for file_path in sound_abnormal_files:
                if not file_util.is_audio_file(file_path):
                    logger.debug(f"오디오 파일이 아님, 건너뛰기: {file_path}")
                    continue

                if file_path in processed_files:
                    logger.debug(f"이미 처리된 파일, 건너뛰기: {file_path}")
                    continue

                try:
                    # 불량 소리 파일 정보를 소리 탐지 큐에 전송
                    file_info = {
                        'file_path': file_path,
                        'folder_info': 'sound_abnormal',
                        'data_type': 'sound',
                        'timestamp': DateTimeUtil.get_current_timestamp()
                    }
                    shared_queue.put(file_info)
                    processed_files.add(file_path)
                    logger.info(f"불량 소리 파일 큐 추가: {file_path}, 현재 소리 큐 크기: {shared_queue.qsize()}")
                except Exception as inner_e:
                    logger.error(f"불량 소리 파일 큐 등록 중 오류: {file_path} | {inner_e}")
            
            # 진동 파일 확인 (정상 데이터)
            vib_files = vib_monitor.check_new_files()
            logger.info(f"신규 감지된 진동 파일 수: {len(vib_files)}")
            
            for file_path in vib_files:
                if not file_util.is_csv_file(file_path):
                    logger.debug(f"CSV 파일이 아님, 건너뛰기: {file_path}")
                    continue

                if file_path in processed_files:
                    logger.debug(f"이미 처리된 파일, 건너뛰기: {file_path}")
                    continue

                try:
                    # 진동 파일 정보를 진동 탐지 큐에 전송
                    file_info = {
                        'file_path': file_path,
                        'folder_info': 'vibration',
                        'data_type': 'vibration',
                        'timestamp': DateTimeUtil.get_current_timestamp()
                    }
                    vib_shared_queue.put(file_info)
                    processed_files.add(file_path)
                    logger.info(f"진동 파일 큐 추가: {file_path}, 현재 진동 큐 크기: {vib_shared_queue.qsize()}")
                except Exception as inner_e:
                    logger.error(f"진동 파일 큐 등록 중 오류: {file_path} | {inner_e}")
            
            # 진동 파일 확인 (불량 데이터)
            vib_abnormal_files = vib_abnormal_monitor.check_new_files()
            logger.info(f"신규 감지된 불량 진동 파일 수: {len(vib_abnormal_files)}")
            
            for file_path in vib_abnormal_files:
                if not file_util.is_csv_file(file_path):
                    logger.debug(f"CSV 파일이 아님, 건너뛰기: {file_path}")
                    continue

                if file_path in processed_files:
                    logger.debug(f"이미 처리된 파일, 건너뛰기: {file_path}")
                    continue

                try:
                    # 불량 진동 파일 정보를 진동 탐지 큐에 전송
                    file_info = {
                        'file_path': file_path,
                        'folder_info': 'vib_abnormal',
                        'data_type': 'vibration',
                        'timestamp': DateTimeUtil.get_current_timestamp()
                    }
                    vib_shared_queue.put(file_info)
                    processed_files.add(file_path)
                    logger.info(f"불량 진동 파일 큐 추가: {file_path}, 현재 진동 큐 크기: {vib_shared_queue.qsize()}")
                except Exception as inner_e:
                    logger.error(f"불량 진동 파일 큐 등록 중 오류: {file_path} | {inner_e}")
            
            time.sleep(1)  # 1초마다 체크
            
        except Exception as e:
            logger.error(f"파일 체크 중 예외: {e}")
            time.sleep(5)