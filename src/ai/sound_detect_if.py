from multiprocessing import Process
from .sound_detect_proc import detect_proc_worker

def detect_if_run(shared_queue, detect_queue, db_queue, config_info, log_file, log_level, log_format):
    """
    AI 탐지 프로세스 인터페이스
    """
    try:
        print('detect_if : AI 탐지 프로세스 준비')
        process = Process(
            target=detect_proc_worker,
            args=(shared_queue, detect_queue, db_queue, config_info, log_file, log_level, log_format)
        )
        print("AI 탐지 프로세스 시작")
        return process
            
    except Exception as e:
        print(f"AI 탐지 프로세스 시작 실패 : {e}")
        raise
