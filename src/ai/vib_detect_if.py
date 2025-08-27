from multiprocessing import Process
from .vib_detect_proc import detect_proc_worker

def detect_if_run(vib_shared_queue, detect_queue, db_queue, config_info, log_file, log_level, log_format):
    """
    진동 AI 탐지 프로세스 인터페이스
    """
    try:
        print('vib_detect_if : 진동 AI 탐지 프로세스 준비')
        process = Process(
            target=detect_proc_worker,
            args=(vib_shared_queue, detect_queue, db_queue, config_info, log_file, log_level, log_format)
        )
        print("진동 AI 탐지 프로세스 시작")
        return process
            
    except Exception as e:
        print(f"진동 AI 탐지 프로세스 시작 실패 : {e}")
        raise
