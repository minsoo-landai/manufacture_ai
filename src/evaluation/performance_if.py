from multiprocessing import Process
from .performance_proc import performance_proc_worker

def performance_if_run(performance_queue, config_info, log_file, log_level, log_format):
    """성능 체크 프로세스 인터페이스"""
    try:
        print('performance_if : 성능 체크 프로세스 준비')
        process = Process(
            target=performance_proc_worker,
            args=(performance_queue, config_info, log_file, log_level, log_format)
        )
        print("성능 체크 프로세스 시작")
        return process
    except Exception as e:
        print(f"성능 체크 프로세스 시작 실패 : {e}")
        raise
