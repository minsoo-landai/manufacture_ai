from multiprocessing import Process
from .evaluation_proc import evaluation_proc_worker, vib_evaluation_proc_worker

def evaluation_if_run(detect_queue, performance_queue, config_info, log_file, log_level, log_format):
    try:
        print('evaluation_if : 소리 평가 프로세스 준비')
        process = Process(
            target=evaluation_proc_worker,
            args=(detect_queue, config_info, log_file, log_level, log_format)
        )
        print("소리 평가 프로세스 시작")
        return process
            
    except Exception as e:
        print(f"소리 평가 프로세스 시작 실패 : {e}")
        raise

def vib_evaluation_if_run(detect_queue, performance_queue, config_info, log_file, log_level, log_format):
    try:
        print('evaluation_if : 진동 평가 프로세스 준비')
        process = Process(
            target=vib_evaluation_proc_worker,
            args=(detect_queue, config_info, log_file, log_level, log_format)
        )
        print("진동 평가 프로세스 시작")
        return process
            
    except Exception as e:
        print(f"진동 평가 프로세스 시작 실패 : {e}")
        raise
