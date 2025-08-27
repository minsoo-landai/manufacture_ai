
from multiprocessing import Process
from .file_check_proc import file_check_proc_worker as fc_proc

def file_check_if_run(shared_queue, vib_shared_queue, path, log_file, log_level, log_format):
    try:
        #logger = Logger("file_check")  
        print('file_check_if : 소리/진동 파일 체크 프로세스 준비')
        process = Process(
            target=fc_proc,
            args=(shared_queue, vib_shared_queue, path, log_file, log_level, log_format)
            #args=(logger, shared_queue)
        )
        #process.start()
        print("소리/진동 파일 체크 프로세스 시작")
        return process
            
    except Exception as e:
        print(f"소리/진동 파일 체크 프로세스 시작 실패 : {e}")
        raise

    