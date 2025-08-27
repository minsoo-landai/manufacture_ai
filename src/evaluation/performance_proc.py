import time
import os
import csv
import psutil
from ..util.util_logger import Logger
from ..util.util_datetime import DateTimeUtil

def performance_proc_worker(performance_queue, config_info, log_file, log_level, log_format):
    """
    성능 체크 프로세스 워커
    시스템 성능을 모니터링하고 CSV 파일로 저장
    """
    logger = Logger("performance_monitor", log_file, log_level, log_format)
    logger.info("performance_proc_worker 시작")

    # 성능 데이터 저장 디렉토리
    performance_dir = "performance"
    if not os.path.exists(performance_dir):
        os.makedirs(performance_dir, exist_ok=True)

    # CSV 파일 경로
    csv_file_path = os.path.join(performance_dir, f"performance_{DateTimeUtil.get_current_date()}.csv")

    # CSV 파일 헤더 초기화
    csv_headers = [
        'timestamp', 'cpu_percent', 'memory_percent', 'memory_used_gb', 
        'disk_usage_percent', 'network_io_sent_mb', 'network_io_recv_mb',
        'process_count', 'thread_count'
    ]

    # CSV 파일이 없으면 헤더와 함께 생성
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
        logger.info(f"성능 모니터링 CSV 파일 생성: {csv_file_path}")

    # 성능 체크 간격 (초)
    check_interval = config_info.get_config("performance.check_interval", 30)

    logger.info(f"성능 모니터링 시작 (체크 간격: {check_interval}초)")

    while True:
        try:
            # 시스템 성능 데이터 수집
            performance_data = collect_performance_data()
            
            # CSV 파일에 성능 데이터 저장
            save_performance_to_csv(csv_file_path, performance_data, logger)
            
            # 큐에 성능 데이터 전송 (다른 프로세스에서 사용할 경우)
            if not performance_queue.empty():
                performance_queue.put(performance_data)
            
            # 로그에 주요 성능 지표 기록
            logger.info(f"CPU: {performance_data['cpu_percent']:.1f}%, "
                       f"Memory: {performance_data['memory_percent']:.1f}%, "
                       f"Disk: {performance_data['disk_usage_percent']:.1f}%")

            time.sleep(check_interval)

        except Exception as e:
            logger.error(f"성능 체크 프로세스 오류: {e}")
            time.sleep(check_interval)

def collect_performance_data():
    """시스템 성능 데이터를 수집합니다."""
    try:
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)  # GB 단위
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # 네트워크 I/O
        network = psutil.net_io_counters()
        network_io_sent_mb = network.bytes_sent / (1024**2)  # MB 단위
        network_io_recv_mb = network.bytes_recv / (1024**2)  # MB 단위
        
        # 프로세스 및 스레드 수
        process_count = len(psutil.pids())
        thread_count = psutil.cpu_count()
        
        return {
            'timestamp': DateTimeUtil.get_current_timestamp(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_gb': round(memory_used_gb, 2),
            'disk_usage_percent': disk_usage_percent,
            'network_io_sent_mb': round(network_io_sent_mb, 2),
            'network_io_recv_mb': round(network_io_recv_mb, 2),
            'process_count': process_count,
            'thread_count': thread_count
        }
        
    except Exception as e:
        # 오류 발생 시 기본값 반환
        return {
            'timestamp': DateTimeUtil.get_current_timestamp(),
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_used_gb': 0.0,
            'disk_usage_percent': 0.0,
            'network_io_sent_mb': 0.0,
            'network_io_recv_mb': 0.0,
            'process_count': 0,
            'thread_count': 0
        }

def save_performance_to_csv(csv_file_path, performance_data, logger):
    """성능 데이터를 CSV 파일에 저장합니다."""
    try:
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=performance_data.keys())
            writer.writerow(performance_data)
            
    except Exception as e:
        logger.error(f"성능 데이터 CSV 저장 실패: {e}")
        raise
