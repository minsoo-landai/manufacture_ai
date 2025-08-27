import queue
import time
from typing import Any
from multiprocessing import Queue


class ProcessQueue:
    """
    프로세스 간 통신을 위한 큐 클래스 : 멀티프로세스 환경에서 안전한 큐 사용
    - put : 큐에 아이템 추가
    - get : 큐에서 아이템 가져오기
    - size : 큐의 크기 반환
    - empty : 큐가 비어있는지 확인
    - full : 큐가 가득 찼는지 확인
    """
    
    def __init__(self, name: str, maxsize: int = None):
        self.name = name                                                  # 큐 이름
        #self.maxsize = maxsize or config.get('process.queue_size', 100)  # 큐 크기
        self.maxsize = maxsize  # 큐 크기
        self.queue = Queue(maxsize=self.maxsize)                         # 멀티프로세스 큐 생성
        #self.logger = Logger(f"process_queue.{name}")                    # 로깅 인스턴스
    
    # 아이템 추가 (아이템, 타임아웃)
    def put(self, item: Any, timeout: float = None):
        """아이템을 큐에 추가합니다."""
        try:
            if timeout:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        self.queue.put_nowait(item)                      # 타임아웃 내에 큐에 추가 시도
                        return True
                    except queue.Full:
                        time.sleep(0.1)                                  # 0.1초 대기 후 재시도
                return False
            else:
                self.queue.put(item)
                return True
        except Exception as e:
            #self.logger.error(f"큐에 아이템 추가 실패: {e}")
            return False
    
    def get(self, timeout: float = None):
        """큐에서 아이템을 가져옵니다."""
        try:
            if timeout:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        return self.queue.get_nowait()
                    except queue.Empty:
                        time.sleep(0.1)
                return None
            else:
                return self.queue.get()
        except Exception as e:
            #self.logger.error(f"큐에서 아이템 가져오기 실패: {e}")
            return None
    
    def size(self) -> int:
        """큐의 크기를 반환합니다."""
        try:
            return self.queue.qsize()
        except:
            return 0
    
    def empty(self) -> bool:
        """큐가 비어있는지 확인합니다."""
        try:
            return self.queue.empty()
        except:
            return True
    
    def full(self) -> bool:
        """큐가 가득 찼는지 확인합니다."""
        try:
            return self.queue.full()
        except:
            return False
