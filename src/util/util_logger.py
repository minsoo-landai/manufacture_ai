import logging
import os
from datetime import datetime

class Logger:
    """로깅 유틸리티 클래스"""
    
    def __init__(self, name: str, log_file: str = None, level: str = None, format: str = None):
        self.name = name                                    # 로거 이름
        self.logger = logging.getLogger(name)               # Python 내장 로거 인스턴스
        
        # 이미 핸들러가 설정되어 있으면 추가하지 않음 (중복 핸들러 방지)
        if self.logger.handlers:
            return
        
        # 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_level = level
        self.logger.setLevel(getattr(logging, level))
        
        # 포맷터 설정 (로그 메시지 형식 정의)
        log_format = format 
        formatter = logging.Formatter(log_format)
        
        # 콘솔 핸들러 (터미널에 로그 출력)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 파일 핸들러 (파일에 로그 저장)
        if log_file is None:
            log_file = log_file
        
        # 로그 디렉토리 생성 (로그 파일 저장 디렉토리가 없으면 생성)
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message):
        """정보 로그"""
        self.logger.info(message)
    
    def warning(self, message):
        """경고 로그"""
        self.logger.warning(message)
    
    def error(self, message):
        """에러 로그"""
        self.logger.error(message)
    
    def debug(self, message):
        """디버그 로그"""
        self.logger.debug(message)
