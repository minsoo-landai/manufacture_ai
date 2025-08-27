from datetime import datetime, timedelta

class DateTimeUtil:
    """날짜시간 유틸리티 클래스"""
    
    @staticmethod
    def get_current_time():
        """현재 시간을 문자열로 반환"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def get_current_date():
        """현재 날짜를 문자열로 반환"""
        return datetime.now().strftime("%Y-%m-%d")
    
    @staticmethod
    def get_timestamp():
        """현재 타임스탬프 반환"""
        return datetime.now().timestamp()
    
    @staticmethod
    def get_current_timestamp():
        """현재 타임스탬프 반환 (get_timestamp와 동일)"""
        return datetime.now().timestamp()
    
    @staticmethod
    def format_datetime(dt, format_str="%Y-%m-%d %H:%M:%S"):
        """날짜시간을 지정된 형식으로 변환"""
        return dt.strftime(format_str)
