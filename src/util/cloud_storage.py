"""
클라우드 스토리지 유틸리티 모듈
Google Cloud Storage를 사용한 파일 모니터링 및 업로드 기능
- CloudStorageUtil: 클라우드 스토리지 연결 및 파일 관리
- CloudFileMonitor: 클라우드 파일 모니터링 및 다운로드
- ReservedFileManager: 처리된 파일을 날짜별로 저장 관리
"""

import os
import time
import tempfile
import io
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from google.cloud import storage
from google.oauth2 import service_account
import json

class CloudStorageUtil:
    """Google Cloud Storage 유틸리티 클래스"""
    
    def __init__(self, credentials_path: str, bucket_name: str, max_files_per_day: int = 30000):
        """
        클라우드 스토리지 유틸리티 초기화
        
        Args:
            credentials_path: 서비스 계정 키 파일 경로
            bucket_name: GCS 버킷 이름
            max_files_per_day: 일일 최대 처리 파일 수 (기본값: 30000)
        """
        self.bucket_name = bucket_name
        self.client = self._initialize_client(credentials_path)
        self.bucket = self.client.bucket(bucket_name)
        self.processed_files = set()
        self.max_files_per_day = max_files_per_day
        self.daily_file_count = {}
    
    def _initialize_client(self, credentials_path: str) -> storage.Client:
        """Google Cloud Storage 클라이언트 초기화"""
        try:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            return storage.Client(credentials=credentials)
        except Exception as e:
            print(f"GCS 클라이언트 초기화 실패: {e}")
            raise
    
    def list_audio_files(self, prefix: str = "audio/", limit: int = None) -> List[Dict]:
        """
        버킷에서 오디오 파일 목록 조회
        
        Args:
            prefix: 파일 경로 접두사 (예: "audio/")
            limit: 조회할 최대 파일 수 (None이면 전체)
            
        Returns:
            오디오 파일 정보 리스트
        """
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            audio_files = []
            
            for blob in blobs:
                if blob.name.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    audio_files.append({
                        'name': blob.name,
                        'size': blob.size,
                        'updated': blob.updated,
                        'generation': blob.generation
                    })
                    
                    # 제한에 도달하면 중단
                    if limit and len(audio_files) >= limit:
                        break
            
            return audio_files
        except Exception as e:
            print(f"오디오 파일 목록 조회 실패: {e}")
            return []
    
    def download_file(self, blob_name: str, local_path: str) -> bool:
        """
        클라우드에서 파일 다운로드
        
        Args:
            blob_name: 클라우드 파일 경로
            local_path: 로컬 저장 경로
            
        Returns:
            다운로드 성공 여부
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            return True
        except Exception as e:
            print(f"파일 다운로드 실패: {blob_name} -> {local_path}, 오류: {e}")
            return False
    
    def get_file_content(self, blob_name: str) -> Optional[bytes]:
        """
        클라우드에서 파일 내용을 메모리로 직접 가져오기
        
        Args:
            blob_name: 클라우드 파일 경로
            
        Returns:
            파일 내용 (bytes) 또는 None
        """
        try:
            blob = self.bucket.blob(blob_name)
            content = blob.download_as_bytes()
            return content
        except Exception as e:
            print(f"파일 내용 가져오기 실패: {blob_name}, 오류: {e}")
            return None
    
    def get_file_as_stream(self, blob_name: str) -> Optional[io.BytesIO]:
        """
        클라우드에서 파일을 스트림으로 가져오기
        
        Args:
            blob_name: 클라우드 파일 경로
            
        Returns:
            파일 스트림 (BytesIO) 또는 None
        """
        try:
            blob = self.bucket.blob(blob_name)
            stream = io.BytesIO()
            blob.download_to_file(stream)
            stream.seek(0)  # 스트림 포인터를 처음으로 이동
            return stream
        except Exception as e:
            print(f"파일 스트림 가져오기 실패: {blob_name}, 오류: {e}")
            return None
    
    def upload_file(self, local_path: str, blob_name: str) -> bool:
        """
        로컬 파일을 클라우드에 업로드
        
        Args:
            local_path: 로컬 파일 경로
            blob_name: 클라우드 저장 경로
            
        Returns:
            업로드 성공 여부
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            return True
        except Exception as e:
            print(f"파일 업로드 실패: {local_path} -> {blob_name}, 오류: {e}")
            return False
    
    def upload_results(self, results_dir: str, results_prefix: str = "results/") -> bool:
        """
        결과 파일들을 클라우드에 업로드
        
        Args:
            results_dir: 로컬 결과 디렉토리
            results_prefix: 클라우드 결과 디렉토리 접두사
            
        Returns:
            업로드 성공 여부
        """
        try:
            if not os.path.exists(results_dir):
                print(f"결과 디렉토리가 존재하지 않습니다: {results_dir}")
                return False
            
            success_count = 0
            total_count = 0
            
            for filename in os.listdir(results_dir):
                if filename.endswith(('.csv', '.json', '.txt')):
                    local_path = os.path.join(results_dir, filename)
                    blob_name = f"{results_prefix}{filename}"
                    
                    if self.upload_file(local_path, blob_name):
                        success_count += 1
                    total_count += 1
            
            print(f"결과 파일 업로드 완료: {success_count}/{total_count}")
            return success_count > 0
        except Exception as e:
            print(f"결과 파일 업로드 실패: {e}")
            return False
    
    def get_new_files(self, prefix: str = "audio/", max_files: int = None) -> List[Dict]:
        """
        새로 추가된 오디오 파일들 조회 (일일 제한 적용)
        
        Args:
            prefix: 파일 경로 접두사
            max_files: 최대 조회 파일 수
            
        Returns:
            새로운 파일 정보 리스트
        """
        try:
            current_files = self.list_audio_files(prefix, max_files)
            new_files = []
            today = datetime.now().strftime('%Y-%m-%d')
            
            # 오늘 처리한 파일 수 확인
            today_count = self.daily_file_count.get(today, 0)
            
            for file_info in current_files:
                file_key = f"{file_info['name']}_{file_info['generation']}"
                
                # 이미 처리된 파일은 건너뛰기
                if file_key in self.processed_files:
                    continue
                
                # 일일 제한 확인
                if today_count >= self.max_files_per_day:
                    print(f"일일 파일 처리 제한에 도달했습니다: {self.max_files_per_day}개")
                    break
                
                new_files.append(file_info)
                self.processed_files.add(file_key)
                today_count += 1
            
            # 오늘 처리한 파일 수 업데이트
            self.daily_file_count[today] = today_count
            
            return new_files
        except Exception as e:
            print(f"새 파일 조회 실패: {e}")
            return []
    
    def get_daily_file_count(self) -> Dict[str, int]:
        """일별 처리된 파일 수 조회"""
        return self.daily_file_count.copy()

class CloudFileMonitor:
    """클라우드 스토리지 파일 모니터링 클래스"""
    
    def __init__(self, cloud_storage_util: CloudStorageUtil, temp_dir: str = None, use_memory_processing: bool = True):
        """
        클라우드 파일 모니터 초기화
        
        Args:
            cloud_storage_util: 클라우드 스토리지 유틸리티
            temp_dir: 임시 파일 저장 디렉토리
            use_memory_processing: 메모리에서 직접 처리할지 여부 (기본값: True)
        """
        self.cloud_storage = cloud_storage_util
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.processed_files = set()
        self.use_memory_processing = use_memory_processing
    
    def check_new_files(self, max_files: int = None) -> List[str]:
        """
        새로운 오디오 파일들 확인 (기존 방식 - 임시 파일 다운로드)
        
        Args:
            max_files: 최대 다운로드 파일 수
            
        Returns:
            새 파일의 로컬 경로 리스트
        """
        try:
            new_files = self.cloud_storage.get_new_files(max_files=max_files)
            local_paths = []
            
            for file_info in new_files:
                # 임시 파일로 다운로드
                temp_filename = os.path.basename(file_info['name'])
                local_path = os.path.join(self.temp_dir, temp_filename)
                
                if self.cloud_storage.download_file(file_info['name'], local_path):
                    local_paths.append(local_path)
                    print(f"파일 다운로드 완료: {file_info['name']} -> {local_path}")
            
            return local_paths
        except Exception as e:
            print(f"새 파일 확인 실패: {e}")
            return []
    
    def get_new_files_info(self, max_files: int = None) -> List[Dict]:
        """
        새로운 오디오 파일 정보 조회 (메모리 처리용)
        
        Args:
            max_files: 최대 조회 파일 수
            
        Returns:
            새 파일 정보 리스트 (다운로드 없음)
        """
        try:
            new_files = self.cloud_storage.get_new_files(max_files=max_files)
            return new_files
        except Exception as e:
            print(f"새 파일 정보 조회 실패: {e}")
            return []
    
    def get_file_stream(self, blob_name: str) -> Optional[io.BytesIO]:
        """
        클라우드 파일을 스트림으로 가져오기 (메모리 처리용)
        
        Args:
            blob_name: 클라우드 파일 경로
            
        Returns:
            파일 스트림 또는 None
        """
        return self.cloud_storage.get_file_as_stream(blob_name)
    
    def get_file_content(self, blob_name: str) -> Optional[bytes]:
        """
        클라우드 파일 내용을 바이트로 가져오기 (메모리 처리용)
        
        Args:
            blob_name: 클라우드 파일 경로
            
        Returns:
            파일 내용 또는 None
        """
        return self.cloud_storage.get_file_content(blob_name)
    
    def cleanup_temp_files(self, file_paths: List[str]):
        """
        임시 파일들 정리
        
        Args:
            file_paths: 정리할 파일 경로 리스트
        """
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"임시 파일 삭제: {file_path}")
            except Exception as e:
                print(f"임시 파일 삭제 실패: {file_path}, 오류: {e}")

class ReservedFileManager:
    """처리된 파일을 날짜별로 reserved 폴더에 저장하는 관리자"""
    
    def __init__(self, base_dir: str = "data/reserved"):
        """
        Reserved 파일 관리자 초기화
        
        Args:
            base_dir: 기본 저장 디렉토리
        """
        self.base_dir = base_dir
        self._ensure_base_directory()
    
    def _ensure_base_directory(self):
        """기본 디렉토리 생성"""
        os.makedirs(self.base_dir, exist_ok=True)
    
    def get_daily_directory(self, date: str = None) -> str:
        """
        날짜별 디렉토리 경로 생성
        
        Args:
            date: 날짜 (YYYY-MM-DD 형식, None이면 오늘)
            
        Returns:
            날짜별 디렉토리 경로
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        daily_dir = os.path.join(self.base_dir, date)
        os.makedirs(daily_dir, exist_ok=True)
        return daily_dir
    
    def move_to_reserved(self, file_path: str, original_name: str = None) -> str:
        """
        파일을 reserved 폴더로 이동
        
        Args:
            file_path: 이동할 파일 경로
            original_name: 원본 파일명 (None이면 현재 파일명 사용)
            
        Returns:
            이동된 파일의 새 경로
        """
        try:
            if not os.path.exists(file_path):
                print(f"파일이 존재하지 않습니다: {file_path}")
                return None
            
            # 날짜별 디렉토리 생성
            daily_dir = self.get_daily_directory()
            
            # 파일명 결정
            if original_name is None:
                original_name = os.path.basename(file_path)
            
            # 중복 방지를 위한 타임스탬프 추가
            timestamp = datetime.now().strftime('%H%M%S')
            filename = f"{timestamp}_{original_name}"
            
            # 새 경로
            new_path = os.path.join(daily_dir, filename)
            
            # 파일 이동
            import shutil
            shutil.move(file_path, new_path)
            
            print(f"파일을 reserved로 이동: {file_path} -> {new_path}")
            return new_path
            
        except Exception as e:
            print(f"파일 이동 실패: {file_path}, 오류: {e}")
            return None
    
    def get_daily_file_count(self, date: str = None) -> int:
        """
        특정 날짜의 파일 수 조회
        
        Args:
            date: 날짜 (YYYY-MM-DD 형식, None이면 오늘)
            
        Returns:
            파일 수
        """
        daily_dir = self.get_daily_directory(date)
        if not os.path.exists(daily_dir):
            return 0
        
        files = [f for f in os.listdir(daily_dir) if os.path.isfile(os.path.join(daily_dir, f))]
        return len(files)
    
    def cleanup_old_directories(self, days_to_keep: int = 30):
        """
        오래된 디렉토리 정리
        
        Args:
            days_to_keep: 보관할 일수
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)
                if os.path.isdir(item_path):
                    try:
                        item_date = datetime.strptime(item, '%Y-%m-%d')
                        if item_date < cutoff_date:
                            import shutil
                            shutil.rmtree(item_path)
                            print(f"오래된 디렉토리 삭제: {item_path}")
                    except ValueError:
                        # 날짜 형식이 아닌 디렉토리는 건너뛰기
                        continue
        except Exception as e:
            print(f"오래된 디렉토리 정리 실패: {e}")
