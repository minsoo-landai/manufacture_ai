"""
# 파일 처리 유틸리티 모듈
파일 처리 유틸리티 모듈 : 파일 시스템 작업을 관리하는 유틸리티
- 역할 : 파일 이동/복사/삭제, 디렉토리 관리, 오디오 파일 검증, 시각화 저장 등 파일 시스템 작업 관리
- 오디오 특화 : 오디오 파일 검증, 스펙트로그램/파형 이미지 생성, 파일 완성 체크 등 오디오 처리 특화 기능
- 폴더 모니터링 : 실시간으로 폴더 변화를 감지하여 새로운 파일 처리
"""

import os
import shutil
import glob
import time
import io
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path
from typing import List, Tuple, Optional
#from .util_config import config
#from .util_logger import Logger

class FileUtil:
    """파일 처리 유틸리티 클래스 : 파일 시스템 작업을 담당하는 메인 클래스"""
    
    def __init__(self):
        #self.logger = Logger("file_util")                   # 로깅 인스턴스
        self.logger = 0
    
    def ensure_directory(self, directory: str) -> bool:
        """디렉토리가 존재하는지 확인하고 없으면 생성합니다."""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)  # 부모 디렉토리까지 생성
            return True
        except Exception as e:
            #self.logger.error(f"디렉토리 생성 실패: {directory} - {e}")
            return False
    
    def get_files_in_directory(self, directory: str, pattern: str = "*") -> List[str]:
        """디렉토리에서 파일 목록을 가져옵니다."""
        try:
            if not os.path.exists(directory):
                self.logger.warning(f"디렉토리가 존재하지 않습니다: {directory}")
                return []
            
            files = glob.glob(os.path.join(directory, pattern))  # 패턴에 맞는 파일들 검색
            return [f for f in files if os.path.isfile(f)]       # 파일만 필터링 (디렉토리 제외)
        except Exception as e:
            #self.logger.error(f"파일 목록 가져오기 실패: {directory} - {e}")
            return []
    
    def move_file(self, src: str, dst: str, overwrite: bool = False) -> bool:
        """파일을 이동합니다."""
        try:
            if not os.path.exists(src):
                #self.logger.error(f"소스 파일이 존재하지 않습니다: {src}")
                return False
            
            # 대상 디렉토리 생성 (이동할 디렉토리가 없으면 생성)
            dst_dir = os.path.dirname(dst)
            if not self.ensure_directory(dst_dir):
                return False
            
            # 파일이 이미 존재하는 경우 (덮어쓰기 옵션 확인)
            if os.path.exists(dst):
                if overwrite:
                    os.remove(dst)  # 기존 파일 삭제
                else:
                    #self.logger.warning(f"대상 파일이 이미 존재합니다: {dst}")
                    return False
            
            shutil.move(src, dst)  # 파일 이동
            #self.logger.info(f"파일 이동 완료: {src} -> {dst}")
            return True
        except Exception as e:
            #self.logger.error(f"파일 이동 실패: {src} -> {dst} - {e}")
            return False
    
    def copy_file(self, src: str, dst: str, overwrite: bool = False) -> bool:
        """파일을 복사합니다."""
        try:
            if not os.path.exists(src):
                #self.logger.error(f"소스 파일이 존재하지 않습니다: {src}")
                return False
            
            # 대상 디렉토리 생성 (복사할 디렉토리가 없으면 생성)
            dst_dir = os.path.dirname(dst)
            if not self.ensure_directory(dst_dir):
                return False
            
            # 파일이 이미 존재하는 경우 (덮어쓰기 옵션 확인)
            if os.path.exists(dst):
                if overwrite:
                    os.remove(dst)  # 기존 파일 삭제
                else:
                    #self.logger.warning(f"대상 파일이 이미 존재합니다: {dst}")
                    return False
            
            shutil.copy2(src, dst)  # 파일 복사 (메타데이터 포함)
            #self.logger.info(f"파일 복사 완료: {src} -> {dst}")
            return True
        except Exception as e:
            #self.logger.error(f"파일 복사 실패: {src} -> {dst} - {e}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """파일을 삭제합니다."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                #self.logger.info(f"파일 삭제 완료: {file_path}")
                return True
            else:
                #self.logger.warning(f"삭제할 파일이 존재하지 않습니다: {file_path}")
                return False
        except Exception as e:
            #self.logger.error(f"파일 삭제 실패: {file_path} - {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Optional[dict]:
        """파일 정보를 가져옵니다."""
        try:
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            return {
                'name': os.path.basename(file_path),
                'path': file_path,
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'extension': os.path.splitext(file_path)[1]
            }
        except Exception as e:
            #self.logger.error(f"파일 정보 가져오기 실패: {file_path} - {e}")
            return None
    
    def is_audio_file(self, file_path: str) -> bool:
        """오디오 파일인지 확인합니다."""
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
        return os.path.splitext(file_path)[1].lower() in audio_extensions
    
    def is_csv_file(self, file_path: str) -> bool:
        """CSV 파일인지 확인합니다."""
        return os.path.splitext(file_path)[1].lower() == '.csv'
    
    def get_audio_files(self, directory: str) -> List[str]:
        """디렉토리에서 오디오 파일만 가져옵니다."""
        all_files = self.get_files_in_directory(directory)
        return [f for f in all_files if self.is_audio_file(f)]
    
    def check_audio_duration(self, file_path: str, target_duration: float = None, 
                           tolerance: float = None) -> bool:
        """오디오 파일의 길이가 목표 길이와 일치하는지 확인합니다."""
        try:
            if not self.is_audio_file(file_path):
                self.logger.warning(f"오디오 파일이 아닙니다: {file_path}")
                return False
            
            # 설정값 가져오기
            if target_duration is None:
                target_duration = config.get('detection.clip_duration', 3.0)
            if tolerance is None:
                tolerance = config.get('detection.audio_duration_tolerance', 0.5)
            
            # 오디오 길이 확인
            duration = librosa.get_duration(path=file_path)
            min_duration = target_duration - tolerance
            max_duration = target_duration + tolerance
            
            is_valid = min_duration <= duration <= max_duration
            
            if is_valid:
                print(f"오디오 길이 확인 통과: {file_path} ({duration:.2f}초)")
            else:
                print(f"오디오 길이 불일치: {file_path} ({duration:.2f}초, 목표: {target_duration}초)")
            
            return is_valid
            
        except Exception as e:
           # self.logger.error(f"오디오 길이 확인 실패: {file_path} - {e}")
            return False
    
    def wait_for_file_completion(self, file_path: str, timeout: float = None, 
                               check_interval: float = 0.1) -> bool:
        """파일 업로드 완료를 기다립니다."""
        # 설정값 가져오기
        if timeout is None:
            timeout = config.get('detection.file_completion_timeout', 30.0)
        required_stable_checks = config.get('detection.file_stability_checks', 5)
        
        start_time = time.time()
        last_size = -1
        stable_count = 0
        
        #self.logger.info(f"파일 완료 대기 시작: {file_path}")
        
        while time.time() - start_time < timeout:
            if not os.path.exists(file_path):
                time.sleep(check_interval)
                continue
            
            current_size = os.path.getsize(file_path)
            
            # 파일 크기가 변하지 않으면 안정성 카운트 증가
            if current_size == last_size and current_size > 0:
                stable_count += 1
                if stable_count >= required_stable_checks:
                    self.logger.info(f"파일 업로드 완료 확인: {file_path} (크기: {current_size} bytes)")
                    return True
            else:
                stable_count = 0
            
            last_size = current_size
            time.sleep(check_interval)
        
        #self.logger.error(f"파일 업로드 타임아웃: {file_path}")
        return False
    
    def check_next_file_exists(self, current_file: str, directory: str) -> bool:
        """다음 파일이 존재하는지 확인합니다."""
        try:
            current_name = os.path.basename(current_file)
            current_num = self._extract_file_number(current_name)
            
            if current_num is None:
                return False
            
            next_num = current_num + 1
            next_file_pattern = self._get_file_pattern(current_name)
            
            # 다음 파일이 있는지 확인
            for file_path in self.get_files_in_directory(directory):
                file_name = os.path.basename(file_path)
                file_num = self._extract_file_number(file_name)
                
                if file_num == next_num:
                    self.logger.info(f"다음 파일 발견: {file_name}")
                    return True
            
            #self.logger.warning(f"다음 파일이 없습니다: 현재={current_name}, 다음 예상 번호={next_num}")
            return False
        except Exception as e:
            #self.logger.error(f"다음 파일 확인 실패: {e}")
            return False
    
    def process_audio_file_with_validation(self, file_path: str, target_duration: float = 3.0,
                                         timeout: float = 30.0) -> bool:
        """오디오 파일을 검증하고 처리합니다."""
        try:
            #self.logger.info(f"오디오 파일 검증 시작: {file_path}")
            
            # 1. 파일 완료 대기
            if not self.wait_for_file_completion(file_path, timeout):
                #self.logger.error(f"파일 완료 대기 실패: {file_path}")
                return False
            
            # 2. 오디오 길이 확인
            if not self.check_audio_duration(file_path, target_duration):
                #self.logger.error(f"오디오 길이 검증 실패: {file_path}")
                return False
            
           # self.logger.info(f"오디오 파일 검증 완료: {file_path}")
            return True
            
        except Exception as e:
           # self.logger.error(f"오디오 파일 검증 중 오류: {file_path} - {e}")
            return False
    
    def _extract_file_number(self, filename: str) -> Optional[int]:
        """파일명에서 숫자를 추출합니다."""
        import re
        
        # 파일명에서 숫자 부분 추출
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return None
    
    def _get_file_pattern(self, filename: str) -> str:
        """파일 패턴을 추출합니다."""
        import re
        
        # 숫자 부분을 *로 대체
        pattern = re.sub(r'\d+', '*', filename)
        return pattern
    
    def save_spectrogram(self, audio: np.ndarray, sr: int, file_path: str, 
                        save_path: str = None, dpi: int = 100, format: str = 'png') -> bool:
        """스펙트로그램을 이미지로 저장합니다."""
        try:
            # 스펙트로그램 생성
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 이미지 생성
            plt.figure(figsize=(10, 4), dpi=dpi)
            librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram - {os.path.basename(file_path)}')
            plt.tight_layout()
            
            # 저장 경로 설정
            if save_path is None:
                images_dir = config.get('data.images_dir', 'data/images')
                spectrogram_dir = os.path.join(images_dir, 'spectrograms')
                self.ensure_directory(spectrogram_dir)
                
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                save_path = os.path.join(spectrogram_dir, f"{base_name}_spectrogram.{format}")
            
            # 이미지 저장
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
          #  self.logger.info(f"스펙트로그램 저장 완료: {save_path}")
            return True
            
        except Exception as e:
          #  self.logger.error(f"스펙트로그램 저장 실패: {file_path} - {e}")
            return False
    
    def save_waveform(self, audio: np.ndarray, sr: int, file_path: str, 
                     save_path: str = None, dpi: int = 100, format: str = 'png') -> bool:
        """웨이브폼을 이미지로 저장합니다."""
        try:
            # 시간축 생성
            time = np.linspace(0, len(audio) / sr, len(audio))
            
            # 이미지 생성
            plt.figure(figsize=(12, 4), dpi=dpi)
            plt.plot(time, audio, linewidth=0.5, alpha=0.7)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'Waveform - {os.path.basename(file_path)}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 저장 경로 설정
            if save_path is None:
                images_dir = config.get('data.images_dir', 'data/images')
                waveform_dir = os.path.join(images_dir, 'waveforms')
                self.ensure_directory(waveform_dir)
                
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                save_path = os.path.join(waveform_dir, f"{base_name}_waveform.{format}")
            
            # 이미지 저장
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
           # self.logger.info(f"웨이브폼 저장 완료: {save_path}")
            return True
            
        except Exception as e:
          #  self.logger.error(f"웨이브폼 저장 실패: {file_path} - {e}")
            return False
    
    def save_audio_visualizations(self, audio: np.ndarray, sr: int, file_path: str) -> bool:
        """오디오 파일의 시각화를 저장합니다 (스펙트로그램 + 웨이브폼)."""
        try:
            success = True
            
            # 스펙트로그램 저장
            if config.get('preprocessing.save_spectrograms', True):
                success &= self.save_spectrogram(
                    audio, sr, file_path,
                    dpi=config.get('preprocessing.spectrogram_dpi', 100),
                    format=config.get('preprocessing.image_format', 'png')
                )
            
            # 웨이브폼 저장
            if config.get('preprocessing.save_waveforms', True):
                success &= self.save_waveform(
                    audio, sr, file_path,
                    dpi=config.get('preprocessing.waveform_dpi', 100),
                    format=config.get('preprocessing.image_format', 'png')
                )
            
            return success
            
        except Exception as e:
           # self.logger.error(f"오디오 시각화 저장 실패: {file_path} - {e}")
            return False

class FolderMonitor:
    """폴더 모니터링 클래스"""
    
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.file_util = FileUtil()
      #  self.logger = Logger("folder_monitor")
        self.monitoring = False
        self.known_files = set()
        self.previous_files = set()
        # 초기 파일 목록 설정
        self._initialize_files()
    
    def start_monitoring(self, callback=None):
        """폴더 모니터링을 시작합니다."""
        self.monitoring = True
        self.known_files = set(self.file_util.get_files_in_directory(self.folder_path))
       # self.logger.info(f"폴더 모니터링 시작: {self.folder_path}")
        
        if callback:
            callback(self.known_files)
    
    def _initialize_files(self):
        """초기 파일 목록을 설정합니다."""
        current_files = set()
        if os.path.exists(self.folder_path):
            for root, _, files in os.walk(self.folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        mtime = os.path.getmtime(file_path)
                        file_info = (file_path, mtime)
                        current_files.add(file_info)
                    except Exception as e:
                        print(f"[ERROR] 초기 파일 접근 실패: {file_path} | {e}")
                        continue
        self.previous_files = current_files
    
    def check_new_files(self) -> List[str]:
        """새로운 파일들을 확인합니다."""
        current_files = set()
        new_files = []

        if os.path.exists(self.folder_path):
            for root, _, files in os.walk(self.folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        mtime = os.path.getmtime(file_path)
                        file_info = (file_path, mtime)
                        current_files.add(file_info)

                        if file_info not in self.previous_files:
                            new_files.append(file_path)

                    except Exception as e:
                        print(f"[ERROR] 파일 접근 실패: {file_path} | {e}")
                        continue

        self.previous_files = current_files
        return new_files
    
    def stop_monitoring(self):
        """폴더 모니터링을 중지합니다."""
        self.monitoring = False
       # self.logger.info(f"폴더 모니터링 중지: {self.folder_path}")

# 전역 파일 유틸리티 인스턴스
file_util = FileUtil() 

def read_audio_file(file_path, sample_rate):
    """오디오 파일 읽기 (파일 경로 또는 메모리 스트림)"""       # 기존 형태로 raw_data에 받아서 처리할 수 있게. 클라우드 상에서 받아올 수 있게 
    try:
        # 메모리 스트림인 경우
        if hasattr(file_path, 'read'):
            # BytesIO 객체를 바이트로 변환
            file_path.seek(0)  # 스트림 포인터를 처음으로 이동
            audio_bytes = file_path.read()
            # 바이트 데이터를 임시 파일로 저장
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            # 임시 파일에서 오디오 로드
            audio_data, sr = librosa.load(temp_file_path, sr=sample_rate)
            
            # 임시 파일 삭제
            import os
            os.unlink(temp_file_path)
            
            return audio_data
        # 파일 경로인 경우
        else:
            audio_data, sr = librosa.load(file_path, sr=sample_rate)
            return audio_data
    except Exception as e:
        print(f"오디오 파일 읽기 실패: {e}")
        return None

def read_audio_from_bytes(audio_bytes, sample_rate):
    """바이트 데이터에서 오디오 읽기"""
    try:
        import io
        audio_stream = io.BytesIO(audio_bytes)
        audio_data, sr = librosa.load(audio_stream, sr=sample_rate)
        return audio_data
    except Exception as e:
        print(f"바이트에서 오디오 읽기 실패: {e}")
        return None