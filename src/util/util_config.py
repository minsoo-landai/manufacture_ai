"""
# 설정 관리 유틸리티 모듈
설정 관리 모듈 : JSON 기반 설정 파일을 관리하는 유틸리티
- 역할 : 설정 파일 로드/저장, 설정값 조회/수정, 기본 설정 생성 등 전체 시스템 설정 관리
- JSON 기반 : config.json 파일을 통해 모든 설정값을 중앙 집중식으로 관리
- 계층적 구조 : system, data, ai, preprocessing, detection, process, db, logging 등 카테고리별 설정
"""

import json
import os
from pathlib import Path

class Config:
    """설정 관리 클래스 : JSON 설정 파일을 관리하는 메인 클래스"""
    
    def __init__(self, config_path="config.json"):
        self.config_path = config_path                    # 설정 파일 경로
        self.config = self._load_config()                 # 설정 데이터 로드
    
    def _load_config(self):
        """설정 파일을 로드합니다."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 기본 설정 생성 (설정 파일이 없을 때 자동 생성)
            default_config = {
                "system": {
                    "mode": "detect"  # detect 또는 full (전체 파이프라인 또는 탐지만)
                },
                "data": {
                    "raw_dir": "data/raw_data",           # 원본 오디오 데이터 디렉토리
                    "segments_dir": "data/segments",      # 분할된 오디오 세그먼트 디렉토리
                    "reference_dir": "data/reference",    # 참조 데이터 디렉토리
                    "detect_dir": "data/detect",          # 탐지 대상 데이터 디렉토리
                    "reserved_dir": "data/reserved",      # 예약된 데이터 디렉토리
                    "models_dir": "models"                # 모델 저장 디렉토리
                },
                "ai": {
                    "model_path": "models/sound_detector.pkl", # 모델 파일 경로
                    "threshold": 0.5,                     # 탐지 임계값 (0.5 = 50%)
                    "window_size": 3.0,                   # 분석 윈도우 크기 (3초)
                    "sample_rate": 16000,                 # 샘플링 레이트 (Hz)
                    "n_mels": 128,                        # 멜 스펙트로그램 빈도 수
                    "hop_length": 512,                    # 스펙트로그램 계산 시 프레임 간격
                    "n_mfcc": 13,                         # MFCC 계수 개수
                    "contamination": 0.1,                 # 이상치 비율 (10%)
                    "n_estimators": 100,                  # 랜덤 포레스트 트리 개수
                    "enable_auto_sample_rate": True,      # 자동 샘플링 레이트 감지 활성화
                    "candidate_sample_rates": [8000, 16000, 22050, 44100, 48000] # 지원 샘플링 레이트 목록
                },
                "preprocessing": {
                    "enable_normalization": True,         # 오디오 정규화 활성화
                    "enable_noise_reduction": False,      # 노이즈 제거 활성화
                    "enable_spectral_subtraction": False, # 스펙트럼 빼기 활성화
                    "min_audio_length": 2.0,              # 최소 오디오 길이 (초)
                    "max_audio_length": 10.0,             # 최대 오디오 길이 (초)
                    "target_sr": 16000,                   # 목표 샘플링 레이트
                    "resample_method": "kaiser_best"      # 리샘플링 방법
                },
                "detection": {
                    "ae_weight": 0.5,                     # AutoEncoder 가중치
                    "dtw_weight": 0.5,                    # DTW 가중치
                    "margin": 1.05,                       # 임계값 마진
                    "auto_sr": True,                      # 자동 샘플링 레이트 감지
                    "clip_duration": 3.0,                 # 클립 길이 (초)
                    "enable_file_completion_check": True, # 파일 완성 체크 활성화
                    "file_completion_timeout": 3.0,       # 파일 완성 대기 시간 (초)
                    "enable_next_file_check": True,       # 다음 파일 체크 활성화
                    "move_to_reserved": False,            # 처리 후 예약 디렉토리로 이동 (비활성화)
                    "audio_duration_tolerance": 0.5,      # 오디오 길이 허용 오차 (초)
                    "next_file_check_retries": 3,         # 다음 파일 체크 재시도 횟수
                    "next_file_check_interval": 1.0,      # 다음 파일 체크 간격 (초)
                    "continuous_check_max_wait": 10.0,    # 연속 체크 최대 대기 시간 (초)
                    "file_stability_checks": 5,           # 파일 안정성 체크 횟수
                    "epochs": 100,                        # 학습 에포크 수
                    "batch_size": 16,                     # 배치 크기
                    "learning_rate": 0.001,               # 학습률
                    "validation_split": 0.2,              # 검증 데이터 비율 (20%)
                    "ae_threshold_percentile": 90,        # AutoEncoder 임계값 백분위수
                    "min_dtw_threshold": 1.0,             # 최소 DTW 임계값
                    "early_stopping_patience": 10,        # 조기 종료 인내심
                    "n_mfcc": 13                          # MFCC 계수 개수
                },
                "process": {
                    "max_workers": 4,                     # 최대 워커 프로세스 수
                    "queue_size": 100,                    # 큐 크기
                    "timeout": 30,                        # 타임아웃 (초)
                    "retry_count": 3                      # 재시도 횟수
                },
                "db": {
                    "project_id": "your-project-id",      # 프로젝트 ID
                    "dataset_id": "sound_manufacture",    # 데이터셋 ID
                    "table_id": "detection_results"       # 테이블 ID
                },
                "logging": {
                    "level": "INFO",                      # 로그 레벨
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", # 로그 형식
                    "file_path": "logs/app.log"           # 로그 파일 경로
                }
            }
            self._save_config(default_config)
            return default_config
    
    def _save_config(self, config):
        """설정을 파일에 저장합니다."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)  # json 저장 / indent=2 : 들여쓰기 2칸 / ensure_ascii=False : 한글 깨짐 방지
    
    def get_config(self, key, default=None):
        """설정값을 가져옵니다."""
        keys = key.split('.')       # . 기준으로 분리 / 예) "ai.model_path" -> ["ai", "model_path"]
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set_config(self, key, value):
        """설정값을 설정합니다."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:                 
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self._save_config(self.config)
    
    def update_config(self, updates):
        """여러 설정값을 한번에 업데이트합니다."""
        for key, value in updates.items():
            self.set(key, value)

# 전역 설정 인스턴스
#config = Config() 