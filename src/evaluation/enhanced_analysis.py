import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

class EnhancedAnalyzer:
    """향상된 결과 분석 도구"""
    
    def __init__(self):
        self.results_dir = "results"
        self.data_dir = "data/raw_data"
        
    def analyze_data_distribution(self):
        """데이터 분포 분석"""
        print("=== 데이터 분포 분석 ===")
        
        # 각 폴더별 파일 수 확인
        folders = ['sound_normal', 'sound_abnormal', 'vib_normal', 'vib_abnormal']
        
        for folder in folders:
            folder_path = os.path.join(self.data_dir, folder)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.csv'))]
                print(f"{folder}: {len(files)}개 파일")
            else:
                print(f"{folder}: 폴더가 존재하지 않음")
    
    def analyze_processing_coverage(self):
        """처리 범위 분석"""
        print("\n=== 처리 범위 분석 ===")
        
        # 탐지 결과에서 처리된 파일 경로 분석
        detection_file = os.path.join(self.results_dir, "detection_results_2025-08-26.csv")
        if os.path.exists(detection_file):
            df = pd.read_csv(detection_file)
            
            # 폴더별 처리된 파일 수
            print("처리된 파일 분포:")
            if 'folder_info' in df.columns:
                folder_counts = df['folder_info'].value_counts()
                print(folder_counts)
            
            # 파일 경로 패턴 분석
            if 'file_path' in df.columns:
                paths = df['file_path'].tolist()
                normal_count = sum(1 for path in paths if 'sound_normal' in path)
                abnormal_count = sum(1 for path in paths if 'sound_abnormal' in path)
                print(f"\n소리 파일 처리:")
                print(f"  정상: {normal_count}개")
                print(f"  비정상: {abnormal_count}개")
    
    def analyze_evaluation_issues(self):
        """평가 결과 문제점 분석"""
        print("\n=== 평가 결과 문제점 분석 ===")
        
        evaluation_file = os.path.join(self.results_dir, "evaluation_clip_results_2025-08-26.csv")
        if os.path.exists(evaluation_file):
            df = pd.read_csv(evaluation_file)
            
            # 라벨 분포 확인
            if 'true_label' in df.columns:
                label_counts = df['true_label'].value_counts()
                print(f"실제 라벨 분포: {label_counts.to_dict()}")
            
            if 'pred_label' in df.columns:
                pred_counts = df['pred_label'].value_counts()
                print(f"예측 라벨 분포: {pred_counts.to_dict()}")
            
            # 파일 경로 분석
            if 'file_path' in df.columns:
                paths = df['file_path'].unique()
                normal_paths = [p for p in paths if 'sound_normal' in p]
                abnormal_paths = [p for p in paths if 'sound_abnormal' in p]
                
                print(f"\n평가된 파일 경로:")
                print(f"  정상 파일: {len(normal_paths)}개")
                print(f"  비정상 파일: {len(abnormal_paths)}개")
                
                if len(abnormal_paths) == 0:
                    print("⚠️  비정상 파일이 처리되지 않았습니다!")
    
    def suggest_improvements(self):
        """개선 제안"""
        print("\n=== 개선 제안 ===")
        
        # 1. 데이터 처리 범위 확대
        print("1. 데이터 처리 범위 확대:")
        print("   - sound_abnormal 폴더의 파일들이 처리되지 않음")
        print("   - 파일 체크 로직에서 비정상 데이터도 포함하도록 수정 필요")
        
        # 2. 라벨링 시스템 개선
        print("\n2. 라벨링 시스템 개선:")
        print("   - 파일 경로 기반 자동 라벨링 구현")
        print("   - 폴더명을 기반으로 정상/비정상 구분")
        
        # 3. 임계값 조정
        print("\n3. 임계값 조정:")
        print("   - 현재 모든 파일이 정상으로 판정됨")
        print("   - DTW 및 AE Loss 임계값을 더 엄격하게 조정 필요")
        
        # 4. 진동 데이터 처리
        print("\n4. 진동 데이터 처리:")
        print("   - vib_abnormal 폴더의 데이터 처리 확인 필요")
        print("   - 진동 데이터 전처리 로직 점검")
    
    def create_data_processing_script(self):
        """데이터 처리 스크립트 생성"""
        script_content = '''# 데이터 처리 개선 스크립트
import os
import shutil
from pathlib import Path

def ensure_all_data_processed():
    """모든 데이터가 처리되도록 보장"""
    
    # 데이터 디렉토리 확인
    data_dirs = [
        "data/raw_data/sound_normal",
        "data/raw_data/sound_abnormal", 
        "data/raw_data/vib_normal",
        "data/raw_data/vib_abnormal"
    ]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith(('.wav', '.csv'))]
            print(f"{data_dir}: {len(files)}개 파일")
        else:
            print(f"⚠️  {data_dir} 폴더가 존재하지 않습니다!")

def fix_labeling_system():
    """라벨링 시스템 수정"""
    print("라벨링 시스템 개선 필요:")
    print("1. 파일 경로 기반 자동 라벨링")
    print("2. 폴더명을 통한 정상/비정상 구분")
    print("3. 라벨 검증 로직 추가")

if __name__ == "__main__":
    ensure_all_data_processed()
    fix_labeling_system()
'''
        
        with open("fix_data_processing.py", "w", encoding="utf-8") as f:
            f.write(script_content)
        
        print("\n📝 데이터 처리 개선 스크립트가 생성되었습니다: fix_data_processing.py")
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("🔍 향상된 결과 분석 시작\n")
        
        self.analyze_data_distribution()
        self.analyze_processing_coverage()
        self.analyze_evaluation_issues()
        self.suggest_improvements()
        self.create_data_processing_script()
        
        print("\n✅ 분석 완료!")

def main():
    """메인 실행 함수"""
    analyzer = EnhancedAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
