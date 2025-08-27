import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

class ResultAnalyzer:
    """결과 분석 도구"""
    
    def __init__(self, config):
        self.config = config
        
    def analyze_detection_results(self, csv_path):
        """탐지 결과 분석"""
        try:
            df = pd.read_csv(csv_path)
            
            print("=== 탐지 결과 분석 ===")
            print(f"총 파일 수: {len(df)}")
            print(f"정상 판정: {len(df[df['file_result'] == '양품'])}")
            print(f"불량 판정: {len(df[df['file_result'] == '불량'])}")
            
            # 파일 타입별 분석
            print("\n=== 파일 타입별 분석 ===")
            print(df['folder_info'].value_counts())
            
            return df
            
        except Exception as e:
            print(f"탐지 결과 분석 실패: {e}")
            return None
    
    def analyze_evaluation_results(self, csv_path):
        """평가 결과 분석"""
        try:
            df = pd.read_csv(csv_path)
            
            print("=== 평가 결과 분석 ===")
            print(f"총 세그먼트 수: {len(df)}")
            
            # 라벨 분포
            if 'true_label' in df.columns:
                print(f"실제 정상: {len(df[df['true_label'] == 0])}")
                print(f"실제 불량: {len(df[df['true_label'] == 1])}")
            
            if 'pred_label' in df.columns:
                print(f"예측 정상: {len(df[df['pred_label'] == 0])}")
                print(f"예측 불량: {len(df[df['pred_label'] == 1])}")
            
            # 성능 지표
            if 'is_correct' in df.columns:
                accuracy = df['is_correct'].mean()
                print(f"정확도: {accuracy:.4f}")
            
            # 점수 분포 분석
            if 'dtw_score' in df.columns:
                print(f"\nDTW 점수 통계:")
                print(f"  평균: {df['dtw_score'].mean():.2f}")
                print(f"  표준편차: {df['dtw_score'].std():.2f}")
                print(f"  최소값: {df['dtw_score'].min():.2f}")
                print(f"  최대값: {df['dtw_score'].max():.2f}")
            
            if 'ae_loss' in df.columns:
                print(f"\nAE Loss 통계:")
                print(f"  평균: {df['ae_loss'].mean():.6f}")
                print(f"  표준편차: {df['ae_loss'].std():.6f}")
                print(f"  최소값: {df['ae_loss'].min():.6f}")
                print(f"  최대값: {df['ae_loss'].max():.6f}")
            
            return df
            
        except Exception as e:
            print(f"평가 결과 분석 실패: {e}")
            return None
    
    def plot_score_distributions(self, df, save_path=None):
        """점수 분포 시각화"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # DTW 점수 분포
            if 'dtw_score' in df.columns:
                axes[0, 0].hist(df['dtw_score'], bins=50, alpha=0.7)
                axes[0, 0].set_title('DTW Score Distribution')
                axes[0, 0].set_xlabel('DTW Score')
                axes[0, 0].set_ylabel('Frequency')
                
                # 정상/불량별 DTW 점수
                if 'true_label' in df.columns:
                    normal_dtw = df[df['true_label'] == 0]['dtw_score']
                    abnormal_dtw = df[df['true_label'] == 1]['dtw_score']
                    
                    axes[0, 1].hist(normal_dtw, bins=30, alpha=0.7, label='Normal', color='blue')
                    axes[0, 1].hist(abnormal_dtw, bins=30, alpha=0.7, label='Abnormal', color='red')
                    axes[0, 1].set_title('DTW Score by Label')
                    axes[0, 1].set_xlabel('DTW Score')
                    axes[0, 1].set_ylabel('Frequency')
                    axes[0, 1].legend()
            
            # AE Loss 분포
            if 'ae_loss' in df.columns:
                axes[1, 0].hist(df['ae_loss'], bins=50, alpha=0.7)
                axes[1, 0].set_title('AE Loss Distribution')
                axes[1, 0].set_xlabel('AE Loss')
                axes[1, 0].set_ylabel('Frequency')
                
                # 정상/불량별 AE Loss
                if 'true_label' in df.columns:
                    normal_ae = df[df['true_label'] == 0]['ae_loss']
                    abnormal_ae = df[df['true_label'] == 1]['ae_loss']
                    
                    axes[1, 1].hist(normal_ae, bins=30, alpha=0.7, label='Normal', color='blue')
                    axes[1, 1].hist(abnormal_ae, bins=30, alpha=0.7, label='Abnormal', color='red')
                    axes[1, 1].set_title('AE Loss by Label')
                    axes[1, 1].set_xlabel('AE Loss')
                    axes[1, 1].set_ylabel('Frequency')
                    axes[1, 1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"분포도 저장됨: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"분포도 생성 실패: {e}")
    
    def suggest_thresholds(self, df):
        """임계값 제안"""
        try:
            print("=== 임계값 제안 ===")
            
            if 'dtw_score' in df.columns and 'true_label' in df.columns:
                normal_dtw = df[df['true_label'] == 0]['dtw_score']
                abnormal_dtw = df[df['true_label'] == 1]['dtw_score']
                
                if len(normal_dtw) > 0 and len(abnormal_dtw) > 0:
                    # 95% 백분위수 기반 제안
                    dtw_threshold_95 = normal_dtw.quantile(0.95)
                    dtw_threshold_90 = normal_dtw.quantile(0.90)
                    
                    print(f"DTW 임계값 제안:")
                    print(f"  90% 백분위수: {dtw_threshold_90:.2f}")
                    print(f"  95% 백분위수: {dtw_threshold_95:.2f}")
            
            if 'ae_loss' in df.columns and 'true_label' in df.columns:
                normal_ae = df[df['true_label'] == 0]['ae_loss']
                abnormal_ae = df[df['true_label'] == 1]['ae_loss']
                
                if len(normal_ae) > 0 and len(abnormal_ae) > 0:
                    # 95% 백분위수 기반 제안
                    ae_threshold_95 = normal_ae.quantile(0.95)
                    ae_threshold_90 = normal_ae.quantile(0.90)
                    
                    print(f"AE Loss 임계값 제안:")
                    print(f"  90% 백분위수: {ae_threshold_90:.6f}")
                    print(f"  95% 백분위수: {ae_threshold_95:.6f}")
                    
        except Exception as e:
            print(f"임계값 제안 실패: {e}")

def main():
    """분석 도구 실행"""
    from src.util.util_config import Config
    
    config = Config("config.json")
    analyzer = ResultAnalyzer(config)
    
    # 탐지 결과 분석
    detection_file = "results/detection_results_2025-08-26.csv"
    if os.path.exists(detection_file):
        detection_df = analyzer.analyze_detection_results(detection_file)
    
    # 평가 결과 분석
    evaluation_file = "results/evaluation_clip_results_2025-08-26.csv"
    if os.path.exists(evaluation_file):
        evaluation_df = analyzer.analyze_evaluation_results(evaluation_file)
        
        # 분포도 생성
        analyzer.plot_score_distributions(evaluation_df, "results/score_distributions.png")
        
        # 임계값 제안
        analyzer.suggest_thresholds(evaluation_df)
    
    # 진동 평가 결과 분석
    vib_evaluation_file = "results/vib_evaluation_clip_results_2025-08-26.csv"
    if os.path.exists(vib_evaluation_file):
        vib_evaluation_df = analyzer.analyze_evaluation_results(vib_evaluation_file)

if __name__ == "__main__":
    main()
