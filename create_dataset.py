import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import random
from tqdm import tqdm

def extract_hand_landmarks(video_path):
    """동영상에서 손 랜드마크 추출"""
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    
    # MediaPipe Hands 인스턴스 생성 (동영상 처리용)
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            # MediaPipe 처리
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 랜드마크 정규화 좌표 추출 (21개 포인트)
                    frame_data = []
                    for lm in hand_landmarks.landmark:
                        frame_data.extend([lm.x, lm.y, lm.z])
                    landmarks_list.append(frame_data)
                    
    cap.release()
    return landmarks_list

# 강화된 데이터 증강 함수 (프레임 단위)
def augment_frame_data(frame):
    augmented_frames = []
    
    # 원본 데이터
    augmented_frames.append(frame)
    
    # 좌우 반전 (x 좌표 반전)
    flipped = frame.copy()
    for i in range(0, len(flipped), 3):  # x 좌표만 처리
        flipped[i] = 1.0 - flipped[i]
    augmented_frames.append(flipped)
    
    # 작은 이동 변환
    for _ in range(3):  # 3가지 이동 버전 생성
        shifted = frame.copy()
        dx = random.uniform(-0.05, 0.05)  # x 이동
        dy = random.uniform(-0.05, 0.05)  # y 이동
        for i in range(0, len(shifted), 3):
            shifted[i] += dx   # x 이동
            shifted[i+1] += dy  # y 이동
        augmented_frames.append(shifted)
    
    # 스케일 변환
    scaled = frame.copy()
    scale_factor = random.uniform(0.9, 1.1)  # 10% 확대/축소
    for i in range(0, len(scaled), 3):
        scaled[i] = 0.5 + (scaled[i] - 0.5) * scale_factor  # x 스케일
        scaled[i+1] = 0.5 + (scaled[i+1] - 0.5) * scale_factor  # y 스케일
    augmented_frames.append(scaled)
    
    # 회전 변환 (간단한 2D 회전)
    angle = random.uniform(-15, 15) * np.pi / 180  # -15~15도 회전
    rotated = frame.copy()
    for i in range(0, len(rotated), 3):
        x = rotated[i] - 0.5
        y = rotated[i+1] - 0.5
        rotated[i] = 0.5 + x * np.cos(angle) - y * np.sin(angle)
        rotated[i+1] = 0.5 + x * np.sin(angle) + y * np.cos(angle)
    augmented_frames.append(rotated)
    
    return augmented_frames

# 데이터셋 생성 함수 (프레임 단위)
def create_frame_dataset(root_dir, output_file, augment=True):
    """폴더 구조의 동영상 처리 (프레임 단위)"""
    all_data = []
    class_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    
    # 클래스별 진행 상황 표시
    for class_name in tqdm(class_folders, desc="Classes"):
        class_dir = os.path.join(root_dir, class_name)
        video_files = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
        
        # 비디오 파일별 처리
        for video_file in tqdm(video_files, desc=f"Videos in {class_name}"):
            video_path = os.path.join(class_dir, video_file)
            
            # 랜드마크 추출
            landmarks = extract_hand_landmarks(video_path)
            
            # 각 프레임에 레이블 추가
            for frame in landmarks:
                if augment:
                    augmented_frames = augment_frame_data(frame)
                    for aug_frame in augmented_frames:
                        all_data.append(np.append(aug_frame, class_name))
                else:
                    all_data.append(np.append(frame, class_name))
    
    # CSV로 저장 (인코딩 명시적으로 지정)
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = [f'{ax}_{i}' for i in range(21) for ax in ['x', 'y', 'z']] + ['label']
        writer.writerow(header)
        writer.writerows(all_data)
    
    print(f"Dataset created with {len(all_data)} samples")
    return all_data

# 데이터셋 업데이트 함수
def update_dataset_with_new_videos(root_dir, existing_csv, output_csv):
    """새로운 동영상만 추가하여 데이터셋 업데이트"""
    # 기존 데이터셋 로드
    existing_data = []
    existing_videos = set()
    
    if os.path.exists(existing_csv):
        with open(existing_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader)  # 헤더 건너뛰기
            for row in reader:
                existing_data.append(row)
                # 마지막 열이 레이블이므로, 비디오 파일명은 추출 불가
                # 대신 모든 데이터를 유지하고 새 데이터만 추가하는 방식으로
        
        print(f"Loaded {len(existing_data)} existing samples")
    
    # 새로 추가된 동영상 찾기
    new_data = []
    class_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    
    for class_name in tqdm(class_folders, desc="Checking new videos"):
        class_dir = os.path.join(root_dir, class_name)
        video_files = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
        
        for video_file in video_files:
            video_path = os.path.join(class_dir, video_file)
            
            # 기존 데이터셋에 이 비디오가 있는지 확인
            # 간단한 방법: 모든 동영상을 다시 처리 (실제로는 타임스탬프로 관리하는 것이 좋음)
            # 여기서는 모든 동영상을 처리하고 중복을 제거하지 않음
            # (동일한 동영상은 같은 데이터를 생성하므로 증강 후 중복될 수 있음)
            
            # 랜드마크 추출
            landmarks = extract_hand_landmarks(video_path)
            
            # 각 프레임에 레이블 추가
            for frame in landmarks:
                augmented_frames = augment_frame_data(frame)
                for aug_frame in augmented_frames:
                    new_data.append(np.append(aug_frame, class_name))
    
    print(f"Found {len(new_data)} new samples")
    
    # 기존 데이터와 새 데이터 결합
    all_data = existing_data + new_data
    
    # CSV로 저장
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = [f'{ax}_{i}' for i in range(21) for ax in ['x', 'y', 'z']] + ['label']
        writer.writerow(header)
        writer.writerows(all_data)
    
    print(f"Updated dataset created with {len(all_data)} samples")
    return all_data

# 메인 실행 옵션
if __name__ == "__main__":
    dataset_directory = "dataset"
    output_csv = "korean_sign_frame_dataset.csv"
    existing_csv = "korean_sign_frame_dataset.csv"  # 기존 데이터셋
    
    # 옵션 1: 전체 데이터셋 새로 생성
    create_frame_dataset(dataset_directory, output_csv, augment=True)
    
    # 옵션 2: 새로운 동영상만 추가하여 업데이트
    # update_dataset_with_new_videos(dataset_directory, existing_csv, output_csv)