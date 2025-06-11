import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
import cv2
import mediapipe as mp
import random
import csv

# 강화된 데이터 증강 함수
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
    
    # MediaPipe 초기화
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    
    for class_name in class_folders:
        class_dir = os.path.join(root_dir, class_name)
        video_files = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
        
        for video_file in video_files:
            video_path = os.path.join(class_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
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
                        
                        # 데이터 증강
                        if augment:
                            augmented_frames = augment_frame_data(frame_data)
                            for aug_frame in augmented_frames:
                                all_data.append(np.append(aug_frame, class_name))
                        else:
                            all_data.append(np.append(frame_data, class_name))
            
            cap.release()
    
    # CSV로 저장
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = [f'{ax}_{i}' for i in range(21) for ax in ['x', 'y', 'z']] + ['label']
        writer.writerow(header)
        writer.writerows(all_data)
    
    hands.close()
    print(f"Dataset created with {len(all_data)} samples")
    return all_data

# 단순화된 MLP 모델
def create_simple_mlp_model(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 메인 실행 코드
if __name__ == "__main__":
    # 데이터셋 생성 (프레임 단위)
    dataset_directory = "dataset"
    output_csv = "korean_sign_frame_dataset.csv"
    
    if not os.path.exists(output_csv):
        create_frame_dataset(dataset_directory, output_csv, augment=True)
    
    # 데이터 로드
    df = pd.read_csv(output_csv, encoding='utf-8-sig')
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df['label'].values
    
    # 클래스당 샘플 수 확인
    class_counts = df['label'].value_counts()
    print("\nClass distribution:")
    print(class_counts)
    
    # 레이블 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 모델 생성
    model = create_simple_mlp_model(
        input_shape=(X_train.shape[1],),
        num_classes=len(le.classes_)
    )
    model.summary()
    
    # 모델 학습 (강력한 정규화 적용)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint("best_frame_model.h5", save_best_only=True)
    ]
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=300,  # 충분한 기회 제공
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 평가
    print("\nEvaluating model...")
    model.load_weights("best_frame_model.h5")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    # 모델 저장
    model.save("korean_sign_frame_model.h5")
    np.save("korean_sign_frame_classes.npy", le.classes_)
    print("Model saved successfully!")