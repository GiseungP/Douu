from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import base64
import time
import os
import uuid
import logging
import threading
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import random
import csv
import subprocess

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SAVED_SENTENCES_FOLDER'] = 'saved_sentences'
app.config['DEBUG_FRAMES_FOLDER'] = 'debug_frames'
app.config['MODEL_FOLDER'] = 'model'
app.config['DATASET_FILE'] = 'korean_sign_frame_dataset.csv'

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SignLanguageApp')

# 필요한 폴더 생성
for folder in [app.config['UPLOAD_FOLDER'], 
               app.config['SAVED_SENTENCES_FOLDER'], 
               app.config['DEBUG_FRAMES_FOLDER'],
               app.config['MODEL_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 모델 및 클래스 매핑 로드
model = None
classes = []
current_model_version = 0
le = LabelEncoder()

# 한글 자모 분류
CONSONANTS = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
VOWELS = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

def load_latest_model():
    global model, classes, current_model_version, le
    
    # 클래스 로드 시도
    classes_path = os.path.join(app.config['MODEL_FOLDER'], 'korean_sign_frame_classes.npy')
    if os.path.exists(classes_path):
        classes = np.load(classes_path, allow_pickle=True)
        le.fit(classes)  # LabelEncoder 초기화
        logger.info(f"✅ 클래스 로드 성공! 개수: {len(classes)}")
    else:
        logger.warning("⚠️ 클래스 파일이 없습니다. 첫 학습 후 생성됩니다.")
    
    try:
        # 가장 최신 모델 찾기
        model_files = [f for f in os.listdir(app.config['MODEL_FOLDER']) 
                      if f.startswith('korean_sign_frame_model_v') and f.endswith('.h5')]
        
        if model_files:
            # 버전 번호로 정렬 (v숫자.h5)
            model_files.sort(key=lambda x: int(x.split('_v')[1].split('.')[0]), reverse=True)
            latest_model = model_files[0]
            current_model_version = int(latest_model.split('_v')[1].split('.')[0])
            
            model_path = os.path.join(app.config['MODEL_FOLDER'], latest_model)
            model = tf.keras.models.load_model(model_path)
            logger.info(f"✅ 최신 모델 로드 성공! (버전: v{current_model_version})")
        else:
            logger.warning("⚠️ 초기 모델 파일이 없습니다. 첫 번째 학습 후 생성됩니다.")
    except Exception as e:
        logger.error(f"❌ 모델 로드 오류: {str(e)}")

# 서버 시작 시 모델 로드
load_latest_model()

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 원래 학습 코드의 데이터 증강 함수 (동일하게 유지)
def augment_frame_data(frame):
    """원래 학습 코드의 증강 함수와 동일"""
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

def process_frame(frame, debug=False):
    if model is None:
        return None, 0
    
    try:
        # 프레임 크기 조정: 학습 시 사용한 640x480으로 조정
        frame = cv2.resize(frame, (640, 480))
        
        # 프레임 처리 (RGB 변환)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe로 손 인식
        results = hands.process(frame_rgb)
        
        prediction = None
        confidence = 0
        
        if results.multi_hand_landmarks:
            # 첫 번째 손의 랜드마크 사용
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 디버그: 랜드마크 그리기
            if debug:
                debug_frame = frame.copy()
                mp_drawing.draw_landmarks(
                    debug_frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                # 디버그 프레임 저장
                timestamp = int(time.time() * 1000)
                debug_frame_path = os.path.join(app.config['DEBUG_FRAMES_FOLDER'], f"frame_{timestamp}.jpg")
                cv2.imwrite(debug_frame_path, debug_frame)
                logger.info(f"📸 디버그 프레임 저장: {debug_frame_path}")
            
            # 학습 코드와 동일하게 랜드마크 추출
            frame_data = []
            for lm in hand_landmarks.landmark:
                frame_data.extend([lm.x, lm.y, lm.z])
            
            # 랜드마크 개수 확인 (21개 랜드마크 * 3 = 63)
            if len(frame_data) != 63:
                logger.warning(f"⚠️ 랜드마크 개수 불일치: {len(frame_data)} (기대값: 63)")
                return None, 0
            
            # 모델 입력 형식에 맞게 배열 재구성
            input_data = np.array(frame_data, dtype=np.float32).reshape(1, -1)
            
            # 입력 차원 검증 (63 고정)
            if input_data.shape[1] != 63:
                logger.error(f"❌ 입력 차원 불일치: 기대값 63 vs 입력 {input_data.shape[1]}")
                return None, 0
            
            # 모델 예측
            predictions = model.predict(input_data, verbose=0)
            
            # 예측 결과 처리
            pred_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # 클래스 매핑 확인
            if classes.size > 0 and 0 <= pred_idx < len(classes):
                prediction = classes[pred_idx]
                logger.info(f"🔍 인식 성공: {prediction} (신뢰도: {confidence:.2f})")
            else:
                logger.warning(f"⚠️ 인덱스 오류: pred_idx={pred_idx}, 클래스 개수={len(classes)}")
                prediction = None
                confidence = 0
        
        else:
            logger.info("🖐️ 손이 인식되지 않았습니다.")
            if debug:
                timestamp = int(time.time() * 1000)
                debug_frame_path = os.path.join(app.config['DEBUG_FRAMES_FOLDER'], f"no_hand_{timestamp}.jpg")
                cv2.imwrite(debug_frame_path, frame)
                logger.info(f"📸 손 미인식 프레임 저장: {debug_frame_path}")
        
        return prediction, confidence
    
    except Exception as e:
        logger.error(f"❌ 프레임 처리 오류: {str(e)}", exc_info=True)
        return None, 0

# 라우트 정의
@app.route('/')
def index():
    session['user_id'] = str(uuid.uuid4())
    session['sentence'] = ""  # 문자열로 문장 저장
    return render_template('index.html')

@app.route('/sentence')
def sentence_page():
    if 'sentence' not in session:
        session['sentence'] = ""
    return render_template('sentence.html', sentence=session['sentence'])

@app.route('/add_sign')
def add_sign_page():
    return render_template('add_sign.html')

# API 엔드포인트
@app.route('/recognize', methods=['POST'])
def recognize():
    if model is None:
        return jsonify({
            'prediction': None,
            'confidence': 0,
            'error': '모델 로드 실패'
        })
    
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': '이미지 데이터 없음'})
    
    try:
        # Base64 이미지 디코딩
        header, encoded = data['image'].split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("❌ 이미지 디코딩 실패")
            return jsonify({'error': '이미지 디코딩 실패'})
        
        # 프레임 처리 (디버그 모드 활성화)
        prediction, confidence = process_frame(frame, debug=True)
        
        # 자음/모음만 반환
        if prediction and (prediction in CONSONANTS or prediction in VOWELS):
            return jsonify({
                'prediction': prediction,
                'confidence': confidence
            })
        else:
            return jsonify({
                'prediction': None,
                'confidence': 0
            })
    
    except Exception as e:
        logger.error(f"❌ 인식 요청 오류: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/add_to_sentence', methods=['POST'])
def add_to_sentence():
    if 'sentence' not in session:
        session['sentence'] = ""
    
    char = request.json.get('char')
    if char:
        # 새 문자 추가
        new_sentence = session['sentence'] + char
        session['sentence'] = new_sentence
        
        return jsonify({
            'success': True,
            'sentence': new_sentence
        })
    
    return jsonify({'success': False, 'error': '문자 없음'})

@app.route('/reset_sentence', methods=['POST'])
def reset_sentence():
    session['sentence'] = ""
    return jsonify({
        'success': True,
        'sentence': ''
    })

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    current_sentence = session.get('sentence', "")
    return jsonify({
        'sentence': current_sentence
    })

@app.route('/save_sentence', methods=['POST'])
def save_sentence():
    current_sentence = session.get('sentence', "")
    
    if not current_sentence:
        return jsonify({'success': False, 'error': '저장할 문장 없음'})
    
    try:
        save_folder = app.config['SAVED_SENTENCES_FOLDER']
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"sentence_{timestamp}_{session['user_id'][:8]}.txt"
        filepath = os.path.join(save_folder, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(current_sentence)
        
        logger.info(f"💾 문장 저장: {filepath}")
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        logger.error(f"❌ 문장 저장 오류: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# 비디오 처리 및 학습 관련 함수들 (동일하게 유지)
def extract_frames_from_video(video_path, target_frames=30):
    """동영상에서 대표 프레임 추출 (webm 직접 지원)"""
    logger.info(f"📹 프레임 추출 시작: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        logger.error(f"❌ 동영상 열기 실패: {video_path}")
        return frames
    
    try:
        # 동영상에서 프레임 샘플링
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"총 프레임 수: {total_frames}, FPS: {fps}")
        
        if total_frames < 1 or fps <= 0:
            logger.warning("⚠️ 유효하지 않은 동영상 정보, 기본 추출 방식으로 전환")
            # 기본 추출 방식: 1초당 1프레임
            fps = 30  # 기본 FPS 값
            total_frames = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                total_frames += 1
                # 1초당 1프레임 추출
                if total_frames % int(fps) == 0:
                    frame = cv2.resize(frame, (640, 480))
                    frames.append(frame)
            logger.info(f"🔄 기본 추출 방식으로 프레임 추출: {len(frames)}개")
            return frames
    
    except Exception as e:
        logger.error(f"❌ 동영상 정보 추출 오류: {str(e)}")
        # 오류 발생 시 기본 추출 방식으로 전환
        total_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            # 1초당 1프레임 추출 (30 FPS 가정)
            if total_frames % 30 == 0:
                frame = cv2.resize(frame, (640, 480))
                frames.append(frame)
        logger.info(f"🔄 예외 후 기본 추출 방식으로 프레임 추출: {len(frames)}개")
        return frames
    
    try:
        # 프레임 샘플링 전략 개선
        if total_frames < target_frames:
            # 프레임 수가 적으면 모든 프레임 사용
            frame_indices = range(0, total_frames)
        else:
            # 고르게 분포된 프레임 선택
            frame_indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 프레임 전처리 (크기 조정: 학습과 동일한 640x480)
                frame = cv2.resize(frame, (640, 480))
                frames.append(frame)
            else:
                logger.warning(f"⚠️ 프레임 읽기 실패: 인덱스 {idx}")
    
    except Exception as e:
        logger.error(f"❌ 프레임 추출 중 오류: {str(e)}", exc_info=True)
        # 오류 발생 시 기본 추출 방식으로 전환
        cap.release()
        cap = cv2.VideoCapture(video_path)
        total_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            # 1초당 1프레임 추출 (30 FPS 가정)
            if total_frames % 30 == 0:
                frame = cv2.resize(frame, (640, 480))
                frames.append(frame)
        logger.info(f"🔄 예외 후 기본 추출 방식으로 프레임 추출: {len(frames)}개")
    
    finally:
        cap.release()
    
    logger.info(f"✅ 추출된 프레임 수: {len(frames)}")
    return frames

def extract_landmarks(frame):
    """프레임에서 랜드마크 추출 (학습 데이터 형식으로)"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        frame_data = []
        for lm in hand_landmarks.landmark:
            frame_data.extend([lm.x, lm.y, lm.z])
        return frame_data
    return None

def create_simple_mlp_model(input_shape, num_classes):
    """원래 학습 코드와 동일한 모델 구조"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def update_dataset(new_data, label):
    """CSV 데이터셋 업데이트 (원래 학습 포맷 유지)"""
    dataset_path = app.config['DATASET_FILE']
    
    # 기존 데이터셋 로드 (존재하는 경우)
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path, encoding='utf-8-sig')
    else:
        # 새 데이터셋 생성
        columns = [f'{ax}_{i}' for i in range(21) for ax in ['x', 'y', 'z']] + ['label']
        df = pd.DataFrame(columns=columns)
    
    # 새 데이터 추가
    for data in new_data:
        row = list(data) + [label]
        df.loc[len(df)] = row
    
    # CSV 저장
    df.to_csv(dataset_path, index=False, encoding='utf-8-sig')
    logger.info(f"💾 데이터셋 업데이트: {dataset_path} (새 샘플: {len(new_data)}개)")

def retrain_model_with_new_data(video_path, label, category):
    """새 동영상 데이터로 모델 재학습 (webm 직접 지원)"""
    global model, classes, current_model_version, le
    
    try:
        logger.info(f"🚀 모델 재학습 시작: {label} ({category})")
        
        # 1. 동영상 프레임 추출 (변환 없이 직접 처리)
        frames = extract_frames_from_video(video_path, target_frames=30)
        
        if not frames:
            logger.error("❌ 프레임 추출 실패 - 빈 프레임 목록")
            return False
            
        logger.info(f"📹 추출된 프레임 수: {len(frames)}")
            
        # 2. 프레임 처리 및 랜드마크 추출 + 증강
        new_data = []
        valid_frames = 0
        
        for i, frame in enumerate(frames):
            landmarks = extract_landmarks(frame)
            if landmarks and len(landmarks) == 63:
                valid_frames += 1
                # 원래 학습 코드의 증강 적용
                augmented_frames = augment_frame_data(landmarks)
                new_data.extend(augmented_frames)
            else:
                logger.warning(f"⚠️ 프레임 {i+1}에서 랜드마크 추출 실패")
        
        logger.info(f"🔍 유효한 프레임 수: {valid_frames}/{len(frames)}")
        
        if not new_data:
            logger.error("❌ 유효한 랜드마크 데이터 없음")
            return False
        
        # 3. 데이터셋 업데이트 (CSV 파일)
        update_dataset(new_data, label)
        
        # 4. 전체 데이터셋 재로드
        dataset_path = app.config['DATASET_FILE']
        if not os.path.exists(dataset_path):
            logger.error("❌ 데이터셋 파일을 찾을 수 없음")
            return False
            
        df = pd.read_csv(dataset_path, encoding='utf-8-sig')
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df['label'].values
        
        # 5. 클래스 업데이트
        classes = np.unique(y)
        np.save(os.path.join(app.config['MODEL_FOLDER'], 'korean_sign_frame_classes.npy'), classes)
        le.fit(classes)
        
        # 6. 레이블 인코딩
        y_encoded = le.transform(y)
        y_categorical = to_categorical(y_encoded, num_classes=len(classes))
        
        # 7. 데이터 분할 (전체 데이터 재학습)
        # 재학습 시에는 전체 데이터 사용
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # 8. 모델 생성 또는 업데이트
        if model is None:
            logger.info("🆕 새 모델 생성 (원래 학습 구조)")
            new_model = create_simple_mlp_model(
                input_shape=(X_train.shape[1],),
                num_classes=len(classes))
        else:
            # 기존 모델 구조 복제
            logger.info("🔄 기존 모델 복제")
            new_model = create_simple_mlp_model(
                input_shape=(X_train.shape[1],),
                num_classes=len(classes))
        
        # 9. 모델 학습 (원래 학습 설정과 동일)
        logger.info(f"🔁 모델 재학습 시작: {X_train.shape[0]}개 샘플")
        history = new_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            verbose=0
        )
        
        # 10. 새 모델 저장
        current_model_version += 1
        new_model_path = os.path.join(app.config['MODEL_FOLDER'], f'korean_sign_frame_model_v{current_model_version}.h5')
        new_model.save(new_model_path)
        
        # 11. 모델 교체
        model = new_model
        logger.info(f"✅ 모델 업데이트 완료! 버전: v{current_model_version}")
        logger.info(f"현재 클래스 개수: {len(classes)}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ 모델 재학습 실패: {str(e)}", exc_info=True)
        return False

# 수어 추가 API 엔드포인트 (개선된 버전)
@app.route('/add_sign', methods=['POST'])
def add_sign():
    try:
        # 필수 항목: 단어와 파일
        word = request.form.get('word')
        file = request.files.get('file')
        
        # 선택 항목 (기본값 설정)
        meaning = request.form.get('meaning', '추가 예정')
        category = request.form.get('category', '기타')
        description = request.form.get('description', '추가 예정')
        
        # 필수 항목 검증
        if not word or not file:
            return jsonify({
                'success': False, 
                'error': '수어 단어와 동영상 파일은 필수 항목입니다'
            })
        
        # 파일 확장자 확인
        filename = file.filename
        _, ext = os.path.splitext(filename)
        supported_formats = ['.mp4', '.avi', '.mov', '.webm']
        
        if ext.lower() not in supported_formats:
            return jsonify({
                'success': False, 
                'error': f'지원하는 동영상 형식이 아닙니다: {", ".join(supported_formats)}'
            })
        
        # 안전한 파일명 생성
        new_filename = f"sign_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        file.save(filepath)
        logger.info(f"📁 파일 저장: {filepath}")
        
        # 비동기로 재학습 시작
        threading.Thread(
            target=retrain_model_with_new_data,
            args=(filepath, word, category)
        ).start()
        
        return jsonify({
            'success': True,
            'message': '수어가 성공적으로 등록되었습니다! 모델 재학습이 시작됩니다.',
            'data': {
                'word': word,
                'meaning': meaning,
                'category': category,
                'description': description,
                'file_path': filepath
            }
        })
        
    except Exception as e:
        logger.error(f"❌ 수어 등록 오류: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)