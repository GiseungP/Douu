import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from PIL import Image, ImageDraw, ImageFont  # PIL 라이브러리 추가

class SignLanguageRecognizer:
    def __init__(self, model_path, class_mapping_path, font_path):
        # 모델 및 클래스 매핑 로드
        self.model = tf.keras.models.load_model(model_path)
        self.classes = np.load(class_mapping_path, allow_pickle=True)
        
        # 한글 폰트 로드
        self.font = ImageFont.truetype(font_path, 60)  # 폰트 크기 60
        
        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # FPS 계산을 위한 변수
        self.prev_time = 0
        self.curr_time = 0
        
        # 예측 결과 저장
        self.last_prediction = None
        self.last_confidence = 0
        self.stable_prediction = None
        self.stable_confidence = 0
        self.stable_count = 0

    def process_frame(self, frame):
        # FPS 계산
        self.curr_time = time.time()
        fps = 1 / (self.curr_time - self.prev_time)
        self.prev_time = self.curr_time
        
        # 프레임 처리
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        
        prediction = None
        confidence = 0
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 랜드마크 그리기
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # 랜드마크 데이터 추출
            frame_data = []
            for lm in hand_landmarks.landmark:
                frame_data.extend([lm.x, lm.y, lm.z])
            
            # 모델 예측
            input_data = np.array([frame_data]).astype(np.float32)
            predictions = self.model.predict(input_data, verbose=0)
            
            # 예측 결과 처리
            pred_idx = np.argmax(predictions[0])
            prediction = self.classes[pred_idx]
            confidence = np.max(predictions[0])
            
            # 안정적인 예측을 위한 로직 (0.5초 이상 같은 예측이 지속될 때만 업데이트)
            if prediction == self.last_prediction:
                self.stable_count += 1
                if self.stable_count > 5 and confidence > self.stable_confidence:
                    self.stable_prediction = prediction
                    self.stable_confidence = confidence
            else:
                self.stable_count = 0
                self.last_prediction = prediction
        
        # FPS 표시 (영어라서 OpenCV로 직접 표시)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 한글 결과 표시
        if self.stable_prediction:
            # OpenCV 이미지를 PIL 이미지로 변환
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            
            # 한글 텍스트 설정
            display_text = f"{self.stable_prediction} ({self.stable_confidence:.2f})"
            text_width = self.font.getbbox(display_text)[2]  # 텍스트 너비 계산
            
            # 텍스트 배경 사각형 그리기
            draw.rectangle([(10, 50), (text_width + 30, 120)], fill=(0, 0, 0))
            
            # 한글 텍스트 그리기
            draw.text((20, 50), display_text, font=self.font, fill=(0, 255, 0))
            
            # PIL 이미지를 OpenCV 이미지로 변환
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        return frame, prediction, confidence

# 메인 함수
def main():
    # 모델 및 클래스 매핑 로드
    model_path = "korean_sign_frame_model.h5"
    class_mapping_path = "korean_sign_frame_classes.npy"
    font_path = "fonts/korean.ttf"  # 폰트 파일 경로
    
    recognizer = SignLanguageRecognizer(model_path, class_mapping_path, font_path)
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Starting sign language recognition. Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # 프레임 처리
        processed_frame, prediction, confidence = recognizer.process_frame(frame)
        
        # 화면 표시
        cv2.imshow('Korean Sign Language Recognition', processed_frame)
        
        # 'q' 키로 종료
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()