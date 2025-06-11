import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from PIL import Image, ImageDraw, ImageFont
import os

class SignLanguageSentenceBuilder:
    def __init__(self, model_path, class_mapping_path, font_path="fonts/korean.ttf"):
        # 모델 및 클래스 매핑 로드
        self.model = tf.keras.models.load_model(model_path)
        self.classes = np.load(class_mapping_path, allow_pickle=True)
        
        # 문장 관리 변수
        self.sentence = []  # 현재 작성 중인 문장 (문자 리스트)
        self.current_word = []  # 현재 인식된 문자 임시 저장
        self.last_char = None  # 마지막으로 인식된 문자
        self.last_char_time = 0  # 마지막 문자 인식 시간
        self.char_delay = 1.0  # 문자 추가 간격 (초)
        
        # 한글 폰트 로드
        try:
            self.font = ImageFont.truetype(font_path, 40)
            self.sentence_font = ImageFont.truetype(font_path, 50)
        except IOError:
            print("폰트 파일을 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            self.font = ImageFont.load_default()
            self.sentence_font = ImageFont.load_default()
        
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
            if prediction == self.last_char:
                self.stable_count += 1
                if self.stable_count > 5 and confidence > self.stable_confidence:
                    self.stable_prediction = prediction
                    self.stable_confidence = confidence
                    self.stable_count = 0
            else:
                self.stable_count = 0
                self.last_char = prediction
        
        # 안정적인 예측이 있고, 일정 시간이 지나면 문자 추가
        if self.stable_prediction:
            current_time = time.time()
            
            # 특수 명령 처리
            if self.stable_prediction == "지우기":
                if current_time - self.last_char_time > 1.0:  # 1초마다 한 글자씩 지움
                    if self.sentence:
                        self.sentence.pop()
                    self.last_char_time = current_time
            elif self.stable_prediction == "공백":
                if current_time - self.last_char_time > self.char_delay:
                    self.sentence.append(" ")
                    self.last_char_time = current_time
            elif self.stable_prediction == "초기화":
                if current_time - self.last_char_time > 2.0:  # 2초 이상 유지해야 초기화
                    self.sentence = []
                    self.last_char_time = current_time
            else:
                # 일반 문자 추가
                if current_time - self.last_char_time > self.char_delay:
                    self.sentence.append(self.stable_prediction)
                    self.last_char_time = current_time
        
        # OpenCV 이미지를 PIL 이미지로 변환
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        # FPS 표시 (영어라서 OpenCV로 직접 표시)
        fps_text = f"FPS: {int(fps)}"
        draw.rectangle([(10, 10), (200, 60)], fill=(0, 0, 0, 128))
        draw.text((20, 15), fps_text, font=self.font, fill=(0, 255, 0))
        
        # 현재 인식 결과 표시
        if self.stable_prediction:
            char_text = f"인식: {self.stable_prediction} ({self.stable_confidence:.2f})"
            draw.rectangle([(10, 70), (400, 130)], fill=(0, 0, 0, 128))
            draw.text((20, 75), char_text, font=self.font, fill=(0, 255, 0))
        
        # 문장 표시 영역
        sentence_text = "".join(self.sentence)
        if not sentence_text.strip():
            sentence_text = "문장을 작성하세요..."
        
        # 문장 배경
        draw.rectangle([(0, frame.shape[0]-100), 
                       (frame.shape[1], frame.shape[0])], 
                      fill=(50, 50, 50, 200))
        
        # 문장 텍스트
        draw.text((20, frame.shape[0]-90), sentence_text, 
                 font=self.sentence_font, fill=(255, 255, 0))
        
        # 안내 메시지
        help_text = "특수 명령: [지우기] - 글자 삭제, [공백] - 공백 추가, [초기화] - 전체 삭제"
        draw.text((20, frame.shape[0]-40), help_text, font=self.font, fill=(200, 200, 200))
        
        # PIL 이미지를 OpenCV 이미지로 변환
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        return frame

    def save_sentence(self):
        """현재 문장을 파일로 저장"""
        if not self.sentence:
            return False
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"sentence_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("".join(self.sentence))
        
        print(f"문장 저장 완료: {filename}")
        return True

# 메인 함수
def main():
    # 모델 및 클래스 매핑 로드
    model_path = "korean_sign_frame_model.h5"
    class_mapping_path = "korean_sign_frame_classes.npy"
    
    # 특수 명령을 위한 클래스 추가 (모델 학습 시 추가해야 함)
    # classes = np.load(class_mapping_path, allow_pickle=True)
    # classes = np.append(classes, ["지우기", "공백", "초기화"])
    # np.save("korean_sign_frame_classes_with_commands.npy", classes)
    
    # 클래스 매핑 파일 수정 버전 사용
    class_mapping_path = "korean_sign_frame_classes_with_commands.npy"
    
    # 문장 생성기 초기화
    sentence_builder = SignLanguageSentenceBuilder(model_path, class_mapping_path)
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("수어 문장 생성기를 시작합니다. 사용 방법:")
    print("- 수어 동작으로 문자 추가")
    print("- '지우기' 동작: 마지막 문자 삭제")
    print("- '공백' 동작: 공백 추가")
    print("- '초기화' 동작: 전체 문장 삭제")
    print("- 's' 키: 문장 저장")
    print("- 'c' 키: 문장 초기화")
    print("- 'q' 키: 종료")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("프레임 읽기 오류")
            continue
        
        # 프레임 처리
        processed_frame = sentence_builder.process_frame(frame)
        
        # 화면 표시
        cv2.imshow('Korean Sign Language Sentence Builder', processed_frame)
        
        # 키 입력 처리
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):  # 종료
            break
        elif key == ord('s'):  # 문장 저장
            sentence_builder.save_sentence()
        elif key == ord('c'):  # 문장 초기화
            sentence_builder.sentence = []
            print("문장 초기화 완료")
        elif key == ord(' '):  # 공백 추가
            sentence_builder.sentence.append(" ")
    
    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()
    print("프로그램 종료")

if __name__ == "__main__":
    main()