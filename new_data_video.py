import cv2
import os
import time
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp

class SignLanguageDataCollector:
    def __init__(self, output_dir="dataset", font_path="fonts/korean.ttf"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 한글 폰트 로드
        try:
            self.font = ImageFont.truetype(font_path, 40)
        except IOError:
            print(f"폰트 파일을 찾을 수 없습니다: {font_path}")
            self.font = ImageFont.load_default()
        
        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # 상태 변수
        self.current_label = ""
        self.recording = False
        self.countdown = 0
        self.start_time = 0
        self.frames = []
        self.video_writer = None

    def start_recording(self, label):
        """새로운 레이블로 녹화 시작"""
        self.current_label = label
        self.recording = True
        self.countdown = 3
        self.frames = []
        self.start_time = time.time()
        
        # 레이블 폴더 생성
        label_dir = os.path.join(self.output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # 동영상 파일명 생성 (타임스탬프 사용)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}.mp4"
        self.video_path = os.path.join(label_dir, filename)
        
        print(f"'{label}' 수어 데이터 수집을 시작합니다...")

    def stop_recording(self):
        """녹화 중지 및 저장"""
        if not self.recording or len(self.frames) == 0:
            return
            
        self.recording = False
        
        # 동영상 저장
        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, 30.0, (width, height))
        
        for frame in self.frames:
            out.write(frame)
        
        out.release()
        print(f"동영상 저장 완료: {self.video_path}")
        print(f"총 {len(self.frames)} 프레임 수집됨")

    def process_frame(self, frame):
        """프레임 처리 및 표시"""
        # OpenCV 프레임을 PIL 이미지로 변환 (한글 표시를 위해)
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        # 상태 메시지 표시
        if self.recording:
            # 녹화 중
            elapsed = time.time() - self.start_time
            
            if self.countdown > 0:
                # 카운트다운 표시
                count_text = f"{self.countdown}"
                text_width = self.font.getbbox(count_text)[2]
                text_x = (frame.shape[1] - text_width) // 2
                text_y = (frame.shape[0] // 2) - 50
                
                # 텍스트 배경
                draw.rectangle([(text_x-20, text_y-20), 
                               (text_x+text_width+20, text_y+80)], 
                              fill=(0, 0, 0, 128))
                
                # 카운트다운 숫자
                draw.text((text_x, text_y), count_text, font=self.font, fill=(0, 255, 0))
                
                # 1초마다 카운트다운 감소
                if elapsed > (4 - self.countdown):
                    self.countdown -= 1
            else:
                # 녹화 중 표시
                status_text = f"녹화 중: {self.current_label}"
                time_text = f"{int(5 - elapsed)}초 남음"
                
                # 상단 상태 바
                draw.rectangle([(0, 0), (frame.shape[1], 60)], fill=(0, 0, 0, 128))
                draw.text((20, 10), status_text, font=self.font, fill=(0, 255, 0))
                draw.text((frame.shape[1] - 150, 10), time_text, font=self.font, fill=(0, 255, 0))
                
                # 프레임 저장 (카운트다운 후)
                self.frames.append(frame.copy())
                
                # 5초 후 자동 중지
                if elapsed >= 5:
                    self.stop_recording()
        else:
            # 대기 중 상태
            draw.rectangle([(0, 0), (frame.shape[1], 60)], fill=(0, 0, 0, 128))
            status_text = "수어 데이터 수집기: 레이블 입력 후 스페이스바로 시작"
            draw.text((20, 10), status_text, font=self.font, fill=(255, 255, 0))
            
            # 사용 가능한 레이블 표시
            if os.path.exists(self.output_dir):
                labels = [d for d in os.listdir(self.output_dir) 
                         if os.path.isdir(os.path.join(self.output_dir, d))]
                labels_text = "사용 가능한 레이블: " + ", ".join(labels)
                draw.text((20, frame.shape[0] - 40), labels_text, font=self.font, fill=(255, 255, 255))
        
        # MediaPipe로 손 랜드마크 추출
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # PIL 이미지를 OpenCV로 변환
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        return frame

    def run(self):
        """메인 실행 루프"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("수어 데이터 수집기를 시작합니다...")
        print("사용 방법:")
        print("1. 한글 자음/모음(예: 'ㄱ') 입력 후 엔터")
        print("2. 스페이스바를 눌러 녹화 시작")
        print("3. 3초 카운트다운 후 5초간 녹화")
        print("4. 'q'를 눌러 종료")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 오류")
                break
            
            # 프레임 처리
            processed_frame = self.process_frame(frame)
            
            # 화면 표시
            cv2.imshow('Sign Language Data Collector', processed_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # 종료
                break
            elif key == 32:  # 스페이스바 - 녹화 시작
                if self.current_label and not self.recording:
                    self.start_recording(self.current_label)
            elif key == 13:  # 엔터 - 레이블 입력
                self.current_label = ""
                # 새 창에서 레이블 입력 받기
                cv2.destroyAllWindows()
                label = input("수어 레이블을 입력하세요 (예: ㄱ, ㄴ): ")
                if label:
                    self.current_label = label
                cv2.namedWindow('Sign Language Data Collector', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Sign Language Data Collector', 1280, 720)
        
        # 종료 처리
        if self.recording:
            self.stop_recording()
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("데이터 수집이 완료되었습니다.")

# 메인 실행
if __name__ == "__main__":
    collector = SignLanguageDataCollector()
    collector.run()