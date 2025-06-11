import numpy as np

# 기존 클래스 로드
classes = np.load("korean_sign_frame_classes.npy", allow_pickle=True)

# 특수 명령 추가
classes = np.append(classes, ["지우기", "공백", "초기화"])

# 새로운 클래스 매핑 저장
np.save("korean_sign_frame_classes_with_commands.npy", classes)