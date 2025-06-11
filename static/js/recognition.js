document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const startBtn = document.getElementById('start-btn');
    const addBtn = document.getElementById('add-btn');
    const resultChar = document.getElementById('result-char');
    const confidenceEl = document.getElementById('confidence');
    
    let stream = null;
    let isProcessing = false;
    let currentPrediction = null;
    let currentConfidence = 0;
    const sendInterval = 500; // 0.5초마다 프레임 전송
    let lastSentTime = 0;

    // 웹캠 시작
    startBtn.addEventListener('click', async () => {
        try {
            if (stream) {
                stopCamera();
                startBtn.textContent = '웹캠 시작';
                addBtn.disabled = true;
                return;
            }
            
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            video.srcObject = stream;
            startBtn.textContent = '웹캠 중지';
            isProcessing = true;
            processVideo();
        } catch (err) {
            console.error('웹캠 접근 오류:', err);
            alert('웹캠에 접근할 수 없습니다. 사용 권한을 확인해주세요.');
        }
    });
    
    // 문자 추가 버튼
    addBtn.addEventListener('click', async () => {
        if (!currentPrediction) return;
        
        try {
            const response = await fetch('/add_to_sentence', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ char: currentPrediction })
            });
            
            const data = await response.json();
            if (data.success) {
                alert(`"${currentPrediction}" 문자가 문장에 추가되었습니다!`);
            } else {
                alert(`문자 추가 실패: ${data.error || '알 수 없는 오류'}`);
            }
        } catch (err) {
            console.error('문자 추가 오류:', err);
            alert('문자 추가 중 오류가 발생했습니다.');
        }
    });
    
    // 웹캠 중지
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        isProcessing = false;
        resultChar.textContent = '-';
        confidenceEl.textContent = '신뢰도: 0%';
    }
    
    // 비디오 처리
    function processVideo() {
        if (!isProcessing) return;
        
        const now = Date.now();
        if (now - lastSentTime >= sendInterval) {
            captureFrame();
            lastSentTime = now;
        }
        
        requestAnimationFrame(processVideo);
    }
    
    // 프레임 캡처 및 서버 전송
    function captureFrame() {
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // 캔버스 이미지를 데이터 URL로 변환
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // 서버로 전송
        fetch('/recognize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('서버 응답 오류');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                console.error('인식 오류:', data.error);
                return;
            }
            
            // 결과 업데이트
            if (data.prediction) {
                resultChar.textContent = data.prediction;
                confidenceEl.textContent = `신뢰도: ${(data.confidence * 100).toFixed(1)}%`;
                currentPrediction = data.prediction;
                currentConfidence = data.confidence;
                addBtn.disabled = false;
            } else {
                resultChar.textContent = '-';
                confidenceEl.textContent = '신뢰도: 0%';
                currentPrediction = null;
                addBtn.disabled = true;
            }
        })
        .catch(err => console.error('인식 오류:', err));
    }
    
    // 페이지 언로드 시 웹캠 정리
    window.addEventListener('beforeunload', () => {
        stopCamera();
    });
});