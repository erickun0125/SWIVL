# BiArt Environment Setup Guide

BiArt (Bimanual Articulated object manipulation) 환경을 시각적으로 테스트하기 위한 환경 세팅 가이드입니다.

## 1. Conda 환경 생성

### Option A: conda로 환경 생성 (권장)

```bash
# 1. 새로운 conda 환경 생성
conda create -n swivl python=3.11 -y

# 2. 환경 활성화
conda activate swivl

# 3. 필수 패키지 설치
conda install -c conda-forge pygame pymunk -y
pip install gymnasium numpy opencv-python shapely

# 4. (옵션) Jupyter notebook 사용시
pip install jupyter ipython matplotlib
```

### Option B: 기존 환경에 설치

```bash
# 현재 환경에 필요한 패키지만 설치
pip install -r requirements.txt
```

## 2. 환경 세팅 확인

```bash
# 패키지 설치 확인
python -c "import gymnasium, pygame, pymunk, cv2, shapely; print('All packages installed successfully!')"

# BiArt 환경 import 확인
python -c "import sys; sys.path.insert(0, '.'); import gym_biart; print('BiArt environment ready!')"
```

## 3. 디스플레이 설정

### Linux (GUI가 있는 경우)
```bash
# X11 디스플레이 확인
echo $DISPLAY

# DISPLAY가 비어있으면 설정
export DISPLAY=:0
```

### Linux (GUI가 없는 경우 - Headless)
```bash
# Xvfb (가상 디스플레이) 설치
sudo apt-get install xvfb

# 가상 디스플레이에서 실행
xvfb-run -a python gym_biart/example.py
```

### Docker 환경
```bash
# X11 forwarding 설정
xhost +local:docker

# Docker 실행시 디스플레이 연결
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ...
```

### macOS
```bash
# XQuartz 설치 필요
brew install --cask xquartz

# XQuartz 실행 후 환경변수 설정
export DISPLAY=:0
```

## 4. 실행 방법

### 기본 실행

```bash
# SWIVL 프로젝트 루트로 이동
cd /path/to/SWIVL

# Conda 환경 활성화
conda activate swivl

# 예제 실행 (human 모드 - 창이 뜹니다)
python gym_biart/example.py
```

### Headless 실행 (서버 환경)

```bash
# RGB array 모드로 실행 (창이 안뜨고 이미지만 저장)
python test_biart_simple.py
```

### Interactive 실행

```bash
# Python 인터프리터에서 직접 실행
python -i -c "import sys; sys.path.insert(0, '.'); import gymnasium as gym; import gym_biart; env = gym.make('gym_biart/BiArt-v0', render_mode='human'); obs, info = env.reset()"
```

## 5. 문제 해결

### pygame display error
```bash
# Error: "pygame.error: No available video device"
# 해결: 가상 디스플레이 사용
export SDL_VIDEODRIVER=dummy
python gym_biart/example.py
```

### import error
```bash
# Error: "ModuleNotFoundError: No module named 'gym_biart'"
# 해결: Python path 설정
export PYTHONPATH="${PYTHONPATH}:/path/to/SWIVL"
```

### OpenGL error
```bash
# Error: OpenGL 관련 에러
# 해결: EGL 백엔드 사용
export MUJOCO_GL=osmesa
```

## 6. 성능 최적화

```bash
# PyTorch가 설치되어 있다면 GPU 사용
export CUDA_VISIBLE_DEVICES=0

# 멀티코어 활용
export OMP_NUM_THREADS=4
```

## 7. 개발 환경 (VSCode)

```bash
# VSCode에서 Python 인터프리터 선택
# Command Palette (Ctrl+Shift+P) > Python: Select Interpreter
# > conda env:swivl 선택

# launch.json 설정 예시
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: BiArt Example",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/gym_biart/example.py",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        }
    ]
}
```

## 8. Jupyter Notebook 사용

```bash
# Jupyter 설치
pip install jupyter ipykernel

# 커널 등록
python -m ipykernel install --user --name swivl --display-name "Python (SWIVL)"

# Jupyter 실행
jupyter notebook

# 노트북에서 첫 셀에 다음 코드 실행:
# import sys
# sys.path.insert(0, '/path/to/SWIVL')
# import gym_biart
```

## 9. 패키지 버전 정보

현재 테스트된 버전:
- Python: 3.11
- gymnasium: 0.28.0+
- pymunk: 7.2.0
- pygame: 2.0.0+
- numpy: 1.20.0+
- opencv-python: 4.5.0+
- shapely: 1.8.0+

## 10. 추가 도구

### 녹화/스크린샷
```bash
# 환경 실행 중 스크린샷 저장 코드 추가
# (run_visualization.py 참고)
```

### 성능 프로파일링
```bash
# cProfile로 성능 측정
python -m cProfile -o profile.stats gym_biart/example.py

# 결과 확인
python -m pstats profile.stats
```
