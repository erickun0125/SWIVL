# BiArt Environment - Quick Start Guide

가장 빠르게 BiArt 환경을 실행하는 방법입니다.

## 🚀 30초 빠른 시작

```bash
# 1. Conda 환경 생성 및 패키지 설치 (한 번만 실행)
bash setup_conda.sh

# 2. 환경 활성화
conda activate swivl

# 3. 시각화 실행!
python run_visualization.py
```

끝! 창이 뜨면서 로봇이 움직이는 것을 볼 수 있습니다.

---

## 📋 실행 옵션들

### 1. 기본 시각화 (추천)
```bash
python run_visualization.py --mode visual --joint revolute
```

### 2. 다양한 관절 타입 테스트
```bash
# Revolute joint (회전 관절)
python run_visualization.py --mode visual --joint revolute

# Prismatic joint (슬라이딩 관절)
python run_visualization.py --mode visual --joint prismatic

# Fixed joint (고정 관절)
python run_visualization.py --mode visual --joint fixed
```

### 3. 슬로우 모션
```bash
python run_visualization.py --mode visual --joint revolute --slow
```

### 4. 제어 패턴 테스트
```bash
python run_visualization.py --mode controlled --joint revolute
```

### 5. 관절 타입 비교
```bash
python run_visualization.py --mode compare
```

### 6. Headless 모드 (서버 환경)
```bash
# 디스플레이 없이 실행하고 이미지 저장
python run_visualization.py --mode headless --joint revolute --save-images
```

---

## 🎮 실행 예제들

### Example 1: 간단한 랜덤 액션 테스트
```bash
python gym_biart/example.py
```

### Example 2: 테스트 스크립트 (창 안뜸)
```bash
python test_biart_simple.py
```

### Example 3: Python 코드로 직접 실행
```python
import sys
sys.path.insert(0, '.')

import gymnasium as gym
import gym_biart

# 환경 생성
env = gym.make("gym_biart/BiArt-v0",
               render_mode="human",
               joint_type="revolute")

# 초기화
obs, info = env.reset()

# 1000 스텝 실행
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

## 🔧 문제 해결

### 문제 1: "No display" 에러
```bash
# 해결방법 1: 가상 디스플레이 사용
export SDL_VIDEODRIVER=dummy
python run_visualization.py --mode headless --save-images

# 해결방법 2: Xvfb 사용 (Linux)
xvfb-run -a python run_visualization.py
```

### 문제 2: "Module not found: gym_biart"
```bash
# 현재 디렉토리가 SWIVL 루트인지 확인
pwd  # /path/to/SWIVL 이어야 함

# 또는 PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 문제 3: Conda 환경이 없음
```bash
# Miniconda 설치 (Linux)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 설치 후 다시 시도
bash setup_conda.sh
```

---

## 📊 실행 결과 보기

실행하면 다음과 같은 정보가 출력됩니다:

```
[Episode 0] Step 50:
  Reward: 0.0027 | Total: 0.1350
  Tracking: 0.0027
  Safety: -0.0000
  Position Error: 37.11
  Success: False
```

**의미:**
- `Reward`: 현재 스텝의 보상
- `Tracking`: 목표 위치 추적 보상
- `Safety`: 안전성 페널티
- `Position Error`: 목표까지 거리 (pixel)
- `Success`: 목표 도달 여부

---

## 🎯 다음 단계

1. **강화학습 훈련**: RL 알고리즘으로 학습 시작
2. **커스텀 정책**: 자신만의 제어 정책 구현
3. **파라미터 튜닝**: 물리 파라미터, 보상 함수 조정
4. **SE(3) 환경**: IsaacLab으로 3D 환경 개발

---

## 📚 더 많은 정보

- **상세 문서**: [gym_biart/README.md](gym_biart/README.md)
- **세팅 가이드**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **API 문서**: [gym_biart/envs/biart.py](gym_biart/envs/biart.py)

---

## 💡 팁

1. **더 빠른 실행**: `--steps 500`으로 스텝 수 줄이기
2. **디버깅**: `--mode headless --save-images`로 프레임별 확인
3. **성능 측정**: `time python run_visualization.py --mode headless`

즐거운 연구 되세요! 🤖
