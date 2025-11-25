# SWIVL Paper Figures

이 폴더는 SWIVL 논문에 필요한 모든 figure 파일들을 포함합니다.

## 필요한 Figure 목록

### 1. `architecture_overview.pdf`
**위치**: `method.tex` - Section 3.2 (Architecture Overview)
**설명**: SWIVL 4-layer 계층 구조
- Layer 1: High-Level Policy (VLA, BC, Teleoperation)
- Layer 2: Reference Twist Field Generator
- Layer 3: Impedance Variable Modulation Policy
- Layer 4: Screw Axes-Decomposed Impedance Controller

**권장 내용**:
- 4개 레이어를 보여주는 블록 다이어그램
- 각 레이어 간 데이터 흐름 (waypoints → twists → impedance variables → wrenches)
- Wrench feedback 루프 표시
- 주파수 정보 (10Hz HL, 100Hz LL)

---

### 2. `objects.pdf`
**위치**: `experiments.tex` - Section 4.1 (Experimental Setup)
**설명**: SE(2) benchmark에서 사용되는 articulated object들

**권장 내용**:
- 2개 joint type을 보여주는 물체 일러스트:
  - **Revolute**: 회전 관절 (피봇 포인트 중심 회전)
  - **Prismatic**: 슬라이딩 관절 (축을 따라 직선 이동)
- 각 물체의 screw axis $\mathcal{S}$ 표시
- End-effector grasp 위치 표시
- 물체의 DoF 및 kinematic constraint 시각화

---

### 3. `force_comparison.pdf`
**위치**: `experiments.tex` - Section 4.2 (Results)
**설명**: Fighting force 프로파일 비교

**권장 내용**:
- 시간에 따른 bulk wrench magnitude $\|\mathcal{F}_{i,\perp}\|$ 그래프
- 3개 방법 비교: Pos-Ctrl, Imp-Ctrl, SWIVL
- Revolute 또는 Prismatic manipulation task 예시
- Y축: Force (N), X축: Time (s)
- SWIVL이 지속적으로 낮은 fighting force 유지하는 것 강조

---

### 4. `impedance_analysis.pdf`
**위치**: `experiments.tex` - Section 4.3 (Analysis)
**설명**: 학습된 impedance 변수 동작 시각화

**권장 내용**:
(a) Revolute manipulation 중 damping coefficients $d_\parallel$, $d_\perp$
    - Policy가 high $d_\perp$ (stiff bulk motion), low $d_\parallel$ (compliant articulation) 학습
(b) Joint type별 characteristic length $\alpha$ 변화
    - Task-appropriate metric structure 발견
(c) Impedance adjustment와 wrench feedback 상관관계
    - Reactive force regulation 시연

---

## Figure 생성 가이드라인

### 파일 형식
- **권장**: PDF (벡터 그래픽) 또는 PNG (고해상도 ≥300 DPI)
- NeurIPS 스타일에 맞춰 `\linewidth` 크기로 조정

### 색상 스키마 제안
- SWIVL (Ours): 파란색 계열 (#3498db)
- Pos-Ctrl: 빨간색 계열 (#e74c3c)
- Imp-Ctrl: 주황색 계열 (#f39c12)
- 배경: 흰색 또는 연한 회색

### 폰트
- Sans-serif (Helvetica, Arial 권장)
- 축 레이블: 10-12pt
- 범례: 8-10pt

---

## LaTeX에서 Figure 사용법

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/architecture_overview.pdf}
    \caption{\textbf{SWIVL Architecture.} ...}
    \label{fig:architecture_overview}
\end{figure}
```

---

## 현재 상태

| Figure | 파일명 | 상태 |
|--------|--------|------|
| Teaser (Bimanual Example) | `bimanipulation_example_image.jpg` | ✅ 완료 |
| Architecture Overview | `architecture_overview.pdf` | ⬜ 미생성 |
| Benchmark Objects | `objects.pdf` | ⬜ 미생성 |
| Force Comparison | `force_comparison.pdf` | ⬜ 미생성 |
| Impedance Analysis | `impedance_analysis.pdf` | ⬜ 미생성 |

---

## Teaser Figure (완료)

### `bimanipulation_example_image.jpg`
**위치**: `introduction.tex` - Section 1 시작부
**설명**: 양팔 로봇이 articulated object를 조작하는 실제 예시
- 두 로봇 팔이 파란색 링크로 연결된 물체를 협력하여 조작
- Kinematic constraint와 inter-arm force coupling의 실제 시나리오 시각화
- `\label{fig:teaser}`로 참조 가능

---

## 추가 권장 Figure

### 5. `teaser.pdf` (Optional)
**위치**: 논문 첫 페이지 상단
**설명**: SWIVL 컨셉을 한눈에 보여주는 teaser figure
- Bimanual robot manipulating articulated object
- Cognitive vs Physical Intelligence 구분
- SWIVL이 bridging하는 역할 시각화

### 6. `se2_geometry.pdf` (Optional for Appendix)
**위치**: `appendix_se2.tex`
**설명**: SE(2) 좌표계 및 screw decomposition 설명
- Spatial frame, body frame 정의
- Twist/wrench 표현
- Projection operators 시각화
