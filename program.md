# Autoresearch Program: 3-Cushion Vision-Only Kinematics Engine (`program.md`)

> Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and `andrej-karpathy-skills`.

## 1. Objective & Target Metrics
단일 60fps 카메라 환경(오디오 센서 배제)에서 쓰리쿠션(3-Cushion) 득점 여부를 판정하는 핵심 물리 엔진의 성능을 자율 실험 루프를 통해 극대화합니다.

* **Target Accuracy**: `Accuracy == 100.0%` (30개 극한 시나리오 전체 정답)
* **Target False Positive Rate (FPR)**: `0.0%` (1mm 근접 비접촉 통과, 2쿠션 조기 키스, 레일 쿠션 타기 오탐 제로)
* **Target False Negative Rate (FNR)**: `0.0%` (0.3mm 초미세 스침, 시속 35km 고속 뱅크샷 터널링, 코너 더블 쿠션, 프레임 드랍 미탐 제로)
* **Target Latency**: 프레임당 평균 연산 시간 **`<= 0.500 ms`**

---

## 2. 3-File Architecture & Rules
1. **`eval_billiards.py` (Immutable Evaluator — 수정 절대 금지)**
   * 30개의 까다로운 물리 시나리오(0.3mm 초미세 스침, 35km/h 고속 뱅크샷, 1mm 근접 비접촉 통과, 코너 더블 쿠션, 프레임 드랍 및 지터 노이즈)를 구동하고 `Accuracy (%)`, `FPR (%)`, `FNR (%)`, `Avg Frame Latency (ms)`를 정량 평가합니다.
2. **`three_cushion_vision/kinematics_detector.py` (Agent-Modifiable Target)**
   * 자율 실험 루프에서 유일하게 수정이 허용되는 타겟 파일입니다.
   * Sub-frame CCD 보간 수식, 고속 레일 터널링(Sub-frame Apex) 감지, 1mm 비접촉 vs 0.3mm 초미세 스침 방향성 충격량/굴절각 필터, 속도 스무딩 및 스칼라 벡터 최적화를 수행합니다.
3. **`program.md` (Autoresearch Specification)**
   * 자율 연구 루프의 목표, 제약 조건, 실험 프로토콜을 정의합니다.

---

## 3. Autonomous Experimentation Protocol
1. **Baseline 측정**: `python eval_billiards.py`를 실행하여 초기 정확도, 실패 시나리오(Failure Cases), 프레임당 처리 시간을 기록합니다.
2. **가설 수립 및 Surgical Change**: 실패 원인을 수학적·물리적으로 분석하고 `three_cushion_vision/kinematics_detector.py`만 정밀 수정합니다.
3. **검증 및 Git 브랜치/커밋 관리**:
   * `python eval_billiards.py` 재실행 후 정확도가 향상되었거나 동일 정확도에서 처리 속도가 단축된 경우:
     `git commit -am "exp: [개선 내용 및 점수]"`
   * 성능이 하락한 경우 즉시 롤백:
     `git checkout -- three_cushion_vision/kinematics_detector.py`
4. **종료 조건**: `Accuracy == 100.0%` 및 `Latency <= 0.500 ms` 달성 시 루프를 종료하고 전체 실험 로그 테이블을 보고합니다.
