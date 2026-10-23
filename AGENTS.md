# Context7 MCP Auto-Invocation Rule

- Always use the `context7` MCP tools (`resolve-library-id` and `get-library-docs` / `query-docs`) automatically whenever writing, debugging, or upgrading code involving external libraries, frameworks, SDKs, or APIs (e.g., OpenCV, PyTorch, Ultralytics YOLO, Supervision, FastAPI, Pydantic, React, Vite, Next.js, Tailwind CSS, Supabase, Firebase, Flutter, Toss Payments, PortOne) — without waiting for the user to explicitly type `"use context7"`.
- Workflow for automatic invocation:
  1. Call `resolve-library-id` with the target library/package name to obtain its official Context7-compatible library ID.
  2. Call `get-library-docs` (or `query-docs`) with that library ID and the specific topic/API being implemented or debugged.
- Prefer official latest documentation from Context7 over pre-trained memory to prevent deprecated API hallucinations.

---

# Andrej Karpathy Coding Agent Guidelines (4대 행동 강령)

1. **Think Before Coding (코딩 전 생각하기 — 추측 금지)**
   - **Don't assume. Don't hide confusion. Surface tradeoffs.**
   - 요구사항이 모호할 때 임의로 넘겨짚어 코딩하지 말고, 가정 사항을 명시하거나 질문하라.
   - 여러 해석이 가능할 때는 대안과 장단점을 제시하고, 더 단순한 방법이 있다면 적극적으로 역제안하라.

2. **Simplicity First (단순함 최우선 — 과잉 설계 금지)**
   - **Minimum code that solves the problem. Nothing speculative.**
   - 요청받지 않은 기능, 한 번만 쓰이는 과도한 추상화, 미래를 대비한 불필요한 확장 코드를 작성하지 마라.
   - 단, 파일당 250~300줄 이하 모듈화 원칙은 철저히 준수한다.

3. **Surgical Changes (외과수술식 정밀 수정 — 핀셋 수정)**
   - **Touch only what you must. Clean up only your own mess.**
   - 요청과 무관한 기존 코드, 주석, 디자인, 포맷팅을 건드리지 마라. 수정된 모든 줄은 사용자의 요청과 직접 연결되어야 한다.
   - 본인의 수정으로 인해 더 이상 사용되지 않게 된 변수/import 구문만 스스로 깨끗이 제거하라.

4. **Goal-Driven Execution (목표 및 검증 중심 실행)**
   - **Define success criteria. Loop until verified.**
   - 작업 전후로 명확한 성공/검증 기준(빌드 성공 여부, 연결 경로 무결성, 화면 테스트 시나리오)을 정의하고 확인될 때까지 점검하라.
