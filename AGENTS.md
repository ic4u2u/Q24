# Context7 MCP Auto-Invocation Rule

- Always use the `context7` MCP tools (`resolve-library-id` and `get-library-docs` / `query-docs`) automatically whenever writing, debugging, or upgrading code involving external libraries, frameworks, SDKs, or APIs (e.g., OpenCV, PyTorch, Ultralytics YOLO, Supervision, FastAPI, Pydantic, React, Next.js, Tailwind CSS, Supabase, Firebase, Flutter) — without waiting for the user to explicitly type `"use context7"`.
- Workflow for automatic invocation:
  1. Call `resolve-library-id` with the target library/package name to obtain its official Context7-compatible library ID.
  2. Call `get-library-docs` (or `query-docs`) with that library ID and the specific topic/API being implemented or debugged.
- Prefer official latest documentation from Context7 over pre-trained memory to prevent deprecated API hallucinations.
