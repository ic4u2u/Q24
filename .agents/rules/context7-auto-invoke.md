---
trigger: always_on
description: Automatically invoke Context7 MCP tools (resolve-library-id, get-library-docs) whenever working with external libraries, frameworks, or SDKs without requiring the user to type "use context7".
---

# Context7 MCP Auto-Invocation Rule

- Always use the `context7` MCP tools (`resolve-library-id` and `get-library-docs` / `query-docs`) automatically whenever writing, debugging, or upgrading code involving external libraries, frameworks, SDKs, or APIs (e.g., OpenCV, PyTorch, Ultralytics YOLO, Supervision, FastAPI, Pydantic, React, Next.js, Tailwind CSS, Supabase, Firebase, Flutter) — without waiting for the user to explicitly type `"use context7"`.
- Prefer official latest documentation from Context7 over pre-trained memory to prevent deprecated API hallucinations.
