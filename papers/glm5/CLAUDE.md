# GLM-5 Paper Project

Paper: "GLM-5: from Vibe Coding to Agentic Engineering" (Zhipu AI & Tsinghua University, Feb 2026)

## Quick Reference

- Run: `uv run glm5`
- Test: `uv run pytest`
- Lint: `uv run ruff check .`
- Add dep: `uv add <package>`

## Key Paper Topics

- 744B MoE model (40B active), 256 experts, 80 layers
- DeepSeek Sparse Attention (DSA) for efficient 200K context
- Multi-latent Attention (MLA) with Muon Split
- Multi-token Prediction (MTP) with parameter sharing (3 layers)
- Async RL infrastructure ("slime" framework)
- 3-stage RL: Reasoning RL → Agentic RL → General RL
- On-Policy Cross-Stage Distillation to prevent catastrophic forgetting
- TITO (Token-in-Token-out) gateway for async RL stability
- Hierarchical Context Management for search agents
- SWE environment scaling (10k+ verifiable environments)
- Terminal environment synthesis from seed data and web corpus

## Project Structure

```
papers/glm5.pdf    # Source PDF
src/glm5/          # Main package
tests/             # Tests
```
