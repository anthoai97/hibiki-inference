---
status: accepted
---

# Gate correctness before performance claims

Independent pinned fixtures must exist before the clean Mimi and Hibiki paths are accepted: generated tokens match exactly, while decoded PCM uses a measured tolerance. After correctness passes, a 120-second run on the M1 Pro reference machine must keep memory bounded, achieve real-time factor at or below 1.0, and keep warm frame-time p95 at or below 80 ms; cold loading and warmup are measured separately, and the maximum frame time is reported rather than used as a brittle one-spike failure gate.
