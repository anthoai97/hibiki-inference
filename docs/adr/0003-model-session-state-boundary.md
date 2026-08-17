---
status: accepted
---

# Keep mutable inference state in a session

A loaded model owns immutable architecture and parameters, while each inference session owns every mutable codec, Transformer, scheduler, tokenizer, random, counter, metric, and lifecycle state. Version one permits only one active session per loaded model, but preserving this boundary prevents accidental state sharing and allows later concurrency or batching without redesigning model ownership.
