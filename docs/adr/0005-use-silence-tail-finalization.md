---
status: accepted
---

# Use explicit silence-tail finalization until EOS is verified

Version one will not claim learned EOS completion because the pinned public runners do not demonstrate it end to end. Finishing pads one non-empty partial source frame, advances exactly six additional silent source frames, exposes only complete target audio frames, discards and counts the final incomplete delayed positions, and makes later finish calls return no duplicate output with an `already_finished` reason; the six-frame policy is deliberately deterministic and is not presented as exact reference behavior.
