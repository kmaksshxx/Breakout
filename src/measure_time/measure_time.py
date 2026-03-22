from contextlib import contextmanager
import time
from collections import defaultdict
from typing import Optional

class SectionTimer:
    def __init__(self, title=None):
        self.t = defaultdict(float)
        self.n = defaultdict(int)
        self.title = title

    def reset(self, title=None):
        self.t.clear()
        self.n.clear()
        self.title = title

    def add(self, key, dt):
        self.t[key] += time.perf_counter() - dt
        self.n[key] += 1

    def _report(self):
        if self.title:
            print(f"======== {str(self.title)} ========")
        total = sum(self.t.values())
        for k in sorted(self.t, key=lambda x: -self.t[x]):
            avg = self.t[k] / max(1, self.n[k])
            print(f"{k:20s}: {self.t[k]:8.3f}s | avg {avg * 1e6:7.1f} µs")
        print(f"{'TOTAL':20s}: {total:8.3f}s")

    def _format_time(self, t):
        if t < 1e-3:
            return f"{t * 1e6:8.1f} µs"
        elif t < 1:
            return f"{t * 1e3:8.2f} ms"
        elif t < 60:
            return f"{t:8.3f} s"
        else:
            m = int(t // 60)
            s = t - 60 * m
            return f"{m:2d}m {s:5.2f}s"

    def report(self):
        if self.title:
            print(f"======== {self.title} ========")

        total = sum(self.t.values())

        print(f"{'section':20s} {'calls':>8s} {'total':>12s} {'avg':>12s} {'%':>8s}")
        print("-" * 64)

        for k in sorted(self.t, key=lambda x: -self.t[x]):
            calls = self.n[k]
            total_t = self.t[k]
            avg = total_t / max(1, calls)
            pct = (total_t / total * 100) if total > 0 else 0

            print(
                f"{k:20s} "
                f"{calls:8d} "
                f"{self._format_time(total_t):>12s} "
                f"{self._format_time(avg):>12s} "
                f"{pct:7.2f}%"
            )

        print("-" * 64)
        print(f"{'TOTAL':20s} {'':8s} {self._format_time(total):>12s}")

timer = SectionTimer()

@contextmanager
def timed(timer: Optional[SectionTimer], label: str):
    if timer is None:
        yield
        return

    t0 = time.perf_counter()
    try:
        yield
    finally:
        timer.add(label, t0)
        