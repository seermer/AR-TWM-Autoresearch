from ar_kernel.liveness import Liveness, tree_mark


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(clock, marks, **kw):
    it = iter(marks)
    return Liveness(100, probe_window_s=10, extension_frac=0.25, signals=[lambda: next(it)],
                    clock=clock, **kw)


def test_before_the_soft_deadline_nothing_is_probed():
    c = Clock()
    lv = make(c, [])                                   # a signal call would raise StopIteration
    c.t = 99
    assert lv.expired() is None


def test_progress_in_the_probe_window_extends_by_a_quarter_of_soft():
    c = Clock()
    lv = make(c, [1, 2, 2, 2])
    c.t = 100
    assert lv.expired() is None                        # probe starts, baseline 1
    c.t = 105
    assert lv.expired() is None and lv.extensions == 1 # changed -> deadline 105 + 25
    c.t = 129
    assert lv.expired() is None                        # before the new deadline: not probed
    c.t = 130
    assert lv.expired() is None                        # new probe, baseline 2
    c.t = 140
    assert "no sign of progress" in lv.expired()


def test_signals_are_throttled_during_the_probe_window():
    c, calls = Clock(), []
    lv = Liveness(100, probe_window_s=60, extension_frac=0.25, probe_every_s=30, clock=c,
                  signals=[lambda: calls.append(c.t) or 0])
    for t in range(100, 161):
        c.t = t
        lv.expired()
    assert calls == [100, 130, 160] and lv.reason                 # the window end is still checked


def test_hard_cap_ends_regardless():
    c = Clock()
    lv = make(c, [1, 2, 3, 4, 5], hard_s=150)
    c.t = 150
    assert "hard cap" in lv.expired() and lv.reason


def test_tree_mark_sees_new_and_grown_files(tmp_path):
    a = tree_mark(tmp_path)
    (tmp_path / "x").write_text("1")
    b = tree_mark(tmp_path)
    (tmp_path / "x").write_text("12")
    assert a != b != tree_mark(tmp_path)
    assert tree_mark(tmp_path / "missing") == (0, 0, 0)
