import timeit


class TimeIt:
    def __init__(self, tag, frames=None):
        self.tag = tag
        self.frames = frames

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.elapsed_secs = timeit.default_timer() - self.start
        msg = self.tag + (": Elapsed time=%.2fs" % self.elapsed_secs)
        if self.frames:
            msg += ", FPS=%.2e" % (self.frames / self.elapsed_secs)
        print(msg)
