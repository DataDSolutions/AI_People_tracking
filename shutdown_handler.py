import os
import sys
import signal
import threading


class ShutdownManager:

    def __init__(self, force_exit_after_signals=2):
        self._shutdown_event = threading.Event()
        self._signal_count = 0
        self._lock = threading.Lock()
        self._cleanup_steps = []
        self._cleanup_done = False
        self._force_exit_after = int(force_exit_after_signals)
        self._installed = False
        self._original_handlers = {}
        self._triggered_by_signal = False

    @property
    def is_shutting_down(self):
        return self._shutdown_event.is_set()

    @property
    def triggered_by_signal(self):
        return self._triggered_by_signal

    def wait(self, timeout=None):
        return self._shutdown_event.wait(timeout)

    def begin_shutdown(self, reason="programmatic"):
        if not self._shutdown_event.is_set():
            sys.stderr.write(f"[SHUTDOWN] shutdown requested ({reason})\n")
            sys.stderr.flush()
        self._shutdown_event.set()

    def register(self, name, fn):
        with self._lock:
            if self._cleanup_done:
                try:
                    fn()
                except Exception as exc:
                    sys.stderr.write(f"[SHUTDOWN] late cleanup step '{name}' failed: {exc}\n")
                return
            self._cleanup_steps.append((name, fn))

    def install(self, signals=(signal.SIGINT, signal.SIGTERM)):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("ShutdownManager.install() must be called from the main thread")
        for sig in signals:
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        self._installed = True
        return self

    def _handle_signal(self, signum, frame):
        with self._lock:
            self._signal_count += 1
            count = self._signal_count
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = str(signum)
        if count == 1:
            self._triggered_by_signal = True
            sys.stderr.write(
                f"\n[SHUTDOWN] {name} received - shutting down gracefully "
                f"(press Ctrl+C again to force-quit)...\n")
            sys.stderr.flush()
            self._shutdown_event.set()
        elif count >= self._force_exit_after:
            sys.stderr.write(f"\n[SHUTDOWN] {name} received again - forcing immediate exit.\n")
            sys.stderr.flush()
            os._exit(1)

    def run_cleanup(self):
        with self._lock:
            if self._cleanup_done:
                return
            self._cleanup_done = True
            steps = list(self._cleanup_steps)
        for name, fn in steps:
            try:
                fn()
            except Exception as exc:
                sys.stderr.write(f"[SHUTDOWN] cleanup step '{name}' failed: {exc}\n")
                sys.stderr.flush()
        self.restore_signal_handlers()

    def restore_signal_handlers(self):
        if not self._installed:
            return
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
        self._installed = False

    def spawn_watcher(self, on_signal, name="shutdown-watcher", daemon=True):
        def _run():
            self.wait()
            try:
                on_signal()
            except Exception as exc:
                sys.stderr.write(f"[SHUTDOWN] watcher callback failed: {exc}\n")
                sys.stderr.flush()
        t = threading.Thread(target=_run, name=name, daemon=daemon)
        t.start()
        return t
