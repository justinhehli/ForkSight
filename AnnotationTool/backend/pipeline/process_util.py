import os
import time


def is_pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def terminate_process_tree(pid: int | None, timeout: float = 5.0) -> None:
    """Best-effort stop of a process and all its descendants

    `pid` must be the PID of a process group leader (e.g. spawned with
    subprocess.Popen(..., start_new_session=True)) so that signalling its
    process group (via killpg) reaches its descendants too - a plain kill(pid)
    would only stop that one process and leave e.g. its own child subprocesses
    running as orphans.

    Tries an unforceful stop first, then escalates to a forceful kill if the tree
    is still alive after `timeout` seconds
    """
    if not is_pid_running(pid):
        return

    import signal
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + timeout
    while is_pid_running(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    if is_pid_running(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
