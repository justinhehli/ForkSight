import os
import subprocess
import sys
import time


def is_pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def terminate_process_tree(pid: int | None, timeout: float = 5.0) -> None:
    """Best-effort stop of a process and all its descendants 

    Tries an unforceful stop first, then escalates to a forceful kill if the tree 
    is still alive after `timeout` seconds
    """
    if not is_pid_running(pid):
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T"],
                       capture_output=True)
        deadline = time.monotonic() + timeout
        while is_pid_running(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        if is_pid_running(pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True)
        return

    import signal
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + timeout
    while is_pid_running(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    if is_pid_running(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
