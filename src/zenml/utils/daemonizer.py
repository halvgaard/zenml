#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.


import os
import pathlib
import psutil
import signal
from typing import Any, Callable, Optional, TypeVar, Union, overload

CHILD_WAIT_TIMEOUT = 5


def terminate_children(signum: int, frame: Optional[Any] = None) -> None:
    """Terminate all processes that are children of the currently running
    process.

    This function can be used as a signal handler to gracefully terminate
    child processes before the current process exits, e.g.:

    ```python
    import signal
    signal.signal(signal.SIGINT, terminate_children)
    signal.signal(signal.SIGTERM, terminate_children)
    ```

    Args:
        signum: signal to send to the
        frame: signal handler frame
    """
    pid = os.getpid()
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        # could not find parent process id
        return
    children = parent.children(recursive=False)

    for p in children:
        p.terminate()
    _, alive = psutil.wait_procs(children, timeout=CHILD_WAIT_TIMEOUT)
    for p in alive:
        p.kill()
    _, alive = psutil.wait_procs(children, timeout=CHILD_WAIT_TIMEOUT)


F = TypeVar("F", bound=Callable[..., Any])


@overload
def daemonize(
    _func: F,
) -> F:
    """Type annotations for deamonizer decorator in case of no arguments."""
    ...


@overload
def daemonize(
    *, log_file: Optional[str], pid_file: Optional[str]
) -> Callable[[F], F]:
    """Type annotations for deamonizer decorator in case of arguments."""
    ...


def daemonize(
    _func: Optional[F] = None,
    *,
    log_file: Optional[str],
    pid_file: Optional[str]
) -> Union[F, Callable[[F], F]]:
    """Decorator that executes the input function as a daemon process.

    Args:
        _func: decorated function
        log_file: file where stdout and stderr are redirected for the daemon
            process. If not supplied, the daemon will be silenced (i.e. have
            its stdout/stderr redirected to /dev/null).
        pid_file: an optional file where the PID of the daemon process will be
            saved.

    Returns:
        Decorated function that, when called, will detach from the current
        process and continue executing in the background, as a daemon process.
    """

    def inner_decorator(_func: F) -> F:
        def daemon(*args: Any, **kwargs: Any) -> int:
            """Standard daemonization of a process.

            Returns:
                The PID of the daemon process is returned to the parent process
                and a zero value is returned to the child process.
            """
            pid = os.fork()
            if pid:
                with open(pid_file, "w+") as fd_pid:
                    fd_pid.write(str(pid))
                return pid

            os.umask(0o22)
            os.setsid()
            os.closerange(0, 3)

            # Remap all of stdin, stdout and stderr on to
            # the specified log file or /dev/null.

            std_redirect = getattr(os, "devnull", "/dev/null")

            if log_file:
                std_redirect = log_file

            with open(std_redirect, "a+") as log:
                os.dup2(log.fileno(), 0)
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)

                signal.signal(signal.SIGINT, terminate_children)
                signal.signal(signal.SIGTERM, terminate_children)
                _func(*args, **kwargs)

            return 0

        return daemon

    if _func is None:
        return inner_decorator
    else:
        return inner_decorator(_func)
