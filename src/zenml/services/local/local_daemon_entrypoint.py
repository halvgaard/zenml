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

import click

import os

from zenml.integrations.registry import integration_registry
from zenml.logger import get_logger
from zenml.services import ServiceRegistry


logger = get_logger(__name__)


try:
    from os import closerange
except ImportError:

    def closerange(fd_low, fd_high):
        # Iterate through and close all file descriptors.
        for fd in range(fd_low, fd_high):
            try:
                os.close(fd)
            except OSError:  # ERROR, fd wasn't open to begin with (ignored)
                pass


def daemonize(name: str, suppress_logging: bool, log_file: str):
    """Standard daemonization of a process."""

    # os.setpgrp()
    # os.setsid()

    # if os.fork():
    #     os._exit(0)
    # os.setsid()

    # if os.fork():
    #     os._exit(0)

    # os.umask(0o22)

    # In both the following any file descriptors above stdin
    # stdout and stderr are left untouched. The inheritance
    # option simply allows one to have output go to a file
    # specified by way of shell redirection when not wanting
    # to use --error-log option.

    # procname.setprocname(name)

    if suppress_logging or log_file:
        # Remap all of stdin, stdout and stderr on to
        # the specified log file or /dev/null.

        devnull = getattr(os, "devnull", "/dev/null")
        log_file = log_file or devnull

        closerange(0, 3)

        fd_log = os.open(log_file, os.O_RDWR)

        if fd_log != 0:
            os.dup2(fd_log, 0)

        os.dup2(fd_log, 1)
        os.dup2(fd_log, 2)


@click.command()
@click.option("--config-file", required=True, type=click.Path(exists=True))
@click.option("--suppress-logging", required=False, default=False, is_flag=True)
@click.option("--log-file", required=False, type=click.Path())
def run(config_file: str, suppress_logging: bool, log_file: str) -> None:
    # same code as materializer
    logger.info("Loading service daemon configuration from %s", config_file)
    with open(config_file, "r") as f:
        config = f.read()

    integration_registry.activate_integrations()

    service = ServiceRegistry().load_service_from_json(config)

    daemonize(service.config.name, suppress_logging, log_file)
    service.run()


if __name__ == "__main__":
    run()
