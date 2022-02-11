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
import pathlib


def daemonize(log_file: str):
    """Standard daemonization of a process."""

    # os.setpgrp()
    # os.setsid()

    # if os.fork():
    #     os._exit(0)
    # os.setsid()

    # if os.fork():
    #     os._exit(0)

    # os.umask(0o22)

    # procname.setprocname(name)

    # Remap all of stdin, stdout and stderr on to
    # the specified log file or /dev/null.

    devnull = getattr(os, "devnull", "/dev/null")

    if log_file:
        pathlib.Path(log_file).touch()
    log_file = log_file or devnull

    os.closerange(0, 3)

    # TODO: close logfile on exit
    fd_log = os.open(log_file, os.O_RDWR)

    if fd_log != 0:
        os.dup2(fd_log, 0)

    os.dup2(fd_log, 1)
    os.dup2(fd_log, 2)


def launch_service(service_config_file: str):
    """Instantiate and launch a ZenML local service from its
    configuration file.
    """

    # doing zenml imports here to avoid polluting the stdout/sterr
    # with messages before daemonization is complete
    from zenml.integrations.registry import integration_registry
    from zenml.logger import get_logger
    from zenml.services import ServiceRegistry

    logger = get_logger(__name__)

    logger.info(
        "Loading service daemon configuration from %s", service_config_file
    )
    with open(service_config_file, "r") as f:
        config = f.read()

    integration_registry.activate_integrations()

    logger.debug("Running service daemon with configuration:\n %s", config)
    service = ServiceRegistry().load_service_from_json(config)
    service.run()


@click.command()
@click.option("--config-file", required=True, type=click.Path(exists=True))
@click.option("--log-file", required=False, type=click.Path())
def run(config_file: str, log_file: str) -> None:

    daemonize(log_file)
    launch_service(config_file)


if __name__ == "__main__":
    run()
