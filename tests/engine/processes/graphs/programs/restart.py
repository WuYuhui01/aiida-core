###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Restart-loop programs: a fallible step retried until it succeeds."""

from aiida.engine.processes.exit_code import ExitCode, ExitCodeEnum
from aiida.engine.processes.graphs.lowering import graph, task
from aiida.engine.processes.graphs.results import Err, Ok, RestartErrors, Result
from aiida.orm import StructureData


class StepErrors(ExitCodeEnum):
    """Declared failures of the `run_step` task."""

    UNSTABLE = ExitCode(410, 'Unstable step.')


@task
def zero() -> int:
    """Seed the attempt counter."""
    return 0


@task
def step(attempt: int) -> int:
    """Count one further attempt."""
    return attempt + 1


@task
def should_restart(error: StepErrors | None, attempt: int, max_attempts: int) -> bool:
    """Decide whether another attempt remains after a declared failure."""
    return error is not None and attempt < max_attempts


@task
def run_step(structure: StructureData, attempt: int) -> Result[float, StepErrors]:
    """Fail the first two attempts, then succeed."""
    if attempt < 2:
        return Err(StepErrors.UNSTABLE)
    return Ok(1.5)


@task
def ensure(value: float | None, error: StepErrors | None) -> float:
    """Read the recovered value, failing loudly when none remains."""
    if error is not None or value is None:
        raise ValueError('the step never succeeded.')
    return value


@task
def flip_step(attempt: int) -> Result[float, StepErrors]:
    """Succeed around one declared failure, so the loop rebinds both arms."""
    if attempt == 1:
        return Err(StepErrors.UNSTABLE)
    return Ok(float(attempt))


@task
def keep_going(attempt: int) -> bool:
    """Iterate twice."""
    return attempt < 2


@task
def mark_absent() -> str:
    """Record that the previous iteration left no value arm."""
    return 'absent'


@task
def mark_present(value: float) -> str:
    """Record that the previous iteration left a value arm."""
    return f'present:{value}'


@task
def check_recovered(marker: str, value: float | None, error: StepErrors | None) -> str:
    """Return the marker, requiring the final iteration to have succeeded."""
    if error is not None or value is None:
        raise ValueError('the final iteration never succeeded.')
    return marker


@graph
def absent_arm_resets() -> str:
    """Rebind a `Result` in a loop; an absent arm reads as `None`, not stale state."""
    attempt = zero()
    marker = mark_absent()
    value, error = flip_step(attempt=attempt)
    retry = keep_going(attempt=attempt)
    while retry:
        attempt = step(attempt=attempt)
        if value is None:
            marker = mark_absent()
        else:
            marker = mark_present(value=value)
        value, error = flip_step(attempt=attempt)
        retry = keep_going(attempt=attempt)
    return check_recovered(marker=marker, value=value, error=error)


@task
def report_unhandled(value: float | None, error: StepErrors | None) -> Result[float, RestartErrors]:
    """Translate an unrecovered failure into the shared machinery vocabulary."""
    if error is not None or value is None:
        return Err(RestartErrors.UNHANDLED_FAILURE)
    return Ok(value)


@graph
def restartable(structure: StructureData, max_attempts: int) -> float:
    """Retry a fallible step until it succeeds or attempts run out."""
    attempt = zero()
    value, error = run_step(structure=structure, attempt=attempt)
    retry = should_restart(error=error, attempt=attempt, max_attempts=max_attempts)
    while retry:
        attempt = step(attempt=attempt)
        value, error = run_step(structure=structure, attempt=attempt)
        retry = should_restart(error=error, attempt=attempt, max_attempts=max_attempts)
    return ensure(value=value, error=error)
