###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Divide/recover programs for declared task failures with a plain float arm."""

import typing as t

from aiida.engine import PortModel
from aiida.engine.processes.exit_code import ExitCode, ExitCodeEnum
from aiida.engine.processes.graphs.lowering import graph, task
from aiida.engine.processes.graphs.results import Err, Ok, Result


class DivisionErrors(ExitCodeEnum):
    """Declared failures of the `divide` task."""

    ZERO_DENOMINATOR = ExitCode(410, 'The denominator is zero.', invalidates_cache=True)


class InputErrors(ExitCodeEnum):
    """A second error enum for union contracts."""

    NEGATIVE_NUMERATOR = ExitCode(411, 'The numerator is negative.')


@task
def divide(numerator: float, denominator: float) -> Result[float, DivisionErrors]:
    """Divide, declaring the zero-denominator failure instead of raising it."""
    if denominator == 0:
        return Err(DivisionErrors.ZERO_DENOMINATOR)
    return Ok(numerator / denominator)


@task
def recover_zero(error: DivisionErrors) -> float:
    """Recover from a zero denominator with a neutral fallback."""
    return 0.0


@task
def recover_negative(numerator: float, error: InputErrors) -> float:
    """Recover from a negative numerator with its absolute value."""
    return abs(numerator)


@task
def divide_signed(numerator: float, denominator: float) -> Result[float, DivisionErrors | InputErrors]:
    """Divide, declaring failures from two error enums."""
    if denominator == 0:
        return Err(DivisionErrors.ZERO_DENOMINATOR)
    if numerator < 0:
        return Err(InputErrors.NEGATIVE_NUMERATOR)
    return Ok(numerator / denominator)


@task
def union_fallback(error: DivisionErrors | InputErrors) -> float:
    """Recover from a failure declared by either error enum."""
    return -float(error.value.status)


@task
def stray_ok(value: int) -> int:
    """Return `Ok` without a `Result` annotation."""
    return Ok(value)  # type: ignore[return-value] - misuse: Ok without Result


@task
def legacy_exit(value: int) -> int:
    """Return a bare `ExitCode`, preserved from before `Result` existed."""
    return ExitCode(500, 'Legacy failure.')  # type: ignore[return-value] - legacy: bare ExitCode


@task
def undeclared_error(value: int) -> Result[int, DivisionErrors]:
    """Return an exit code outside the declared enum."""
    return Err(ExitCode(999, 'Not declared'))  # type: ignore[arg-type] - misuse: code outside the enum


@task
def missing_result_wrapper(value: int) -> Result[int, DivisionErrors]:
    """Return bare outputs where a `Result` wrapper is required."""
    return value  # type: ignore[return-value] - misuse: bare Result return


@graph
def ratio(numerator: float, denominator: float) -> float:
    """Divide, routing each declared failure to its own handler."""
    value, error = divide_signed(numerator=numerator, denominator=denominator)
    if error is None:
        selected = value
    elif error is DivisionErrors.ZERO_DENOMINATOR:
        selected = recover_zero(error=error)
    else:
        selected = recover_negative(numerator=numerator, error=error)
    return selected


@graph
def ratio_is_none(numerator: float, denominator: float) -> float:
    """Divide, testing the success arm for `None` instead of the error arm."""
    value, error = divide(numerator=numerator, denominator=denominator)
    if error is None:
        selected = value
    else:
        selected = recover_zero(error=error)
    return selected


@graph
def signed_ratio(numerator: float, denominator: float) -> float:
    """Divide, recovering failures from either error enum."""
    value, error = divide_signed(numerator=numerator, denominator=denominator)
    if error is not None:
        selected = union_fallback(error=error)
    else:
        selected = value
    return selected


@task
def union_total(value: float | None, error: DivisionErrors | InputErrors | None) -> float:
    """Read the successful value, or negate the declared failure status."""
    if value is not None:
        return value
    assert error is not None
    return -float(error.value.status)


@graph
def matched_signed(numerator: float, denominator: float) -> float:
    """Branch on one declared member; the union handler takes the rest."""
    value, error = divide_signed(numerator=numerator, denominator=denominator)
    if error is DivisionErrors.ZERO_DENOMINATOR:
        selected = recover_zero(error=error)
    else:
        selected = union_total(value=value, error=error)
    return selected


@graph
def matched_negated(numerator: float, denominator: float) -> float:
    """Branch away from one declared member; the union handler takes the rest."""
    value, error = divide_signed(numerator=numerator, denominator=denominator)
    if error is not DivisionErrors.ZERO_DENOMINATOR:
        selected = union_total(value=value, error=error)
    else:
        selected = recover_zero(error=error)
    return selected


@graph
def matched_equality(value: float) -> float:
    """Compare an error arm for equality; only `is` tests members."""
    outcome, error = divide(numerator=value, denominator=0.0)
    if error == 1:
        selected = recover_zero(error=error)
    else:
        selected = outcome
    return selected


@graph
def matched_unknown_member(value: float) -> float:
    """Test an unknown name; the compared member must exist."""
    outcome, error = divide(numerator=value, denominator=0.0)
    if error is MISSING:  # noqa: F821 - the unknown member is rejected
        selected = recover_zero(error=error)
    else:
        selected = outcome
    return selected


@graph
def matched_constant(value: float) -> float:
    """Test a constant; only enum members can be matched."""
    outcome, error = divide(numerator=value, denominator=0.0)
    if error is 1:  # noqa: F632 - the constant identity test is rejected
        selected = recover_zero(error=error)
    else:
        selected = outcome
    return selected


class ModelSettings(PortModel):
    """A structured success type, rejected: `Result` arms are single leaves for now."""

    count: int


class NamedPair(t.TypedDict):
    """A named success type, rejected: `Result` arms are single leaves for now."""

    first: int


def bare_result(value: int) -> Result:
    """Reject an unparameterized `Result` at decoration."""
    return value  # type: ignore[return-value] - misuse: bare Result


def nested_result(value: int) -> Result[Result[int, DivisionErrors], DivisionErrors]:
    """Reject a nested `Result` success type at decoration."""
    return Ok(Ok(value))


def structured_success(value: int) -> Result[ModelSettings, DivisionErrors]:
    """Reject a structured success type at decoration."""
    return Ok(ModelSettings(count=value))


def named_success(value: int) -> Result[NamedPair, DivisionErrors]:
    """Reject a named success type at decoration."""
    return Ok({'first': value})


def tuple_success(value: int) -> Result[tuple[int, int], DivisionErrors]:
    """Reject a tuple success type at decoration."""
    return Ok((value, value))
