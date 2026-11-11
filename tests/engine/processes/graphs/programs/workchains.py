###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""`WorkChain` tasks shared by the graph behavior suites."""

from aiida.engine import WorkChain, calcfunction, task
from aiida.engine.processes.exit_code import ExitCode, ExitCodeEnum
from aiida.engine.processes.graphs.lowering import graph
from aiida.orm import Bool, Float, Int


@calcfunction
def add(x: Int, y: Int) -> Int:
    """Add two integers; creating nodes belongs in a `calcfunction`, not the `WorkChain` body."""
    return Int(x.value + y.value)


class AddWorkChain(WorkChain):
    """Add two integers through a single outline step."""

    @classmethod
    def define(cls, spec) -> None:  # type: ignore[override]
        super().define(spec)
        spec.input('x', valid_type=Int)
        spec.input('y', valid_type=Int)
        spec.output('result', valid_type=Int)
        spec.outline(cls.add_numbers)

    def add_numbers(self):
        """Store the sum of the two inputs."""
        self.out('result', add(self.inputs.x, self.inputs.y))


add_chain = task(AddWorkChain)


@graph
def add_single(x: Int, y: Int) -> int:
    """Add two numbers through a single `WorkChain` placement."""
    total = add_chain(x=x, y=y)
    return total.result


@graph
def add_twice(x: Int, y: Int) -> int:
    """Add two numbers through two chained `WorkChain` placements."""
    first = add_chain(x=x, y=y)
    second = add_chain(x=first.result, y=y)
    return second.result


class ChainErrors(ExitCodeEnum):
    """Declared failures of the `flaky_chain` task."""

    UNSTABLE = ExitCode(410, 'Unstable chain.')


@calcfunction
def make_result() -> Float:
    """Store the successful result."""
    return Float(1.0)


class FlakyChain(WorkChain):
    """Fail with a mapped status or store one result, leaving the other arm absent."""

    @classmethod
    def define(cls, spec) -> None:  # type: ignore[override]
        super().define(spec)
        spec.input('fail', valid_type=Bool)
        spec.outputs.dynamic = True
        spec.outline(cls.produce)

    def produce(self):
        """Exit mapped without outputs, or store the result."""
        if self.inputs.fail.value:
            return ChainErrors.UNSTABLE.value
        self.out('result', make_result())


flaky_chain = task(FlakyChain, outputs={'result': Float}, errors=ChainErrors)


@task
def recover_chain(error: ChainErrors) -> str:
    """Recover from the declared chain failure."""
    return f'recovered from {error.value.status}'


@task
def read_result(result: float) -> str:
    """Read the successful result."""
    return f'result:{result}'


@task
def ensure_result(value: float | None, error: ChainErrors | None) -> str:
    """Read both arms; the absent success arm arrives as `None`."""
    if error is not None:
        return f'recovered from {error.value.status}'
    assert value is not None
    return f'result:{value}'


@graph
def recover_unproduced(fail: bool) -> str:
    """Run a chain that may fail mapped without outputs, recovering on its error arm."""
    ran = flaky_chain(fail=fail)
    if ran.error is not None:
        selected = recover_chain(error=ran.error)
    else:
        selected = read_result(result=ran.result)
    return selected


@graph
def read_absent_arm(fail: bool) -> str:
    """Read both arms unconditionally; the absent success arm arrives as `None`."""
    ran = flaky_chain(fail=fail)
    return ensure_result(value=ran.result, error=ran.error)
