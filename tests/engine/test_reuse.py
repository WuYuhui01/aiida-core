###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Tests for engine-level reuse decisions and controlled cache selection."""

from __future__ import annotations

import uuid

import pytest

from aiida import orm
from aiida.engine import (
    Process,
    WorkChain,
    ReuseConfiguration,
    ReuseDecisionError,
    ReuseMaterialization,
    ReusePolicy,
    ReuseReason,
    get_reuse_decision_records,
    make_reuse_decision,
    store_reuse_decision,
)
from aiida.manage import enable_caching

pytestmark = pytest.mark.usefixtures('aiida_profile_clean')


def _request_id() -> str:
    return f'req-{uuid.uuid4()}'


def _candidate(result_uuid: str, *, qualified: bool = True) -> dict:
    return {'uuid': result_uuid, 'qualified': qualified, 'fingerprint': 'candidate-fingerprint'}


def _source_process() -> orm.CalculationNode:
    """Create a stored process node usable as a candidate source in unit tests."""
    return orm.CalculationNode().store()


def _make_configuration(
    *,
    policy: ReusePolicy | str,
    candidates=(),
    original_seed: int = 41,
    source_resolver=None,
    decision_function=None,
    seed_factory=None,
    materialization=ReuseMaterialization.CACHE_PROCESS,
):
    kwargs = {}
    if source_resolver is not None:
        kwargs['resolve_candidate_source'] = source_resolver
    if decision_function is not None:
        kwargs['decision_function'] = decision_function
    if seed_factory is not None:
        kwargs['seed_factory'] = seed_factory
    return ReuseConfiguration(
        request_id=_request_id(),
        reuse_policy=policy,
        original_seed=original_seed,
        input_rewriter=lambda inputs, decision: inputs,
        candidates=candidates,
        materialization=materialization,
        **kwargs,
    )


def test_allow_reuse_selects_first_qualified_candidate():
    """The default policy selects the first qualified candidate and preserves its source."""
    source = _source_process()
    candidate_uuid = str(uuid.uuid4())
    configuration = _make_configuration(
        policy=ReusePolicy.ALLOW_REUSE,
        candidates=[_candidate(candidate_uuid)],
        source_resolver=lambda _: source,
    )

    decision = make_reuse_decision(configuration, str(uuid.uuid4()), timestamp='2026-11-11T00:00:00Z')

    assert decision.action.value == 'reuse'
    assert decision.reason_code is ReuseReason.REUSE_QUALIFIED
    assert decision.selected_candidate_uuid == candidate_uuid
    assert decision.selected_process_uuid == source.uuid
    assert decision.effective_seed == 41
    assert decision.uses_cache is True
    assert decision.record['final_decision']['action'] == 'reuse'


def test_allow_reuse_without_qualified_candidate_reruns():
    """Without a qualified candidate the default policy falls back to a controlled rerun."""
    configuration = _make_configuration(
        policy=ReusePolicy.ALLOW_REUSE,
        candidates=[_candidate(str(uuid.uuid4()), qualified=False)],
    )

    decision = make_reuse_decision(configuration, str(uuid.uuid4()))

    assert decision.action.value == 'rerun'
    assert decision.reason_code is ReuseReason.ALLOW_REUSE_NO_CANDIDATE
    assert decision.selected_candidate_uuid is None
    assert decision.effective_seed == 41


def test_force_rerun_ignores_qualified_candidates():
    """Force rerun must not select a candidate even when one is available."""
    configuration = _make_configuration(
        policy=ReusePolicy.FORCE_RERUN,
        candidates=[_candidate(str(uuid.uuid4()))],
    )

    decision = make_reuse_decision(configuration, str(uuid.uuid4()))

    assert decision.action.value == 'rerun'
    assert decision.reason_code is ReuseReason.FORCE_RERUN
    assert decision.selected_candidate_uuid is None
    assert decision.selected_process_uuid is None


def test_new_sample_uses_a_different_seed():
    """New sample keeps the policy action while changing the effective seed."""
    configuration = _make_configuration(
        policy=ReusePolicy.NEW_SAMPLE,
        candidates=[_candidate(str(uuid.uuid4()))],
        seed_factory=lambda original_seed: original_seed + 1,
    )

    decision = make_reuse_decision(configuration, str(uuid.uuid4()))

    assert decision.action.value == 'new_sample'
    assert decision.reason_code is ReuseReason.NEW_SAMPLE
    assert decision.selected_candidate_uuid is None
    assert decision.effective_seed == 42


def test_custom_decision_cannot_override_force_rerun_policy():
    """A custom decision callback cannot turn force_rerun into reuse."""
    configuration = _make_configuration(
        policy=ReusePolicy.FORCE_RERUN,
        candidates=[_candidate(str(uuid.uuid4()))],
        decision_function=lambda _: {
            'action': 'reuse',
            'selected_candidate_uuid': None,
            'effective_seed': 41,
            'reason_text': 'custom reuse',
        },
    )

    with pytest.raises(ReuseDecisionError, match='does not allow action reuse'):
        make_reuse_decision(configuration, str(uuid.uuid4()))


def test_decision_records_are_replaced_and_copied():
    """Decision records are queryable, replaceable and returned as detached copies."""
    source = _source_process()
    configuration = _make_configuration(
        policy=ReusePolicy.ALLOW_REUSE,
        candidates=[_candidate(str(uuid.uuid4()))],
        source_resolver=lambda _: source,
    )
    decision = make_reuse_decision(configuration, str(uuid.uuid4()))
    target = orm.CalculationNode().store()

    store_reuse_decision(target, decision.record)
    store_reuse_decision(target, decision.record)
    records = get_reuse_decision_records(target)

    assert len(records) == 1
    assert records[0]['request_id'] == configuration.request_id
    records[0]['final_decision']['action'] = 'tampered'
    assert get_reuse_decision_records(target)[0]['final_decision']['action'] == 'reuse'


def test_invalid_decision_record_is_rejected():
    """A record that violates policy invariants cannot be persisted."""
    target = orm.CalculationNode().store()
    invalid_record = {
        'request_id': _request_id(),
        'process_uuid': target.uuid,
        'reuse_policy': 'force_rerun',
        'candidate_compare_list': [],
        'final_decision': {
            'action': 'reuse',
            'selected_candidate_uuid': str(uuid.uuid4()),
            'effective_seed': 41,
        },
        'reason': {'reason_code': 'R03', 'reason_text': 'invalid override'},
        'timestamp': '2026-11-11T00:00:00Z',
    }

    with pytest.raises(ReuseDecisionError):
        store_reuse_decision(target, invalid_record)


class ControlledReuseProcess(Process):
    """A cacheable test process whose reuse policy is configured by the test."""

    _node_class = orm.CalcJobNode
    policy = ReusePolicy.FORCE_RERUN
    candidates = ()
    decision_function = None
    seed_factory = None

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('seed', valid_type=orm.Int)
        spec.output('result', valid_type=orm.Int)

    def get_reuse_configuration(self):
        kwargs = {}
        if self.decision_function is not None:
            kwargs['decision_function'] = self.decision_function
        if self.seed_factory is not None:
            kwargs['seed_factory'] = self.seed_factory
        return ReuseConfiguration(
            request_id=_request_id(),
            reuse_policy=self.policy,
            original_seed=self.inputs.seed.value,
            input_rewriter=self._rewrite_inputs,
            candidates=self.candidates,
            **kwargs,
        )

    @staticmethod
    def _rewrite_inputs(inputs, decision):
        rewritten = dict(inputs)
        rewritten['seed'] = orm.Int(decision.effective_seed)
        return rewritten

    async def run(self):
        self.out('result', orm.Int(self.inputs.seed.value).store())


class ControlledConsumeCandidateWorkChain(WorkChain):
    """A WorkChainNode consumer that materializes a CalcJob result through an input rewrite."""

    _node_class = orm.WorkChainNode
    policy = ReusePolicy.FORCE_RERUN
    candidates = ()
    candidate_result = None

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('seed', valid_type=orm.Int)
        spec.input('candidate_result', valid_type=orm.Int, required=False)
        spec.output('result', valid_type=orm.Int)
        spec.outline(cls.consume)

    def get_reuse_configuration(self):
        return ReuseConfiguration(
            request_id=_request_id(),
            reuse_policy=self.policy,
            original_seed=self.inputs.seed.value,
            input_rewriter=self._rewrite_inputs,
            candidates=self.candidates,
            materialization=ReuseMaterialization.CONSUME_CANDIDATE,
        )

    @staticmethod
    def _rewrite_inputs(inputs, decision):
        rewritten = dict(inputs)
        if decision.action.value == 'reuse':
            rewritten['candidate_result'] = orm.load_node(decision.selected_candidate_uuid)
        else:
            rewritten.pop('candidate_result', None)
        return rewritten

    def consume(self):
        self.out('result', self.inputs.candidate_result)


class NativeCacheProcess(Process):
    """A cacheable process without a reuse configuration."""

    _node_class = orm.CalcJobNode

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('value', valid_type=orm.Int)
        spec.output('result', valid_type=orm.Int)

    async def run(self):
        self.out('result', orm.Int(self.inputs.value.value).store())


def _register_process(entry_points):
    entry_points.add(ControlledReuseProcess, 'aiida.workflows:core.controlled-reuse')
    entry_points.add(ControlledConsumeCandidateWorkChain, 'aiida.workflows:core.controlled-consume-candidate')


def _configure_process(*, policy, candidates=(), decision_function=None, seed_factory=None):
    ControlledReuseProcess.policy = policy
    ControlledReuseProcess.candidates = candidates
    ControlledReuseProcess.decision_function = (
        staticmethod(decision_function) if decision_function is not None else None
    )
    ControlledReuseProcess.seed_factory = staticmethod(seed_factory) if seed_factory is not None else None


def _run_source(runner, *, seed=41):
    """Run one real source node with a non-cache reuse decision."""
    _configure_process(policy=ReusePolicy.FORCE_RERUN)
    _, node = runner.run_get_node(ControlledReuseProcess, seed=orm.Int(seed))
    return node


def test_workchain_reuse_hits_only_selected_source(runner, entry_points):
    """An allow_reuse decision uses the selected source and records R01."""
    _register_process(entry_points)
    source = _run_source(runner)
    source_result = source.outputs.result

    _configure_process(
        policy=ReusePolicy.ALLOW_REUSE,
        candidates=[_candidate(source_result.uuid)],
    )
    _, target = runner.run_get_node(ControlledReuseProcess, seed=orm.Int(41))

    assert target.base.caching.get_cache_source() == source.uuid
    assert target.outputs.result.value == source_result.value
    records = get_reuse_decision_records(target)
    assert len(records) == 1
    assert records[0]['reason']['reason_code'] == 'R01'
    assert records[0]['final_decision']['action'] == 'reuse'


def test_consume_candidate_does_not_use_native_process_cache(runner, entry_points):
    """A reuse decision can consume a result without a native process cache hit."""
    _register_process(entry_points)
    source = _run_source(runner)
    source_result = source.outputs.result

    ControlledConsumeCandidateWorkChain.policy = ReusePolicy.ALLOW_REUSE
    ControlledConsumeCandidateWorkChain.candidates = [_candidate(source_result.uuid)]
    with enable_caching():
        _, target = runner.run_get_node(
            ControlledConsumeCandidateWorkChain,
            seed=orm.Int(41),
        )

    assert isinstance(target, orm.WorkChainNode)
    assert target.base.caching.get_cache_source() is None
    assert target.outputs.result.uuid == source_result.uuid
    records = get_reuse_decision_records(target)
    assert len(records) == 1
    assert records[0]['reason']['reason_code'] == 'R01'
    assert records[0]['final_decision']['action'] == 'reuse'


def test_force_rerun_does_not_fall_back_to_native_cache(runner, entry_points):
    """A force_rerun decision must create a new process node despite an existing match."""
    _register_process(entry_points)
    source = _run_source(runner)

    _configure_process(
        policy=ReusePolicy.FORCE_RERUN,
        candidates=[_candidate(source.outputs.result.uuid)],
    )
    with enable_caching():
        _, target = runner.run_get_node(ControlledReuseProcess, seed=orm.Int(41))

    assert target.uuid != source.uuid
    assert target.base.caching.get_cache_source() is None
    assert get_reuse_decision_records(target)[0]['reason']['reason_code'] == 'R03'


def test_new_sample_changes_seed_and_does_not_cache(runner, entry_points):
    """A new_sample decision rewrites the seed and starts a fresh calculation."""
    _register_process(entry_points)
    source = _run_source(runner)

    _configure_process(
        policy=ReusePolicy.NEW_SAMPLE,
        candidates=[_candidate(source.outputs.result.uuid)],
        seed_factory=lambda original_seed: original_seed + 1,
    )
    with enable_caching():
        _, target = runner.run_get_node(ControlledReuseProcess, seed=orm.Int(41))

    assert target.inputs.seed.value == 42
    assert target.base.caching.get_cache_source() is None
    assert get_reuse_decision_records(target)[0]['reason']['reason_code'] == 'R04'


def test_allow_reuse_without_candidate_does_not_cache(runner, entry_points):
    """The R02 fallback is a rerun, not an implicit native cache hit."""
    _register_process(entry_points)
    source = _run_source(runner)

    _configure_process(policy=ReusePolicy.ALLOW_REUSE)
    with enable_caching():
        _, target = runner.run_get_node(ControlledReuseProcess, seed=orm.Int(41))

    assert target.uuid != source.uuid
    assert target.base.caching.get_cache_source() is None
    assert get_reuse_decision_records(target)[0]['reason']['reason_code'] == 'R02'


def test_reuse_rejects_mismatched_rewritten_inputs(runner, entry_points):
    """A selected source mismatch must not fall back to another matching cache."""
    _register_process(entry_points)
    source = _run_source(runner, seed=41)
    _run_source(runner, seed=42)

    _configure_process(
        policy=ReusePolicy.ALLOW_REUSE,
        candidates=[_candidate(source.outputs.result.uuid)],
    )
    with pytest.raises(ReuseDecisionError, match='does not match the rewritten process inputs'):
        runner.run_get_node(ControlledReuseProcess, seed=orm.Int(42))


def test_reuse_rejects_cache_when_disable_cache_is_set(runner, entry_points):
    """A reuse decision cannot bypass metadata.disable_cache."""
    _register_process(entry_points)
    source = _run_source(runner)

    _configure_process(
        policy=ReusePolicy.ALLOW_REUSE,
        candidates=[_candidate(source.outputs.result.uuid)],
    )
    with pytest.raises(ReuseDecisionError, match='disable_cache'):
        runner.run_get_node(
            ControlledReuseProcess,
            seed=orm.Int(41),
            metadata={'disable_cache': True},
        )


def test_native_caching_is_unchanged_without_reuse_configuration(runner, entry_points):
    """Processes without a reuse configuration keep AiiDA's native cache behavior."""
    entry_points.add(NativeCacheProcess, 'aiida.workflows:core.native-cache')
    _, source = runner.run_get_node(NativeCacheProcess, value=orm.Int(7))
    with enable_caching():
        _, target = runner.run_get_node(NativeCacheProcess, value=orm.Int(7))

    assert target.base.caching.get_cache_source() == source.uuid
    assert get_reuse_decision_records(source) == []
    assert get_reuse_decision_records(target) == []
