###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Generic reuse-policy decisions for AiiDA processes.

The engine decides whether a process may reuse an existing result, must rerun
with its original inputs, or must run a new sample. Domain-specific plugins
provide candidates, seed handling, input rewriting, and candidate resolution
through :class:`ReuseConfiguration`. The engine owns policy enforcement,
cache-source selection, and persisted decision records.
"""

from __future__ import annotations

import copy
import json
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

DECISION_RECORDS_EXTRA_KEY = 'ai4s_decision_records'
MAX_SEED = 2**32 - 1

_REQUEST_ID_PATTERN = re.compile(
    r'^req-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
    r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
)
_TIMESTAMP_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')


class ReuseDecisionError(ValueError):
    """Raised when a reuse configuration, decision, or record is invalid."""


class ReusePolicy(str, Enum):
    """Policy selected by the user for a process request."""

    ALLOW_REUSE = 'allow_reuse'
    FORCE_RERUN = 'force_rerun'
    NEW_SAMPLE = 'new_sample'


class ReuseAction(str, Enum):
    """Action selected by the reuse decision."""

    REUSE = 'reuse'
    RERUN = 'rerun'
    NEW_SAMPLE = 'new_sample'


class ReuseMaterialization(str, Enum):
    """How a reuse decision is materialized."""

    CACHE_PROCESS = 'cache_process'
    CONSUME_CANDIDATE = 'consume_candidate'


class ReuseReason(str, Enum):
    """Reason codes defined by agreement 5."""

    REUSE_QUALIFIED = 'R01'
    ALLOW_REUSE_NO_CANDIDATE = 'R02'
    FORCE_RERUN = 'R03'
    NEW_SAMPLE = 'R04'
    CUSTOM_DECISION = 'R05'


_ALLOWED_ACTIONS = {
    ReusePolicy.ALLOW_REUSE: frozenset({ReuseAction.REUSE, ReuseAction.RERUN}),
    ReusePolicy.FORCE_RERUN: frozenset({ReuseAction.RERUN}),
    ReusePolicy.NEW_SAMPLE: frozenset({ReuseAction.NEW_SAMPLE}),
}

__all__ = (
    'DECISION_RECORDS_EXTRA_KEY',
    'MAX_SEED',
    'ReuseAction',
    'ReuseCandidate',
    'ReuseConfiguration',
    'ReuseDecision',
    'ReuseDecisionError',
    'ReuseMaterialization',
    'ReusePolicy',
    'ReuseReason',
    'default_candidate_source',
    'default_seed_factory',
    'get_reuse_decision_records',
    'make_reuse_decision',
    'store_reuse_decision',
    'validate_decision_record',
)

DecisionFunction = Callable[[dict[str, Any]], Mapping[str, Any]]
SeedFactory = Callable[[int], int]
CandidateSourceResolver = Callable[[str], Any]
InputRewriter = Callable[[Mapping[str, Any], 'ReuseDecision'], Mapping[str, Any]]


def _canonical_uuid(value: Any, field_name: str) -> str:
    """Return a canonical UUID string or raise a decision error."""
    if not isinstance(value, str):
        raise ReuseDecisionError(f'{field_name} must be a UUID string')
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as exception:
        raise ReuseDecisionError(f'{field_name} must be a UUID string') from exception
    canonical = str(parsed)
    if canonical != value.lower():
        raise ReuseDecisionError(f'{field_name} must use canonical UUID formatting')
    return canonical


def _validate_request_id(request_id: Any) -> str:
    if not isinstance(request_id, str) or not _REQUEST_ID_PATTERN.fullmatch(request_id):
        raise ReuseDecisionError('request_id must use the req-<UUID> format')
    return request_id


def _validate_seed(seed: Any, field_name: str = 'seed') -> int:
    if type(seed) is not int or not 1 <= seed <= MAX_SEED:
        raise ReuseDecisionError(f'{field_name} must be an integer between 1 and {MAX_SEED}')
    return seed


def _json_copy(value: Any, field_name: str) -> Any:
    """Return a detached JSON-safe copy."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exception:
        raise ReuseDecisionError(f'{field_name} must be JSON serializable') from exception


@dataclass(frozen=True)
class ReuseCandidate:
    """One candidate supplied to the reuse policy.

    ``uuid`` identifies the candidate result Data node. The plugin's candidate
    resolver translates it to the ProcessNode that created that result node.
    Additional comparison fields are preserved in ``metadata`` in the decision
    record.
    """

    uuid: str
    qualified: bool
    fingerprint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'uuid', _canonical_uuid(self.uuid, 'candidate uuid'))
        if type(self.qualified) is not bool:
            raise ReuseDecisionError('candidate qualified must be a boolean')
        if self.fingerprint is not None and not isinstance(self.fingerprint, str):
            raise ReuseDecisionError('candidate fingerprint must be a string or None')
        object.__setattr__(self, 'metadata', _json_copy(dict(self.metadata), 'candidate metadata'))

    @classmethod
    def from_mapping(cls, candidate: Mapping[str, Any]) -> ReuseCandidate:
        """Create a candidate from a mapping while preserving extra fields."""
        if not isinstance(candidate, Mapping):
            raise ReuseDecisionError('candidate must be a mapping')
        candidate_data = dict(candidate)
        if 'uuid' not in candidate_data or 'qualified' not in candidate_data:
            raise ReuseDecisionError('candidate must contain uuid and qualified')
        fingerprint = candidate_data.pop('fingerprint', None)
        uuid = candidate_data.pop('uuid')
        qualified = candidate_data.pop('qualified')
        return cls(uuid=uuid, qualified=qualified, fingerprint=fingerprint, metadata=candidate_data)

    def as_dict(self) -> dict[str, Any]:
        """Return the flattened representation stored in the decision record."""
        record = copy.deepcopy(dict(self.metadata))
        record['uuid'] = self.uuid
        record['qualified'] = self.qualified
        if self.fingerprint is not None:
            record['fingerprint'] = self.fingerprint
        return record


def default_seed_factory(original_seed: int) -> int:
    """Generate a valid seed that differs from ``original_seed``."""
    original_seed = _validate_seed(original_seed, 'original_seed')
    while True:
        candidate = secrets.randbelow(MAX_SEED) + 1
        if candidate != original_seed:
            return candidate


def default_candidate_source(candidate_uuid: str) -> Any:
    """Resolve a result Data node UUID to the ProcessNode that created it."""
    from aiida import orm
    from aiida.common import exceptions

    try:
        node = orm.load_node(candidate_uuid)
    except (exceptions.NotExistent, ValueError):
        return None
    if isinstance(node, orm.ProcessNode):
        return node
    return getattr(node, 'creator', None)


@dataclass(frozen=True)
class ReuseConfiguration:
    """Plugin-provided configuration for one process reuse decision.

    :param request_id: Request identifier in the ``req-<UUID>`` format.
    :param reuse_policy: User-selected policy.
    :param original_seed: Original reproducible seed from the request.
    :param input_rewriter: Callback returning a complete rewritten input mapping.
        The callback must apply ``decision.effective_seed`` when the plugin uses seeds.
    :param candidates: Candidate results in comparison order.
    :param decision_function: Optional plugin decision callback.
    :param seed_factory: Factory used by the default ``new_sample`` policy.
    :param resolve_candidate_source: Callback resolving a result UUID to its creator process.
    """

    request_id: str
    reuse_policy: ReusePolicy | str
    original_seed: int
    input_rewriter: InputRewriter
    candidates: Sequence[ReuseCandidate | Mapping[str, Any]] = field(default_factory=tuple)
    decision_function: DecisionFunction | None = None
    seed_factory: SeedFactory = default_seed_factory
    resolve_candidate_source: CandidateSourceResolver = default_candidate_source
    materialization: ReuseMaterialization | str = ReuseMaterialization.CACHE_PROCESS

    def __post_init__(self) -> None:
        object.__setattr__(self, 'request_id', _validate_request_id(self.request_id))
        try:
            policy = ReusePolicy(self.reuse_policy)
        except ValueError as exception:
            allowed = ', '.join(policy.value for policy in ReusePolicy)
            raise ReuseDecisionError(f'reuse_policy must be one of: {allowed}') from exception
        object.__setattr__(self, 'reuse_policy', policy)
        try:
            materialization = ReuseMaterialization(self.materialization)
        except (TypeError, ValueError) as exception:
            allowed = ', '.join(item.value for item in ReuseMaterialization)
            raise ReuseDecisionError(f'materialization must be one of: {allowed}') from exception
        object.__setattr__(self, 'materialization', materialization)
        object.__setattr__(self, 'original_seed', _validate_seed(self.original_seed, 'original_seed'))
        if not callable(self.input_rewriter):
            raise ReuseDecisionError('input_rewriter must be callable')
        if self.decision_function is not None and not callable(self.decision_function):
            raise ReuseDecisionError('decision_function must be callable or None')
        if not callable(self.seed_factory):
            raise ReuseDecisionError('seed_factory must be callable')
        if not callable(self.resolve_candidate_source):
            raise ReuseDecisionError('resolve_candidate_source must be callable')
        candidates = tuple(
            candidate if isinstance(candidate, ReuseCandidate) else ReuseCandidate.from_mapping(candidate)
            for candidate in self.candidates
        )
        object.__setattr__(self, 'candidates', candidates)


@dataclass(frozen=True)
class ReuseDecision:
    """Validated decision returned by the engine."""

    action: ReuseAction
    materialization: ReuseMaterialization
    selected_candidate_uuid: str | None
    selected_process_uuid: str | None
    effective_seed: int
    reason_code: ReuseReason
    reason_text: str
    candidate_compare_list: tuple[dict[str, Any], ...]
    record: Mapping[str, Any]

    @property
    def uses_cache(self) -> bool:
        """Return whether this decision requires the selected cache source."""
        return (self.action is ReuseAction.REUSE and self.materialization is ReuseMaterialization.CACHE_PROCESS)


def _candidate_records(configuration: ReuseConfiguration) -> tuple[dict[str, Any], ...]:
    """Return detached candidate comparison records in caller order."""
    return tuple(candidate.as_dict() for candidate in configuration.candidates)


def _find_candidate(candidates: Sequence[Mapping[str, Any]], selected_uuid: str) -> Mapping[str, Any]:
    """Return the unique selected candidate and reject unqualified choices."""
    matches = [candidate for candidate in candidates if candidate['uuid'] == selected_uuid]
    if not matches:
        raise ReuseDecisionError('reuse selected a candidate outside candidate_compare_list')
    if any(candidate.get('qualified') is not True for candidate in matches):
        raise ReuseDecisionError('reuse selected an unqualified candidate')
    if len(matches) != 1:
        raise ReuseDecisionError('candidate_compare_list contains the selected UUID more than once')
    return matches[0]


def _normalize_outcome(
    outcome: Mapping[str, Any],
    policy: ReusePolicy,
    original_seed: int,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the small decision contract and enforce the user's policy."""
    if not isinstance(outcome, Mapping):
        raise ReuseDecisionError('decision outcome must be a mapping')

    raw_action = outcome.get('action')
    try:
        action = ReuseAction(raw_action)
    except ValueError as exception:
        allowed = ', '.join(item.value for item in ReuseAction)
        raise ReuseDecisionError(f'decision action must be one of: {allowed}') from exception

    if action not in _ALLOWED_ACTIONS[policy]:
        allowed = ', '.join(sorted(item.value for item in _ALLOWED_ACTIONS[policy]))
        raise ReuseDecisionError(f'policy {policy.value} does not allow action {action.value}; allowed: {allowed}')

    raw_selected_uuid = outcome.get('selected_candidate_uuid')
    selected_uuid = None
    if action is ReuseAction.REUSE:
        if raw_selected_uuid is None:
            raise ReuseDecisionError('reuse decision must provide selected_candidate_uuid')
        selected_uuid = _canonical_uuid(raw_selected_uuid, 'selected_candidate_uuid')
        _find_candidate(candidates, selected_uuid)
    elif raw_selected_uuid is not None:
        raise ReuseDecisionError('only a reuse decision may select a candidate')

    effective_seed = _validate_seed(outcome.get('effective_seed'), 'effective_seed')
    if action in (ReuseAction.REUSE, ReuseAction.RERUN):
        if effective_seed != original_seed:
            raise ReuseDecisionError(f'{action.value} must preserve the original seed')
    elif effective_seed == original_seed:
        raise ReuseDecisionError('new_sample effective_seed must differ from original_seed')

    reason_text = outcome.get('reason_text')
    if not isinstance(reason_text, str) or not reason_text.strip():
        raise ReuseDecisionError('reason_text must be a non-empty string')

    return {
        'action': action,
        'selected_candidate_uuid': selected_uuid,
        'effective_seed': effective_seed,
        'reason_text': reason_text.strip(),
    }


def _default_outcome(
    configuration: ReuseConfiguration,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ReuseReason]:
    """Apply the default policy semantics without invoking plugin code."""
    policy = configuration.reuse_policy
    original_seed = configuration.original_seed

    if policy is ReusePolicy.ALLOW_REUSE:
        selected = next((candidate for candidate in candidates if candidate.get('qualified') is True), None)
        if selected is not None:
            return (
                _normalize_outcome(
                    {
                        'action': ReuseAction.REUSE,
                        'selected_candidate_uuid': selected['uuid'],
                        'effective_seed': original_seed,
                        'reason_text': f'allow_reuse selected the first qualified candidate {selected["uuid"]}',
                    },
                    policy,
                    original_seed,
                    candidates,
                ),
                ReuseReason.REUSE_QUALIFIED,
            )
        return (
            _normalize_outcome(
                {
                    'action': ReuseAction.RERUN,
                    'selected_candidate_uuid': None,
                    'effective_seed': original_seed,
                    'reason_text': 'allow_reuse found no qualified candidate; rerun with the original seed',
                },
                policy,
                original_seed,
                candidates,
            ),
            ReuseReason.ALLOW_REUSE_NO_CANDIDATE,
        )

    if policy is ReusePolicy.FORCE_RERUN:
        return (
            _normalize_outcome(
                {
                    'action': ReuseAction.RERUN,
                    'selected_candidate_uuid': None,
                    'effective_seed': original_seed,
                    'reason_text': 'force_rerun ignores candidates and reruns with the original seed',
                },
                policy,
                original_seed,
                candidates,
            ),
            ReuseReason.FORCE_RERUN,
        )

    if policy is ReusePolicy.NEW_SAMPLE:
        effective_seed = _validate_seed(configuration.seed_factory(original_seed), 'new_sample effective_seed')
        if effective_seed == original_seed:
            raise ReuseDecisionError('new_sample effective_seed must differ from original_seed')
        return (
            _normalize_outcome(
                {
                    'action': ReuseAction.NEW_SAMPLE,
                    'selected_candidate_uuid': None,
                    'effective_seed': effective_seed,
                    'reason_text': 'new_sample ignores candidates and reruns with a different seed',
                },
                policy,
                original_seed,
                candidates,
            ),
            ReuseReason.NEW_SAMPLE,
        )

    raise ReuseDecisionError(f'unsupported reuse policy: {policy}')


def _custom_outcome(
    configuration: ReuseConfiguration,
    candidates: Sequence[Mapping[str, Any]],
    process_uuid: str,
) -> dict[str, Any]:
    """Invoke the plugin decision function and enforce policy constraints."""
    assert configuration.decision_function is not None
    context = {
        'request_id': configuration.request_id,
        'process_uuid': process_uuid,
        'reuse_policy': configuration.reuse_policy.value,
        'candidate_compare_list': [copy.deepcopy(candidate) for candidate in candidates],
        'original_seed': configuration.original_seed,
    }
    outcome = configuration.decision_function(copy.deepcopy(context))
    return _normalize_outcome(
        outcome,
        configuration.reuse_policy,
        configuration.original_seed,
        candidates,
    )


def _resolve_candidate_process_uuid(
    configuration: ReuseConfiguration,
    candidate_uuid: str,
) -> str:
    """Resolve a result UUID to the UUID of a stored creator process node."""
    source = configuration.resolve_candidate_source(candidate_uuid)
    if source is None:
        raise ReuseDecisionError(f'candidate {candidate_uuid} has no creator process')

    if isinstance(source, str):
        process_uuid = _canonical_uuid(source, 'candidate creator process_uuid')
    else:
        source_node = getattr(source, 'node', source)
        if not hasattr(source_node, 'uuid'):
            raise ReuseDecisionError('candidate resolver must return a ProcessNode or UUID')
        process_uuid = _canonical_uuid(str(source_node.uuid), 'candidate creator process_uuid')

    from aiida import orm
    from aiida.common import exceptions

    try:
        process_node = orm.load_node(process_uuid)
    except (exceptions.NotExistent, ValueError) as exception:
        raise ReuseDecisionError(f'cannot load candidate creator process {process_uuid}') from exception
    if not isinstance(process_node, orm.ProcessNode):
        raise ReuseDecisionError(f'candidate creator {process_uuid} is not a ProcessNode')
    return process_uuid


def make_reuse_decision(
    configuration: ReuseConfiguration,
    process_uuid: str,
    *,
    timestamp: str | None = None,
) -> ReuseDecision:
    """Build and validate one decision for a concrete AiiDA process."""
    if not isinstance(configuration, ReuseConfiguration):
        raise TypeError('configuration must be a ReuseConfiguration')
    process_uuid = _canonical_uuid(process_uuid, 'process_uuid')
    candidates = _candidate_records(configuration)

    if configuration.decision_function is None:
        outcome, reason_code = _default_outcome(configuration, candidates)
    else:
        outcome = _custom_outcome(configuration, candidates, process_uuid)
        reason_code = ReuseReason.CUSTOM_DECISION

    selected_process_uuid = None
    if outcome['action'] is ReuseAction.REUSE:
        selected_process_uuid = _resolve_candidate_process_uuid(
            configuration,
            outcome['selected_candidate_uuid'],
        )

    record = {
        'request_id': configuration.request_id,
        'process_uuid': process_uuid,
        'reuse_policy': configuration.reuse_policy.value,
        'candidate_compare_list': [copy.deepcopy(candidate) for candidate in candidates],
        'final_decision': {
            'action': outcome['action'].value,
            'selected_candidate_uuid': outcome['selected_candidate_uuid'],
            'effective_seed': outcome['effective_seed'],
        },
        'reason': {
            'reason_code': reason_code.value,
            'reason_text': outcome['reason_text'],
        },
        'timestamp': timestamp or datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    validate_decision_record(record)
    return ReuseDecision(
        action=outcome['action'],
        materialization=configuration.materialization,
        selected_candidate_uuid=outcome['selected_candidate_uuid'],
        selected_process_uuid=selected_process_uuid,
        effective_seed=outcome['effective_seed'],
        reason_code=reason_code,
        reason_text=outcome['reason_text'],
        candidate_compare_list=candidates,
        record=record,
    )


def validate_decision_record(record: Mapping[str, Any]) -> None:
    """Validate the persisted decision-record schema and policy invariants."""
    if not isinstance(record, Mapping):
        raise ReuseDecisionError('decision record must be a mapping')

    required = (
        'request_id',
        'process_uuid',
        'reuse_policy',
        'candidate_compare_list',
        'final_decision',
        'reason',
        'timestamp',
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ReuseDecisionError(f'decision record is missing fields: {", ".join(missing)}')

    _validate_request_id(record['request_id'])
    _canonical_uuid(record['process_uuid'], 'process_uuid')
    try:
        policy = ReusePolicy(record['reuse_policy'])
    except ValueError as exception:
        allowed = ', '.join(item.value for item in ReusePolicy)
        raise ReuseDecisionError(f'reuse_policy must be one of: {allowed}') from exception

    raw_candidates = record['candidate_compare_list']
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        raise ReuseDecisionError('candidate_compare_list must be a sequence')
    candidates = tuple(
        candidate.as_dict()
        if isinstance(candidate, ReuseCandidate)
        else ReuseCandidate.from_mapping(candidate).as_dict()
        for candidate in raw_candidates
    )
    candidate_uuids = [candidate['uuid'] for candidate in candidates]
    if len(candidate_uuids) != len(set(candidate_uuids)):
        raise ReuseDecisionError('candidate_compare_list contains duplicate UUIDs')

    final_decision = record['final_decision']
    if not isinstance(final_decision, Mapping):
        raise ReuseDecisionError('final_decision must be a mapping')
    missing_decision = [
        field for field in ('action', 'selected_candidate_uuid', 'effective_seed') if field not in final_decision
    ]
    if missing_decision:
        raise ReuseDecisionError(f'final_decision is missing fields: {", ".join(missing_decision)}')

    try:
        action = ReuseAction(final_decision['action'])
    except ValueError as exception:
        allowed = ', '.join(item.value for item in ReuseAction)
        raise ReuseDecisionError(f'final_decision action must be one of: {allowed}') from exception
    if action not in _ALLOWED_ACTIONS[policy]:
        allowed = ', '.join(sorted(item.value for item in _ALLOWED_ACTIONS[policy]))
        raise ReuseDecisionError(f'policy {policy.value} does not allow action {action.value}; allowed: {allowed}')

    selected_uuid = final_decision['selected_candidate_uuid']
    if action is ReuseAction.REUSE:
        if selected_uuid is None:
            raise ReuseDecisionError('reuse decision must provide selected_candidate_uuid')
        selected_uuid = _canonical_uuid(selected_uuid, 'selected_candidate_uuid')
        _find_candidate(candidates, selected_uuid)
    elif selected_uuid is not None:
        raise ReuseDecisionError('rerun and new_sample decisions must not select a candidate')

    _validate_seed(final_decision['effective_seed'], 'effective_seed')

    reason = record['reason']
    if not isinstance(reason, Mapping):
        raise ReuseDecisionError('reason must be a mapping')
    try:
        reason_code = ReuseReason(reason.get('reason_code'))
    except ValueError as exception:
        allowed = ', '.join(item.value for item in ReuseReason)
        raise ReuseDecisionError(f'reason_code must be one of: {allowed}') from exception
    reason_text = reason.get('reason_text')
    if not isinstance(reason_text, str) or not reason_text.strip():
        raise ReuseDecisionError('reason_text must be a non-empty string')

    compatible_actions = {
        ReuseReason.REUSE_QUALIFIED: {ReuseAction.REUSE},
        ReuseReason.ALLOW_REUSE_NO_CANDIDATE: {ReuseAction.RERUN},
        ReuseReason.FORCE_RERUN: {ReuseAction.RERUN},
        ReuseReason.NEW_SAMPLE: {ReuseAction.NEW_SAMPLE},
        ReuseReason.CUSTOM_DECISION: set(ReuseAction),
    }
    if action not in compatible_actions[reason_code]:
        raise ReuseDecisionError(f'{reason_code.value} is not compatible with action {action.value}')
    compatible_policies = {
        ReuseReason.REUSE_QUALIFIED: {ReusePolicy.ALLOW_REUSE},
        ReuseReason.ALLOW_REUSE_NO_CANDIDATE: {ReusePolicy.ALLOW_REUSE},
        ReuseReason.FORCE_RERUN: {ReusePolicy.FORCE_RERUN},
        ReuseReason.NEW_SAMPLE: {ReusePolicy.NEW_SAMPLE},
        ReuseReason.CUSTOM_DECISION: set(ReusePolicy),
    }
    if policy not in compatible_policies[reason_code]:
        allowed = ', '.join(sorted(item.value for item in compatible_policies[reason_code]))
        raise ReuseDecisionError(f'{reason_code.value} requires reuse_policy {allowed}; got {policy.value}')
    if reason_code is ReuseReason.ALLOW_REUSE_NO_CANDIDATE:
        if any(candidate.get('qualified') is True for candidate in candidates):
            raise ReuseDecisionError('R02 requires that no qualified candidate exists')

    timestamp = record['timestamp']
    if not isinstance(timestamp, str) or not _TIMESTAMP_PATTERN.fullmatch(timestamp):
        raise ReuseDecisionError('timestamp must use ISO 8601 UTC with second precision')
    try:
        datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError as exception:
        raise ReuseDecisionError('timestamp must be a real UTC timestamp') from exception

    try:
        json.dumps(dict(record), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exception:
        raise ReuseDecisionError('decision record must be JSON serializable') from exception


def _process_node(process_or_node: Any) -> Any:
    """Return a stored ProcessNode from a process object or node."""
    node = getattr(process_or_node, 'node', process_or_node)
    if not hasattr(node, 'base') or not hasattr(node.base, 'extras'):
        raise TypeError('decision records can only be saved on an AiiDA ProcessNode')
    from aiida import orm

    if not isinstance(node, orm.ProcessNode):
        raise TypeError('decision records can only be saved on an AiiDA ProcessNode')
    if not node.is_stored:
        raise ReuseDecisionError('decision records require a stored ProcessNode')
    return node


def store_reuse_decision(process_or_node: Any, record: Mapping[str, Any]) -> dict[str, Any]:
    """Persist one decision record, replacing a replay of the same process and request."""
    validate_decision_record(record)
    node = _process_node(process_or_node)
    stored = node.base.extras.get(DECISION_RECORDS_EXTRA_KEY, [])
    if not isinstance(stored, list):
        raise ReuseDecisionError(f'extra {DECISION_RECORDS_EXTRA_KEY} must contain a list')
    for existing in stored:
        if not isinstance(existing, Mapping):
            raise ReuseDecisionError(f'extra {DECISION_RECORDS_EXTRA_KEY} contains a non-mapping record')
        validate_decision_record(existing)

    record_copy = copy.deepcopy(dict(record))
    kept = [
        item
        for item in stored
        if not (
            item.get('request_id') == record_copy['request_id']
            and item.get('process_uuid') == record_copy['process_uuid']
        )
    ]
    kept.append(record_copy)
    node.base.extras.set(DECISION_RECORDS_EXTRA_KEY, kept)
    return record_copy


def get_reuse_decision_records(process_or_node: Any) -> list[dict[str, Any]]:
    """Return all valid decision records attached to one ProcessNode."""
    node = _process_node(process_or_node)
    stored = node.base.extras.get(DECISION_RECORDS_EXTRA_KEY, [])
    if not isinstance(stored, list):
        raise ReuseDecisionError(f'extra {DECISION_RECORDS_EXTRA_KEY} must contain a list')
    records = []
    for item in stored:
        if not isinstance(item, Mapping):
            raise ReuseDecisionError(f'extra {DECISION_RECORDS_EXTRA_KEY} contains a non-mapping record')
        validate_decision_record(item)
        records.append(copy.deepcopy(dict(item)))
    return records
