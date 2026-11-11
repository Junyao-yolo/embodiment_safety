"""Ensure paired recovery planning and both-arm training acceptance."""
import json
import pytest
from tools.task1_contact_recovery.dual_queue import plan_cases, training_gate


def test_paired_planning_preserves_seed_and_observation_alignment(tmp_path):
    reports = {}
    for variant in ('official', 'current'):
        episodes = []
        for seed in range(5):
            episode = tmp_path / variant / str(seed); episode.mkdir(parents=True)
            stage = 'insertion_ready' if variant == 'official' else 'grasp_clear'
            for index in range(30, 51):
                label = dict(index=index-1, events=[dict(stage=stage)] if index==30 else [],
                    evidence=dict(bilateral_grasp_stable=index<45, orientation_ok=index<45,
                                  grasp_in_allowed_band=True, no_safety_violation=True))
                (episode / f'stage_transition_{index-1:03d}.json').write_text(json.dumps(label))
                (episode / f'raw_{index:03d}.json').write_text(json.dumps(dict(observation_index=index)))
            episodes.append(dict(seed=seed, stages=['grasp_clear', stage],
                                 episode=str(episode/'native/episode.json')))
        reports[variant] = dict(episodes=episodes)
    planned = plan_cases(reports)
    for variant in reports:
        assert len(planned[variant]) == 10
        assert [c['seed'] for c in planned[variant]] == [s for s in range(5) for _ in range(2)]
        for first, second in zip(planned[variant][::2], planned[variant][1::2]):
            assert first['handoff_index'] == 30
            assert second['handoff_index'] == 43
            assert second['reference_raw']['observation_index'] == 43


def test_both_arms_need_accepted_multiseed_recovery():
    def rows(seeds):
        return [dict(seed=s, teacher=dict(training_eligible=True)) for s in seeds]
    passed, _ = training_gate(dict(official=rows([1,1,2]), current=rows([1,1,1])))
    assert not passed
    passed, _ = training_gate(dict(official=rows([1,1,2]), current=rows([3,3,4])))
    assert passed
