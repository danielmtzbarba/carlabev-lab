import pytest

from src.config.studies.registry import STUDY_REGISTRY, get_eval_protocol, get_experiment_spec, get_train_protocol


@pytest.mark.unit
def test_registered_studies_resolve_all_protocol_references():
    assert STUDY_REGISTRY

    for study_id, study in STUDY_REGISTRY.items():
        assert study.experiments
        for exp_id, spec in study.experiments.items():
            assert get_experiment_spec(study_id, exp_id) == spec
            assert get_train_protocol(study_id, spec.train_protocol_id).protocol_id == spec.train_protocol_id
            for protocol_id in spec.eval_protocol_ids:
                assert get_eval_protocol(study_id, protocol_id).protocol_id == protocol_id


@pytest.mark.unit
def test_registered_studies_have_unique_experiment_ids():
    for study in STUDY_REGISTRY.values():
        ids = list(study.experiments.keys())
        assert ids == sorted(set(ids))
