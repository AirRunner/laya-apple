from __future__ import annotations

import pytest

from laya_apple.errors import InvalidRequestError
from laya_apple.prompt import Calibration, clamp_temperature, prepare


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_prepared_token_ids_match_goldens_exactly(tokenizer, config, golden):
    """prepare() must reproduce the upstream token ids/markers for every golden case."""
    for case in golden["cases"]:
        prep = prepare(tokenizer, config, case["state"], case["questions"])
        assert prep.items == case["items"], case["name"]


@pytest.mark.parametrize(
    "bad_questions",
    [
        {},
        [],
        "not a dict or list",
        {"q1": {"instructions": "missing type"}},
        {"q1": {"type": "bogus", "instructions": "x"}},
        {"q1": {"type": "choice", "instructions": "x"}},  # missing criteria
        {"q1": {"type": "choice", "instructions": "x", "criteria": {}}},
        {"q1": {"type": "choice", "instructions": "x", "criteria": ["a", "a"]}},  # dup labels
        {"q1": {"type": "score", "instructions": "x", "criteria": []}},
        {"q1": {"type": "score", "instructions": "x", "criteria": "not a list"}},
        {"q1": "not a dict"},
        ["dup", "dup"],
        [1, 2],
    ],
)
@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_malformed_questions_raise_invalid_request_error(tokenizer, config, bad_questions):
    with pytest.raises(InvalidRequestError):
        prepare(tokenizer, config, "some context", bad_questions)


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_list_of_strings_becomes_noul_questions(tokenizer, config):
    prep = prepare(tokenizer, config, "ctx", ["Is this urgent?", "Did they mention billing?"])
    assert prep.question_count == 2
    assert all(q["t"] == "noul" for q in prep.internal)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (1.0, 1.0),
        (0.1, 0.5),
        (10.0, 5.0),
        (0.5, 0.5),
        (5.0, 5.0),
        ("not a number", 1.0),
        (None, 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
    ],
)
def test_calibration_temperature_clamp(raw, expected):
    assert clamp_temperature(raw) == expected


def test_calibration_warns_when_checkpoint_values_are_clamped():
    with pytest.warns(RuntimeWarning):
        Calibration({"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {"choice:2": 10.0}})


def test_calibration_no_warning_when_within_range():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Calibration({"temperature": [1.0, 2.0, 3.0]})


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_answer_schema_choice(tokenizer, config, calibration):
    from laya_apple.prompt import format_answers

    prep = prepare(tokenizer, config, "ctx", {"q1": {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}})
    logits = [[1.0, 0.5] + [-1e4] * 30]
    act = [[0.2, 0.8]]
    answers = format_answers(prep, logits, act, calibration)
    ans = answers["q1"]
    assert set(ans) >= {"type", "confidence", "action", "choice", "probabilities"}
    assert ans["type"] == "choice"
    assert ans["choice"] in ("a", "b")
    for v in ans["probabilities"].values():
        assert round(v, 4) == v
    assert round(ans["action"]["act_probability"], 4) == ans["action"]["act_probability"]


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_answer_schema_score(tokenizer, config, calibration):
    from laya_apple.prompt import format_answers

    prep = prepare(
        tokenizer, config, "ctx", {"q1": {"type": "score", "instructions": "rate", "criteria": ["lo", "mid", "hi"]}}
    )
    logits = [[1.0, 0.5, 0.1] + [-1e4] * 29]
    act = [[0.2, 0.8]]
    ans = format_answers(prep, logits, act, calibration)["q1"]
    assert set(ans) >= {"type", "confidence", "action", "score", "legend", "probabilities"}


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_answer_schema_noul(tokenizer, config, calibration):
    from laya_apple.prompt import format_answers

    prep = prepare(tokenizer, config, "ctx", {"q1": {"type": "noul", "instructions": "is it?"}})
    logits = [[0.3, 0.7] + [-1e4] * 30]
    act = [[0.9, 0.1]]
    ans = format_answers(prep, logits, act, calibration)["q1"]
    assert set(ans) >= {"type", "confidence", "action", "noul"}
    assert 0.0 <= ans["noul"] <= 1.0


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_answer_probabilities_are_rounded_to_4_decimals(tokenizer, config, calibration):
    from laya_apple.prompt import format_answers

    prep = prepare(
        tokenizer, config, "ctx", {"q1": {"type": "choice", "instructions": "pick", "criteria": ["a", "b", "c"]}}
    )
    logits = [[1.0 / 3, 0.123456789, -0.5] + [-1e4] * 29]
    act = [[0.111111, 0.888889]]
    ans = format_answers(prep, logits, act, calibration)["q1"]
    for v in ans["probabilities"].values():
        assert v == round(v, 4)
    assert ans["action"]["act_probability"] == round(ans["action"]["act_probability"], 4)
    assert ans["confidence"] == round(ans["confidence"], 4)


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_answer_non_finite_logits_raise_floating_point_error(tokenizer, config, calibration):
    from laya_apple.prompt import format_answers

    prep = prepare(tokenizer, config, "ctx", {"q1": {"type": "noul", "instructions": "is it?"}})
    logits = [[float("nan"), 0.5] + [-1e4] * 30]
    act = [[0.5, 0.5]]
    with pytest.raises(FloatingPointError):
        format_answers(prep, logits, act, calibration)


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_answer_non_finite_action_logits_raise_floating_point_error(tokenizer, config, calibration):
    from laya_apple.prompt import format_answers

    prep = prepare(tokenizer, config, "ctx", {"q1": {"type": "noul", "instructions": "is it?"}})
    logits = [[0.5, 0.5] + [-1e4] * 30]
    act = [[float("inf"), 0.5]]
    with pytest.raises(FloatingPointError):
        format_answers(prep, logits, act, calibration)
