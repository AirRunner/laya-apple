from __future__ import annotations

import pytest

from laya_apple.workload import make_request

# tokenizer/config need a downloaded checkpoint.
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("length", [64, 128, 200])
def test_make_request_hits_exact_length(tokenizer, config, length):
    state, questions = make_request(tokenizer, config, length, n_questions=1)
    from laya_apple.prompt import prepare

    prep = prepare(tokenizer, config, state, questions)
    assert prep.sequence_length == length


def test_make_request_multi_question_hits_exact_length(tokenizer, config):
    from laya_apple.prompt import prepare

    state, questions = make_request(tokenizer, config, 96, n_questions=3)
    prep = prepare(tokenizer, config, state, questions)
    assert prep.sequence_length == 96
    assert prep.question_count == 3


def test_make_request_exceeding_max_len_raises(tokenizer, config):
    with pytest.raises(ValueError):
        make_request(tokenizer, config, int(config["max_len"]) + 1000)
