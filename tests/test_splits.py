import pandas as pd
import pytest

from dispatch_router.config import ROOT
from dispatch_router.datasets import assert_no_overlap, read_jsonl

P = ROOT / "data" / "prompts"


def test_overlap_detects_id_and_text():
    tr = pd.DataFrame({"id": ["a", "b"], "question": ["What is 2+2?", "Hi there"]})
    with pytest.raises(AssertionError):
        assert_no_overlap(tr, pd.DataFrame({"id": ["a"], "question": ["new"]}))
    with pytest.raises(AssertionError):
        assert_no_overlap(tr, pd.DataFrame({"id": ["z"], "question": ["what is 2 + 2"]}))
    assert_no_overlap(tr, pd.DataFrame({"id": ["z"], "question": ["What is 3+3?"]}))


@pytest.mark.skipif(not (P / "test_public.jsonl").exists(), reason="test set not built")
def test_real_train_and_test_do_not_overlap():
    train = read_jsonl(P / "train_pool.jsonl")
    test = read_jsonl(P / "test_public.jsonl")
    assert_no_overlap(train, test)
    assert set(train.split) == {"train", "val"}
    # validation share is ~15% within each dataset
    for _, g in train.groupby("dataset"):
        assert abs((g.split == "val").mean() - 0.15) < 0.01
