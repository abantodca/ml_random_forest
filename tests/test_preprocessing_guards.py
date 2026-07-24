from __future__ import annotations

import pandas as pd

from src.step_02_clean.rare_categories import RareCategoryGrouper
from src.step_04_train.validation_split import temporal_tail_holdout_indices


def test_rare_category_map_is_learned_from_train_only() -> None:
    train = pd.DataFrame({"FORMATO": ["A", "A", "A", "B"]})
    transform = RareCategoryGrouper(columns=["FORMATO"], min_count=2)
    transform.fit(train)

    test = pd.DataFrame({"FORMATO": ["A", "B", "C", None]})
    assert transform.transform(test)["FORMATO"].tolist() == [
        "A",
        "OTROS",
        "OTROS",
        "OTROS",
    ]


def test_early_stopping_holdout_is_the_temporal_tail() -> None:
    train, valid = temporal_tail_holdout_indices(100)
    assert train.tolist() == list(range(88))
    assert valid.tolist() == list(range(88, 100))
