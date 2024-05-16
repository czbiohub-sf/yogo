from typing import List, Dict, Optional


class InvalidSplitFraction(Exception):
    pass


class SplitFractions:
    """
    Fraction of data to be used for training, validation, and testing. This class basically
    just provides `from_<type>` methods.
    """

    def __init__(self, train: float, val: float, test: Optional[float]) -> None:
        self.train: float = train
        self.val: float = val
        self.test: Optional[float] = test

        train_in_range = 0 <= self.train <= 1
        val_in_range = 0 <= self.val <= 1
        test_in_range = 0 <= (self.test or 0) <= 1

        if not (train_in_range and val_in_range and test_in_range):
            raise ValueError(
                f"train, val, and test must be in range [0,1]; they are {self.train}, {self.val}, and {self.test}"
            )
        elif not abs(self.train + self.val + (self.test or 0) - 1) < 1e-10:
            raise ValueError(
                f"train, val, and test must sum to 1; they sum to {self.train + self.val + (self.test or 0)}"
            )

    def __repr__(self) -> str:
        return f"SplitFractions(train={self.train}, val={self.val}, test={self.test})"

    def __contains__(self, item: object) -> bool:
        return item in self.to_dict()

    @classmethod
    def from_list(
        cls, lst: List[float], test_paths_present: bool = True
    ) -> "SplitFractions":
        if len(lst) != 3:
            raise InvalidSplitFraction(
                f"SplitFractions.from_list's list must have length 3, but found length {len(lst)}"
            )
        return cls.from_dict(
            dict(zip(["train", "val", "test"], lst)),
            test_paths_present=test_paths_present,
        )

    @classmethod
    def train_only(cls) -> "SplitFractions":
        return cls(1, 0, None)

    @classmethod
    def from_dict(
        cls, dct: Dict[str, float], test_paths_present: bool = True
    ) -> "SplitFractions":
        if test_paths_present and "test" in dct:
            raise InvalidSplitFraction(
                "when `test_paths` is present in a dataset descriptor file, 'test' "
                "is not a valid key for `dataset_split_fractions`, since we will use "
                "all the data from `test_paths` for testing"
            )
        if not any(v in dct for v in ("train", "val", "test")):
            raise InvalidSplitFraction(
                f"dct must have keys `train`, `val`, and `test` - found keys {dct.keys()}"
            )
        if len(dct) > 3:
            raise InvalidSplitFraction(
                f"dct must have keys `train`, `val`, and `test` only, but found {len(dct)} keys"
            )
        return cls(dct["train"], dct["val"], dct.get("test", None))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SplitFractions):
            return False
        return (
            self.train == other.train
            and self.val == other.val
            and self.test == other.test
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            **({"train": self.train} if self.train is not None else dict()),
            **({"val": self.val} if self.val is not None else dict()),
            **({"test": self.test} if self.test is not None else dict()),
        }

    def keys(self) -> List[str]:
        return list(self.to_dict().keys())

    def partition_sizes(self, total_size: int) -> Dict[str, int]:
        split_fractions = self.to_dict()
        keys = self.keys()

        dataset_sizes = {k: round(split_fractions[k] * total_size) for k in keys[:-1]}
        final_dataset_size = {keys[-1]: total_size - sum(dataset_sizes.values())}
        split_sizes = {**dataset_sizes, **final_dataset_size}

        all_sizes_are_gt_0 = all([sz >= 0 for sz in split_sizes.values()])
        split_sizes_eq_dataset_size = sum(split_sizes.values()) == total_size

        if not (all_sizes_are_gt_0 and split_sizes_eq_dataset_size):
            raise ValueError(
                f"could not create valid dataset split sizes: {split_sizes}, "
                f"full dataset size is {total_size}"
            )

        return split_sizes
