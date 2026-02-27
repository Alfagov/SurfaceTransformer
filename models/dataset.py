import glob
from typing import List
from torch.utils.data import Dataset, DataLoader
import lightning as l
import torch
import numpy as np
import polars as pl

SURFACE_FEATURE_COLUMNS = ["S", "K", "T", "vix", "dividend_yield", "rate"]
GREEK_LABEL_COLUMNS = ["delta", "gamma", "theta"]

class SurfaceOptionsDataModule(l.LightningDataModule):
    def __init__(
            self,
            folder: str,
            sofr_path: str = None,
            context_size: int = 64,
            query_size: int = 64,
            batch_size_dates: int = 8,
            num_workers: int = 4,
            seed: int = 42,
    ):
        super().__init__()
        self.folder = folder
        self.sofr_path = sofr_path
        self.batch_size_dates = batch_size_dates
        self.context_size = context_size
        self.query_size = query_size
        self.num_workers = num_workers
        self.seed = seed
        self.train = None
        self.test = None

    @staticmethod
    def _compute_date_splits(final_df: pl.DataFrame):
        unique_dates = final_df.select(
            pl.col("date").dt.date().alias("d")
        ).unique().sort("d")["d"].to_list()

        n_dates = len(unique_dates)
        if n_dates < 3:
            raise ValueError(f"Need at least 3 unique dates for train/val/test split, got {n_dates}.")

        train_date_count = max(1, int(n_dates * 0.85))

        train_dates = unique_dates[:train_date_count]
        test_dates = unique_dates[train_date_count:]

        return train_dates, test_dates

    @staticmethod
    def _group_by_date(frame: pl.DataFrame):
        x_by_date = []
        y_by_date = []
        greek_labels_by_date = []

        frame = frame.with_columns(date_only=pl.col("date").dt.date())

        for _, group in frame.group_by("date_only", maintain_order=True):
            if group.height == 0:
                continue

            x = group.select(SURFACE_FEATURE_COLUMNS).to_torch(dtype=pl.Float32)
            y = group.select("Price").to_torch(dtype=pl.Float32)
            greek_labels = (
                group
                .select(GREEK_LABEL_COLUMNS)
                .with_columns((pl.col("theta") * 365.0).alias("theta"))
                .to_torch(dtype=pl.Float32)
            )

            x_by_date.append(x)
            y_by_date.append(y)
            greek_labels_by_date.append(greek_labels)

        if not x_by_date:
            raise ValueError("No date groups remain after preprocessing.")

        return x_by_date, y_by_date, greek_labels_by_date

    def setup(self, stage: str) -> None:
        if self.train is not None and self.test is not None:
            return

        csv_files = f"{self.folder}/*.csv"
        if not glob.glob(csv_files):
            raise ValueError(f"No csv files found in folder {self.folder}.")

        lf = pl.scan_csv(csv_files, try_parse_dates=True)
        schema = getattr(lf, "collect_schema", lambda: lf.schema)()

        if schema["date"] == pl.String:
            lf = lf.with_columns(pl.col("date").str.to_datetime(strict=False))

        lf = lf.filter(
            (pl.col("T") < 365) & (pl.col("T") > 14)
        ).with_columns(
            M=pl.col("S") / pl.col("K"),
            vix=pl.col("vix") / 100.0,
            T=pl.col("T") / 365.0
        ).filter(
            (pl.col("M") >= 0.7) & (pl.col("M") <= 1.15)
        )

        base_required = ["date", "Price", "S", "K", "T", "vix", "dividend_yield", "rate", "delta", "gamma", "theta"]
        lf = lf.drop_nulls(subset=base_required)

        final_df = lf.collect().sort("date")

        train_dates, test_dates = self._compute_date_splits(final_df)
        date_expr = pl.col("date").dt.date()

        train_df = final_df.filter(date_expr.is_in(train_dates))
        test_df = final_df.filter(date_expr.is_in(test_dates))

        train_x, train_y, train_greeks = self._group_by_date(train_df)
        test_x, test_y, test_greeks = self._group_by_date(test_df)

        self.train = DateGroupedOptionsDataset(
            train_x, train_y, train_greeks,
            context_size=self.context_size, query_size=self.query_size,
            deterministic=False, seed=self.seed,
        )

        self.test = DateGroupedOptionsDataset(
            test_x, test_y, test_greeks,
            context_size=self.context_size, query_size=self.query_size,
            deterministic=True, seed=self.seed + 20_000,
        )

        train_rows = sum(split_x.shape[0] for split_x in train_x)
        test_rows = sum(split_x.shape[0] for split_x in test_x)
        print(f"Date split -> Train: {len(train_dates)} | Test: {len(test_dates)}")
        print(f"Rows -> Train: {train_rows} | Test: {test_rows}")
        print(f"Surface batching -> Dates/Batch: {self.batch_size_dates} | Context: {self.context_size} | Query: {self.query_size}")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size_dates,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size_dates,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

class DateGroupedOptionsDataset(Dataset):
    def __init__(
            self,
            x_by_date: List[torch.Tensor],
            y_by_date: List[torch.Tensor],
            greek_labels_by_date: List[torch.Tensor],
            context_size: int,
            query_size: int,
            deterministic: bool = False,
            seed: int = 42,
    ):
        if not (len(x_by_date) == len(y_by_date) == len(greek_labels_by_date)):
            raise ValueError("x_by_date, y_by_date, and greek_labels_by_date must have the same length.")
        self.x_by_date = x_by_date
        self.y_by_date = y_by_date
        self.greek_labels_by_date = greek_labels_by_date
        self.context_size = context_size
        self.query_size = query_size
        self.deterministic = deterministic
        self.seed = seed

    def __len__(self) -> int:
        return len(self.x_by_date)

    def _make_rng(self, item: int) -> np.random.Generator:
        gen = torch.Generator()
        if self.deterministic:
            gen.manual_seed(self.seed + item)
        else:
            gen.seed()
        return gen

    @staticmethod
    def _sample_indices(
            row_count: int, sample_size: int, rng: torch.Generator, replace: bool
    ) -> torch.Tensor:
        if replace:
            return torch.randint(0, row_count, (sample_size,), generator=rng)
        return torch.randperm(row_count, generator=rng)[:sample_size]

    def _select_context_query_indices(
            self, row_count: int, rng: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        required_rows = self.context_size + self.query_size
        if row_count >= required_rows:
            chosen = self._sample_indices(row_count, required_rows, rng, replace=False)
            return chosen[:self.context_size], chosen[self.context_size:]

        # When rows are limited, keep context/query disjoint and let masks handle padding.
        chosen = self._sample_indices(row_count, row_count, rng, replace=False)
        context_limit = self.context_size
        if self.query_size > 0 and row_count > 1:
            context_limit = min(context_limit, row_count - 1)
        context_count = min(context_limit, row_count)
        query_count = min(self.query_size, row_count - context_count)

        context_idx = chosen[:context_count]
        query_idx = chosen[context_count:context_count + query_count]
        return context_idx, query_idx

    @staticmethod
    def _pad_selected_rows(
            values: torch.Tensor,
            selected_idx: torch.Tensor,
            target_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padded = values.new_zeros((target_size, *values.shape[1:]))
        mask = torch.zeros(target_size, dtype=torch.bool, device=values.device)

        selected_count = int(selected_idx.shape[0])
        if selected_count > 0:
            padded[:selected_count] = values[selected_idx]
            mask[:selected_count] = True

        return padded, mask

    def __getitem__(self, item: int):
        x = self.x_by_date[item]
        y = self.y_by_date[item]
        greek_labels = self.greek_labels_by_date[item]
        row_count = x.shape[0]
        if row_count == 0:
            raise ValueError(f"Date-group at index {item} has no rows.")

        rng = self._make_rng(item)
        context_idx, query_idx = self._select_context_query_indices(row_count, rng)

        context_idx_t = context_idx.long()
        query_idx_t = query_idx.long()

        context_x, context_mask = self._pad_selected_rows(x, context_idx_t, self.context_size)
        context_y, _ = self._pad_selected_rows(y, context_idx_t, self.context_size)
        query_x, query_mask = self._pad_selected_rows(x, query_idx_t, self.query_size)
        query_y, _ = self._pad_selected_rows(y, query_idx_t, self.query_size)
        query_greek_labels, _ = self._pad_selected_rows(greek_labels, query_idx_t, self.query_size)

        return {
            "context_x": context_x,
            "context_y": context_y,
            "query_x": query_x,
            "query_y": query_y,
            "query_greeks": {
                "delta": query_greek_labels[:, 0:1],
                "gamma": query_greek_labels[:, 1:2],
                "theta": query_greek_labels[:, 2:3],
            },
            "context_mask": context_mask,
            "query_mask": query_mask,
        }
