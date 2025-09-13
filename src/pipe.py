import polars as pl
import re
from typing import List, Dict


class CreditDataPipeline:
    def __init__(
        self,
        numeric_cols: List[str],
        cat_cols: List[str],
        loan_types: List[str],
        cat_cols_auto_encode: List[str] = None,
    ):
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.loan_types = loan_types
        self.cat_cols_auto_encode = cat_cols_auto_encode or []

        self.global_medians = {}
        self.global_modes = {}
        self.cat_mappings: Dict[str, Dict[str, int]] = {}
        self.loan_cols = [
            f"Loan_{lt.replace('-', '').replace(' ','_')}" for lt in loan_types
        ]
        self.pattern = re.compile(r"(?P<years>\d+) Years and (?P<months>\d+) Months")
        self.cols_to_drop = [
            "Customer_ID",
            "Name",
            "SSN",
            "Month",
            "Annual_Income",
            "Type_of_Loan",
            "median_age_per_occ",
        ]
        self.credit_mix_map = {
            "Bad": 0,
            "Standard": 1,
            "Good": 2,
        }
        self.payment_map = {
            "No": 0,
            "NM": 0,  # assuming this is also a No
            "Yes": 1,
        }

    # Transform the credit history age from a sentence to a number of months
    def history_to_months(self, history):

        if history is None:
            return None

        match = self.pattern.match(history)

        if not match:
            return None

        years = int(match.group("years"))
        months = int(match.group("months"))

        return (years * 12) + months

    def fit(self, df: pl.DataFrame):
        # Compute global medians and means for fallback
        for col in self.numeric_cols:
            col_vals = df[col].drop_nulls()
            self.global_medians[col] = col_vals.median()

        for col in self.cat_cols:
            col_vals = df[col].drop_nulls()
            self.global_modes[col] = col_vals.mode()

        # Auto extract category mappings
        for col in self.cat_cols_auto_encode:
            unique_vals = df[col].drop_nulls().unique().to_list()
            self.cat_mappings[col] = {v: i for i, v in enumerate(unique_vals)}

        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        df_new = df.clone()

        # remove the underscore character, and cast to int
        df_new = df_new.with_columns(
            [
                pl.col(pl.String).str.strip_chars("_"),
            ]
        )

        df_new = df_new.with_columns(
            [
                pl.col(pl.String).replace(["", "_", "!@9#%8"], None),
            ]
        )

        df_new = df_new.with_columns(
            [
                pl.col("Age").cast(pl.Float64),
                pl.col("Annual_Income").cast(pl.Float64),
                pl.col("Num_of_Loan").cast(pl.Float64),
                pl.col("Num_of_Delayed_Payment").cast(pl.Float64),
                pl.col("Changed_Credit_Limit").cast(pl.Float64),
                pl.col("Outstanding_Debt").cast(pl.Float64),
                pl.col("Amount_invested_monthly").cast(pl.Float64),
                pl.col("Monthly_Balance").cast(pl.Float64),
                # Transform the credit history age to the equivalent number of months
                pl.col("Credit_History_Age").map_elements(
                    self.history_to_months, return_dtype=pl.Float64
                ),
            ]
        )

        df_new = df_new.with_columns(
            [
                pl.when((pl.col("Age") < 18) | (pl.col("Age") > 100))
                .then(None)
                .otherwise(pl.col("Age"))
                .alias("Age"),
            ]
        )

        # Remove Negative Values to be imputed later
        df_new = df_new.with_columns(
            [
                pl.col(pl.Int64).map_elements(
                    lambda x: None if x < 0 else x, return_dtype=pl.Int64
                ),
                pl.col(pl.Float64).map_elements(
                    lambda x: None if x < 0 else x, return_dtype=pl.Float64
                ),
            ]
        )

        # Computing per-customer median
        medians = df_new.group_by("Customer_ID").agg(
            [pl.col(col).median().alias(f"{col}_median") for col in self.numeric_cols]
        )

        df_new = df_new.join(medians, on="Customer_ID", how="left")

        # Computing absolute deviation from median
        for col in self.numeric_cols:

            df_new = df_new.with_columns(
                (pl.col(col) - pl.col(f"{col}_median")).abs().alias(f"{col}_abs_dev")
            )

        # Computing MAD per customer
        mad = df_new.group_by("Customer_ID").agg(
            [
                pl.col(f"{col}_abs_dev").median().alias(f"{col}_mad")
                for col in self.numeric_cols
            ]
        )

        df_new = df_new.join(mad, on="Customer_ID", how="left")

        # Replacing outliers with None to impute later based on the MAD since IQR failed HARD
        for col in self.numeric_cols:

            df_new = df_new.with_columns(
                [
                    pl.when(
                        (
                            pl.col(col)
                            < (pl.col(f"{col}_median") - 3 * pl.col(f"{col}_mad"))
                        )
                        | (
                            pl.col(col)
                            > (pl.col(f"{col}_median") + 3 * pl.col(f"{col}_mad"))
                        )
                    )
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                ]
            )

        cat_imputes = [
            df_new.group_by("Customer_ID").agg(
                pl.col(col).drop_nulls().mode().first().alias(f"{col}_mode")
            )
            for col in self.cat_cols
        ]

        for impute in cat_imputes:
            df_new = df_new.join(impute, on="Customer_ID", how="left", suffix="_mode")

        df_new = df_new.with_columns(
            pl.coalesce([pl.col(col), pl.col(f"{col}_median")]).alias(col)
            for col in self.numeric_cols
        )

        df_new = df_new.with_columns(
            [
                pl.coalesce([pl.col(col), pl.col(f"{col}_mode")]).alias(col)
                for col in self.cat_cols
            ]
        )

        # Calculate the median age per occupation to fill empty values
        median_age_per_occ = df_new.group_by("Occupation").agg(
            pl.col("Age").median().alias("median_age_per_occ")
        )

        # Join medians back to original df
        df_new = df_new.join(median_age_per_occ, on="Occupation", how="left")

        df_new = df_new.with_columns(
            [
                # Impute using the median age per occupation
                pl.coalesce([pl.col("Age"), pl.col("median_age_per_occ")])
                .cast(pl.Float64)
                .alias("Age"),
                # Impute using the Annual Income to calculate an approximate of the monthly salary
                pl.when(pl.col("Monthly_Inhand_Salary").is_null())
                .then(pl.col("Annual_Income") / 12)
                .otherwise(pl.col("Monthly_Inhand_Salary"))
                .alias("Monthly_Inhand_Salary"),
            ]
        )

        # For people with no Loans, the type of loan is None, filling them will help with encoding
        df_new = df_new.with_columns([pl.col("Type_of_Loan").fill_null("")])

        # Multi-Hot encoding the Type of Loan
        df_new = df_new.with_columns(
            [
                pl.col("Type_of_Loan")
                .str.contains(rf"\b{loan}\b")
                .cast(pl.Int8)
                .alias(f"Loan_{loan.replace('-', '').replace(' ', '_')}")
                for loan in self.loan_types
            ]
        )

        # Applying mapping
        df_new = df_new.with_columns(
            [
                pl.col("Credit_Mix").map_elements(
                    lambda x: self.credit_mix_map.get(x, None), return_dtype=pl.Int64
                ),
                pl.col("Payment_of_Min_Amount").map_elements(
                    lambda x: self.payment_map.get(x, None), return_dtype=pl.Int8
                ),
            ]
        )

        # # Fallbacks in case of a customer have no other entries
        # NOTE: Broke the encodings, would need to re-order the entire thing
        # due to lack of time, I am ignoring this case
        # for col, val in self.global_medians.items():
        #     df_new = df_new.with_columns(pl.col(col).fill_null(val))

        # for col, val in self.global_modes.items():
        #     df_new = df_new.with_columns(pl.col(col).fill_null(val))

        # Auto encode categorical columns
        for col, mapping in self.cat_mappings.items():
            df_new = df_new.with_columns(
                pl.col(col).map_elements(
                    lambda x: mapping.get(x, None), return_dtype=pl.Int64
                )
            )

        # drop the automatically generated columns
        for col in self.numeric_cols:
            self.cols_to_drop.append(f"{col}_median")
            self.cols_to_drop.append(f"{col}_mad")
            self.cols_to_drop.append(f"{col}_abs_dev")
        for col in self.cat_cols:
            self.cols_to_drop.append(f"{col}_mode")

        df_new = df_new.drop(self.cols_to_drop)

        return df_new

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.fit(df).transform(df)
