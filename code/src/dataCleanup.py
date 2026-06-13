import numpy as np
import pandas as pd
import polars as pl

make_map = {
  "vw": "volkswagen",
  "mercedes": "mercedes-benz",
  "mercedes-b": "mercedes-benz",
  "landrover": "land rover",
  "ford truck": "ford",
  "ford tk": "ford",
  "chev truck": "chevrolet",
  "gmc truck": "gmc",
  "mazda tk": "mazda",
  "dodge tk": "dodge",
  "hyundai tk": "hyundai",
}

body_map = {
  "regular-cab": "regular cab",

  "supercrew": "crew cab",
  "supercab": "extended cab",
  "crewmax cab": "crew cab",
  "double cab": "crew cab",

  "access cab": "extended cab",
  "club cab": "extended cab",
  "king cab": "extended cab",
  "xtracab": "extended cab",

  "cab plus": "extended cab",
  "cab plus 4": "extended cab",

  "e-series van": "van",
  "promaster cargo van": "van",
  "transit van": "van",
  "ram van": "van",

  "g sedan": "sedan",
  "g coupe": "coupe",
  "g convertible": "convertible",

  "genesis coupe": "coupe",
  "elantra coupe": "coupe",
  "q60 coupe": "coupe",
  "g37 coupe": "coupe",
  "cts coupe": "coupe",
  "cts-v coupe": "coupe",

  "beetle convertible": "convertible",
  "q60 convertible": "convertible",
  "g37 convertible": "convertible",
  "granturismo convertible": "convertible",
  "granturismo convertible": "convertible",

  "cts wagon": "wagon",
  "cts-v wagon": "wagon",
  "tsx sport wagon": "wagon",
}

def value_counts(df, col, **filters):
  for filter_col, filter_val in filters.items():
    df = df.filter(pl.col(filter_col) == filter_val)

  return (
    df.group_by(col)
      .len()
      .sort("len", descending=True)
  )

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(100)

def cleanup_data(data):
  # Drop missing values in key columns
  df = data.drop_nulls(subset=['make', 'body', 'color', 'condition', 'year', 'saledate', 'sellingprice', 'state', 'transmission'])

  # make, body, transmission, state, color need to be consistent and valid

  # Clean 'make' column
  df = df.with_columns(
    pl.col("make")
    .str.strip_chars()
    .str.to_lowercase()
  )

  df = df.with_columns(
    pl.col("make")
    .replace(make_map)
  )

  # Clean up body
  df = df.with_columns(
    pl.col("body")
      .str.strip_chars()
      .str.to_lowercase()
  )

  df = df.with_columns(
    pl.col("body").replace(body_map)
  )

  # Clean up transmission, convert any not automatic/manual to null and then drop them
  df = df.with_columns(
    pl.when(
      ~pl.col("transmission")
        .str.to_lowercase()
        .is_in(["automatic", "manual"])
    )
    .then(None)
    .otherwise(pl.col("transmission").str.to_lowercase())
    .alias("transmission")
  )

  df = df.drop_nulls(subset=["transmission"])

  # CLean up color
  df = df.with_columns(
    pl.when(pl.col("color") == "—")
      .then(pl.lit("unknown"))
      .otherwise(pl.col("color"))
      .alias("color")
  )

  # Extract 'sale_year' from 'saledate'
  df = df.with_columns(
    sale_year=pl.col('saledate').str.extract(r'(\d{4})').cast(pl.Int64, strict=False)
  )

  # Calculate vehicle age: sale_year - year
  df = df.with_columns(
    vehicle_age=pl.col('sale_year') - pl.col('year')
  )

  # Filter out invalid ages
  df = df.filter(pl.col('vehicle_age') >= 0)

  df = df.filter(pl.col('odometer') < 250000)

  # Potential states for project 
  # │ in    ┆ 3933  │
  # │ ne    ┆ 3914  │
  # │ sc    ┆ 3882  │
  # │ pr    ┆ 2445  │
  # │ la    ┆ 2029  │
  # │ ut    ┆ 1766  │
  # │ ms    ┆ 1730  │
  # │ hi    ┆ 1209  │
  # │ or    ┆ 1049  │
  # │ nm    ┆ 163   │

  # Debug unique values, optionally filter by something
  # print(value_counts(df, "make", state="ut"))
  # print(value_counts(df, "body", state="ut"))
  # print(value_counts(df, "transmission", state="ut"))
  # print(value_counts(df, "state"))
  # print(value_counts(df, "color", state="ut"))
  # print(value_counts(df, "sale_year", state="ut"))
  # print(value_counts(df, "condition", state='ut'))

  # print(df.schema)

  return df
