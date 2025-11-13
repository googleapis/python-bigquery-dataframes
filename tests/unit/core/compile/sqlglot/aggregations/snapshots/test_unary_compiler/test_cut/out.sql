WITH `bfcte_0` AS (
  SELECT
    `int64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    CASE
      WHEN `int64_col` <= MIN(`int64_col`) OVER () + (
        1 * IEEE_DIVIDE(MAX(`int64_col`) OVER () - MIN(`int64_col`) OVER (), 3)
      )
      THEN STRUCT(
        (
          MIN(`int64_col`) OVER () + (
            0 * IEEE_DIVIDE(MAX(`int64_col`) OVER () - MIN(`int64_col`) OVER (), 3)
          )
        ) - (
          (
            MAX(`int64_col`) OVER () - MIN(`int64_col`) OVER ()
          ) * 0.001
        ) AS `left_exclusive`,
        MIN(`int64_col`) OVER () + (
          1 * IEEE_DIVIDE(MAX(`int64_col`) OVER () - MIN(`int64_col`) OVER (), 3)
        ) + 0 AS `right_inclusive`
      )
      WHEN `int64_col` <= MIN(`int64_col`) OVER () + (
        2 * IEEE_DIVIDE(MAX(`int64_col`) OVER () - MIN(`int64_col`) OVER (), 3)
      )
      THEN STRUCT(
        (
          MIN(`int64_col`) OVER () + (
            1 * IEEE_DIVIDE(MAX(`int64_col`) OVER () - MIN(`int64_col`) OVER (), 3)
          )
        ) - 0 AS `left_exclusive`,
        MIN(`int64_col`) OVER () + (
          2 * IEEE_DIVIDE(MAX(`int64_col`) OVER () - MIN(`int64_col`) OVER (), 3)
        ) + 0 AS `right_inclusive`
      )
      WHEN `int64_col` IS NOT NULL
      THEN STRUCT(
        (
          MIN(`int64_col`) OVER () + (
            2 * IEEE_DIVIDE(MAX(`int64_col`) OVER () - MIN(`int64_col`) OVER (), 3)
          )
        ) - 0 AS `left_exclusive`,
        MIN(`int64_col`) OVER () + (
          3 * IEEE_DIVIDE(MAX(`int64_col`) OVER () - MIN(`int64_col`) OVER (), 3)
        ) + 0 AS `right_inclusive`
      )
    END AS `bfcol_1`,
    CASE
      WHEN `int64_col` > 0 AND `int64_col` <= 1
      THEN STRUCT(0 AS left_exclusive, 1 AS right_inclusive)
      WHEN `int64_col` > 1 AND `int64_col` <= 2
      THEN STRUCT(1 AS left_exclusive, 2 AS right_inclusive)
    END AS `bfcol_2`,
    CASE
      WHEN `int64_col` < MIN(`int64_col`) OVER () + (
        1 * IEEE_DIVIDE(MAX(`int64_col`) OVER () - MIN(`int64_col`) OVER (), 3)
      )
      THEN 'a'
      WHEN `int64_col` < MIN(`int64_col`) OVER () + (
        2 * IEEE_DIVIDE(MAX(`int64_col`) OVER () - MIN(`int64_col`) OVER (), 3)
      )
      THEN 'b'
      WHEN `int64_col` IS NOT NULL
      THEN 'c'
    END AS `bfcol_3`,
    CASE
      WHEN `int64_col` > 0 AND `int64_col` <= 1
      THEN 0
      WHEN `int64_col` > 1 AND `int64_col` <= 2
      THEN 1
    END AS `bfcol_4`
  FROM `bfcte_0`
)
SELECT
  `bfcol_1` AS `int_bins`,
  `bfcol_2` AS `interval_bins`,
  `bfcol_3` AS `int_bins_labels`,
  `bfcol_4` AS `interval_bins_labels`
FROM `bfcte_1`