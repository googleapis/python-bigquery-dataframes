WITH `bfcte_0` AS (
  SELECT
    `date_col` AS `bfcol_0`,
    `rowindex` AS `bfcol_1`,
    `timestamp_col` AS `bfcol_2`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    *,
    172800000000 AS `bfcol_6`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *,
    172800000000 AS `bfcol_7`
  FROM `bfcte_1`
)
SELECT
  `bfcol_1` AS `rowindex`,
  `bfcol_2` AS `timestamp_col`,
  `bfcol_0` AS `date_col`,
  `bfcol_6` AS `timedelta_mul_numeric`,
  `bfcol_7` AS `numeric_mul_timedelta`
FROM `bfcte_2`