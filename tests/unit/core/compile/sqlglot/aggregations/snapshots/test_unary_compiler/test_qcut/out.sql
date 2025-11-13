WITH `bfcte_0` AS (
  SELECT
    `int64_col`,
    `rowindex`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    *,
    NOT `int64_col` IS NULL AS `bfcol_4`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *,
    IF(
      `int64_col` IS NULL,
      NULL,
      CAST(GREATEST(
        CEIL(PERCENT_RANK() OVER (PARTITION BY `bfcol_4` ORDER BY `int64_col` ASC) * 4) - 1,
        0
      ) AS INT64)
    ) AS `bfcol_5`
  FROM `bfcte_1`
), `bfcte_3` AS (
  SELECT
    *,
    NOT `int64_col` IS NULL AS `bfcol_9`
  FROM `bfcte_2`
), `bfcte_4` AS (
  SELECT
    *,
    CASE
      WHEN PERCENT_RANK() OVER (PARTITION BY `bfcol_9` ORDER BY `int64_col` ASC) < 0
      THEN NULL
      WHEN PERCENT_RANK() OVER (PARTITION BY `bfcol_9` ORDER BY `int64_col` ASC) <= 0.25
      THEN 0
      WHEN PERCENT_RANK() OVER (PARTITION BY `bfcol_9` ORDER BY `int64_col` ASC) <= 0.5
      THEN 1
      WHEN PERCENT_RANK() OVER (PARTITION BY `bfcol_9` ORDER BY `int64_col` ASC) <= 0.75
      THEN 2
      WHEN PERCENT_RANK() OVER (PARTITION BY `bfcol_9` ORDER BY `int64_col` ASC) <= 1
      THEN 3
      ELSE NULL
    END AS `bfcol_10`
  FROM `bfcte_3`
)
SELECT
  `rowindex`,
  `int64_col`,
  `bfcol_5` AS `qcut_w_int`,
  `bfcol_10` AS `qcut_w_list`
FROM `bfcte_4`