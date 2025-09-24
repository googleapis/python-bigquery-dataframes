WITH `bfcte_0` AS (
  SELECT
    `int64_col` AS `bfcol_0`,
    `rowindex` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    *,
    DENSE_RANK() OVER (
      ORDER BY `bfcol_0` IS NULL ASC NULLS LAST, `bfcol_0` ASC NULLS LAST
      RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS `bfcol_4`
  FROM `bfcte_0`
)
SELECT
  `bfcol_1` AS `bfuid_col_1`,
  `bfcol_0` AS `int64_col`,
  `bfcol_4` AS `agg_int64`
FROM `bfcte_1`