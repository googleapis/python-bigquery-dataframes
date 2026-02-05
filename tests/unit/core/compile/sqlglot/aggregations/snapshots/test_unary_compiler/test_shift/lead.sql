WITH `bfcte_0` AS (
  SELECT
    `int64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  LEAD(`int64_col`, 1) OVER (ORDER BY `int64_col` ASC) AS `lead`
FROM `bfcte_0`