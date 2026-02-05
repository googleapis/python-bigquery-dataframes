WITH `bfcte_0` AS (
  SELECT
    `date_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  CAST(FLOOR(
    DATE_DIFF(`date_col`, LAG(`date_col`, 1) OVER (ORDER BY `date_col` ASC NULLS LAST), DAY) * 86400000000
  ) AS INT64) AS `diff_date`
FROM `bfcte_0`