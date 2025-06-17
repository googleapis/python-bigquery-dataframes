WITH `bfcte_0` AS (
  SELECT
    `int64_col` AS `bfcol_0`,
    `rowindex` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    `bfcol_1` AS `bfcol_2`,
    `bfcol_1` AS `bfcol_3`,
    `bfcol_0` AS `bfcol_4`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `rowindex`,
  `bfcol_3` AS `rowindex_1`,
  `bfcol_4` AS `int64_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_4` ASC NULLS LAST
LIMIT 10