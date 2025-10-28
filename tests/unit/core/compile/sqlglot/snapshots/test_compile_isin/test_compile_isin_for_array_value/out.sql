WITH `bfcte_0` AS (
  SELECT
    `int64_col` AS `bfcol_0`,
    `float64_col` AS `bfcol_1`,
    `rowindex` AS `bfcol_2`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    `bfcol_2` AS `bfcol_3`,
    `bfcol_0` AS `bfcol_4`,
    `bfcol_1` AS `bfcol_5`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    `bfcte_1`.*,
    EXISTS(
      SELECT
        1
      FROM (
        SELECT
          `float64_col` AS `bfcol_6`
        FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
      ) AS `bft_0`
      WHERE
        COALESCE(`bfcte_1`.`bfcol_4`, 0) = COALESCE(`bft_0`.`bfcol_6`, 0)
        AND COALESCE(`bfcte_1`.`bfcol_4`, 1) = COALESCE(`bft_0`.`bfcol_6`, 1)
    ) AS `bfcol_7`
  FROM `bfcte_1`
)
SELECT
  `bfcol_3` AS `bfuid_col_1`,
  `bfcol_4` AS `int64_col`,
  `bfcol_5` AS `float64_col`,
  `bfcol_7` AS `bfuid_col_2`
FROM `bfcte_2`