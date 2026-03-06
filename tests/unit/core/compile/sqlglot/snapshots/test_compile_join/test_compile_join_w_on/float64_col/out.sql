WITH `bfcte_0` AS (
  SELECT
    `float64_col` AS `bfcol_4`,
    `rowindex` AS `bfcol_5`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
), `bfcte_1` AS (
  SELECT
    `float64_col` AS `bfcol_0`,
    `rowindex` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
), `bfcte_2` AS (
  SELECT
    `bfcte_0`.*
  FROM `bfcte_0`
)
SELECT
  (
    SELECT
      `bfcol_2` AS `rowindex_x`,
      `bfcol_3` AS `float64_col`,
      `bfcol_6` AS `rowindex_y`
    FROM (
      SELECT
        `bfcol_1` AS `bfcol_2`,
        `bfcol_0` AS `bfcol_3`
      FROM `bfcte_1`
    )
    INNER JOIN (
      SELECT
        `bfcol_5` AS `bfcol_6`,
        `bfcol_4` AS `bfcol_7`
      FROM `bfcte_2`
    )
      ON IF(IS_NAN(`bfcol_3`), 2, COALESCE(`bfcol_3`, 0)) = IF(IS_NAN(`bfcol_7`), 2, COALESCE(`bfcol_7`, 0))
      AND IF(IS_NAN(`bfcol_3`), 3, COALESCE(`bfcol_3`, 1)) = IF(IS_NAN(`bfcol_7`), 3, COALESCE(`bfcol_7`, 1))
  ).*
FROM (
  SELECT
    `bfcol_2` AS `rowindex_x`,
    `bfcol_3` AS `float64_col`,
    `bfcol_6` AS `rowindex_y`
  FROM (
    SELECT
      `bfcol_1` AS `bfcol_2`,
      `bfcol_0` AS `bfcol_3`
    FROM `bfcte_1`
  )
  INNER JOIN (
    SELECT
      `bfcol_5` AS `bfcol_6`,
      `bfcol_4` AS `bfcol_7`
    FROM `bfcte_2`
  )
    ON IF(IS_NAN(`bfcol_3`), 2, COALESCE(`bfcol_3`, 0)) = IF(IS_NAN(`bfcol_7`), 2, COALESCE(`bfcol_7`, 0))
    AND IF(IS_NAN(`bfcol_3`), 3, COALESCE(`bfcol_3`, 1)) = IF(IS_NAN(`bfcol_7`), 3, COALESCE(`bfcol_7`, 1))
)