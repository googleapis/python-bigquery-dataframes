SELECT
FROM (
  SELECT
    `bfcol_3` AS `int64_col`,
    `bfcol_7` AS `int64_too`
  FROM (
    SELECT
      `rowindex` AS `bfcol_2`,
      `int64_col` AS `bfcol_3`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  )
  LEFT JOIN (
    SELECT
      `int64_col` AS `bfcol_6`,
      `int64_too` AS `bfcol_7`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  )
    ON COALESCE(`bfcol_2`, 0) = COALESCE(`bfcol_6`, 0)
    AND COALESCE(`bfcol_2`, 1) = COALESCE(`bfcol_6`, 1)
)