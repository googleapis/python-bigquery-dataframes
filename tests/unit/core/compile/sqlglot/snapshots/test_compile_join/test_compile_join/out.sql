SELECT
  `bfcol_7` AS `int64_col`,
  `bfcol_5` AS `int64_too`
FROM (
  SELECT
    `rowindex` AS `bfcol_6`,
    `int64_col` AS `bfcol_7`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)
LEFT JOIN (
  SELECT
    `int64_col` AS `bfcol_4`,
    `int64_too` AS `bfcol_5`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
)
  ON COALESCE(`bfcol_6`, 0) = COALESCE(`bfcol_4`, 0)
  AND COALESCE(`bfcol_6`, 1) = COALESCE(`bfcol_4`, 1)