WITH `bfcte_0` AS (
  SELECT
    `rowindex`,
    `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  `rowindex` AS `rowindex`,
  OBJ.FETCH_METADATA(
    OBJ.MAKE_REF(`string_col`, 'bigframes-dev.test-region.bigframes-default-connection')
  ).`version` AS `version`
FROM `bfcte_0`