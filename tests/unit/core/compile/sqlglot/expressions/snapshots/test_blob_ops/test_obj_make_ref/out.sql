WITH `bfcte_0` AS (
  SELECT
    `rowindex`,
    `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  `rowindex`,
  OBJ.MAKE_REF(`string_col`, 'bigframes-dev.test-region.bigframes-default-connection') AS `string_col`
FROM `bfcte_0`