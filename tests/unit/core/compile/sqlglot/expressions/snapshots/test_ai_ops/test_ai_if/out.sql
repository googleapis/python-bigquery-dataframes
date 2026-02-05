WITH `bfcte_0` AS (
  SELECT
    `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  AI.IF(
    prompt => (`string_col`, ' is the same as ', `string_col`),
    connection_id => 'bigframes-dev.us.bigframes-default-connection'
  ) AS `result`
FROM `bfcte_0`