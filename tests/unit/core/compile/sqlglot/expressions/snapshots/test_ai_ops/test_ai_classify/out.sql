WITH `bfcte_0` AS (
  SELECT
    `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  AI.CLASSIFY(
    input => (`string_col`),
    categories => ['greeting', 'rejection'],
    connection_id => 'bigframes-dev.us.bigframes-default-connection'
  ) AS `result`
FROM `bfcte_0`