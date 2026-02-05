WITH `bfcte_0` AS (
  SELECT
    `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  AI.GENERATE_BOOL(
    prompt => (`string_col`, ' is the same as ', `string_col`),
    endpoint => 'gemini-2.5-flash',
    request_type => 'SHARED'
  ) AS `result`
FROM `bfcte_0`