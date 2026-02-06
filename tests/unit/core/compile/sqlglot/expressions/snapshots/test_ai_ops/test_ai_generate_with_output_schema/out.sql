WITH `bfcte_0` AS (
  SELECT
    `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  AI.GENERATE(
    prompt => (`string_col`, ' is the same as ', `string_col`),
    endpoint => 'gemini-2.5-flash',
    request_type => 'SHARED',
    output_schema => 'x INT64, y FLOAT64'
  ) AS `result`
FROM `bfcte_0`