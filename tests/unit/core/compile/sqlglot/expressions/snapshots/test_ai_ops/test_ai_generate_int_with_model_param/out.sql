WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS STRING), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    AI.GENERATE_INT(
      prompt => (`bfcol_0`, ' is the same as ', `bfcol_0`),
      connection_id => 'bigframes-dev.us.bigframes-default-connection',
      request_type => 'SHARED',
      model_params => JSON '{}'
    ) AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `result`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST