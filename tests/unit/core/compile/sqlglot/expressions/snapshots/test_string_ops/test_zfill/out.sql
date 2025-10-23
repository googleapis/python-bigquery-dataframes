WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS STRING), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    CASE
      WHEN SUBSTRING(`bfcol_0`, 1, 1) = '-'
      THEN CONCAT('-', LPAD(SUBSTRING(`bfcol_0`, 1), 9, '0'))
      ELSE LPAD(`bfcol_0`, 10, '0')
    END AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `string_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST