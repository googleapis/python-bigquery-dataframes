WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS STRING), 0)])
), `bfcte_1` AS (
  SELECT
    COALESCE(
      STRING_AGG(
        `bfcol_0`, ','
        ORDER BY
          `bfcol_0` IS NULL ASC,
          `bfcol_0` ASC,
          `bfcol_1` IS NULL ASC,
          `bfcol_1` ASC
      ),
      ''
    ) AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `string_col`
FROM `bfcte_1`