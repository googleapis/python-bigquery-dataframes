WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS STRING), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    LPAD(`bfcol_0`, GREATEST(LENGTH(`bfcol_0`), 10), '-') AS `bfcol_2`,
    RPAD(`bfcol_0`, GREATEST(LENGTH(`bfcol_0`), 10), '-') AS `bfcol_3`,
    RPAD(
      LPAD(
        `bfcol_0`,
        CAST(SAFE_DIVIDE(GREATEST(LENGTH(`bfcol_0`), 10) - LENGTH(`bfcol_0`), 2) AS INT64) + LENGTH(`bfcol_0`),
        '-'
      ),
      GREATEST(LENGTH(`bfcol_0`), 10),
      '-'
    ) AS `bfcol_4`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `left`,
  `bfcol_3` AS `right`,
  `bfcol_4` AS `both`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST