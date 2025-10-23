WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS STRING), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    ENDS_WITH(`bfcol_0`, 'ab') AS `bfcol_2`,
    ENDS_WITH(`bfcol_0`, 'ab') OR ENDS_WITH(`bfcol_0`, 'cd') AS `bfcol_3`,
    FALSE AS `bfcol_4`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `single`,
  `bfcol_3` AS `double`,
  `bfcol_4` AS `empty`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST