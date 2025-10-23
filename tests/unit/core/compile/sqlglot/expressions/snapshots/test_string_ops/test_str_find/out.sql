WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS STRING), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    INSTR(`bfcol_0`, 'e', 1) - 1 AS `bfcol_2`,
    INSTR(`bfcol_0`, 'e', 3) - 1 AS `bfcol_3`,
    INSTR(SUBSTRING(`bfcol_0`, 1, 5), 'e') - 1 AS `bfcol_4`,
    INSTR(SUBSTRING(`bfcol_0`, 3, 3), 'e') - 1 AS `bfcol_5`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `none_none`,
  `bfcol_3` AS `start_none`,
  `bfcol_4` AS `none_end`,
  `bfcol_5` AS `start_end`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST