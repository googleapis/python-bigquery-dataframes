WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` STRING, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS STRING), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    OBJ.MAKE_REF(`bfcol_1`, 'bigframes-dev.test-region.bigframes-default-connection') AS `bfcol_6`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *,
    OBJ.FETCH_METADATA(`bfcol_6`) AS `bfcol_10`
  FROM `bfcte_1`
), `bfcte_3` AS (
  SELECT
    *,
    `bfcol_10`.`version` AS `bfcol_14`
  FROM `bfcte_2`
)
SELECT
  `bfcol_0` AS `rowindex`,
  `bfcol_14` AS `version`
FROM `bfcte_3`
ORDER BY
  `bfcol_2` ASC NULLS LAST