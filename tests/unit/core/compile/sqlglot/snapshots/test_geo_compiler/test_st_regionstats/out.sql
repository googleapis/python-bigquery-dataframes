WITH `bfcte_1` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` STRING>>[STRUCT(0, 'POINT(1 1)')])
), `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_2` INT64, `bfcol_3` STRING>>[STRUCT(0, 'raster_uri')])
), `bfcte_2` AS (
  SELECT
    `bfcol_2` AS `bfcol_4`,
    `bfcol_3` AS `bfcol_5`
  FROM `bfcte_0`
), `bfcte_3` AS (
  SELECT
    *
  FROM `bfcte_1`
  LEFT JOIN `bfcte_2`
    ON COALESCE(`bfcol_0`, 0) = COALESCE(`bfcol_4`, 0)
    AND COALESCE(`bfcol_0`, 1) = COALESCE(`bfcol_4`, 1)
), `bfcte_4` AS (
  SELECT
    *,
    ST_REGIONSTATS(`bfcol_1`, `bfcol_5`, 'band1', _(OPTIONS, JSON('{"scale": 100}'))) AS `bfcol_8`
  FROM `bfcte_3`
), `bfcte_5` AS (
  SELECT
    *,
    `bfcol_8`.`min` AS `bfcol_10`,
    `bfcol_8`.`max` AS `bfcol_11`,
    `bfcol_8`.`sum` AS `bfcol_12`,
    `bfcol_8`.`count` AS `bfcol_13`,
    `bfcol_8`.`mean` AS `bfcol_14`,
    `bfcol_8`.`area` AS `bfcol_15`
  FROM `bfcte_4`
)
SELECT
  `bfcol_10` AS `min`,
  `bfcol_11` AS `max`,
  `bfcol_12` AS `sum`,
  `bfcol_13` AS `count`,
  `bfcol_14` AS `mean`,
  `bfcol_15` AS `area`
FROM `bfcte_5`