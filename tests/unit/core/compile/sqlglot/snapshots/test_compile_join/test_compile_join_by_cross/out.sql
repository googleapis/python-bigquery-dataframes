WITH `bfcte_1` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64>>[STRUCT('foo', 1), STRUCT('bar', 2), STRUCT('baz', 3), STRUCT('foo', 5)])
), `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_2` STRING, `bfcol_3` INT64>>[STRUCT('foo', 5), STRUCT('bar', 6), STRUCT('baz', 7), STRUCT('foo', 8)])
), `bfcte_2` AS (
  SELECT
    `bfcol_2` AS `bfcol_4`,
    `bfcol_3` AS `bfcol_5`
  FROM `bfcte_0`
), `bfcte_3` AS (
  SELECT
    *
  FROM `bfcte_1`
  CROSS JOIN `bfcte_2`
)
SELECT
  `bfcol_0` AS `lkey`,
  `bfcol_1` AS `value_x`,
  `bfcol_4` AS `rkey`,
  `bfcol_5` AS `value_y`
FROM `bfcte_3`