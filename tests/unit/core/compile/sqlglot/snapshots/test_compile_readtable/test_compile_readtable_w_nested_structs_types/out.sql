WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` STRUCT<name STRING, age INT64, address STRUCT<city STRING, country STRING>>, `bfcol_2` INT64>>[STRUCT(
    CAST(NULL AS INT64),
    STRUCT('' AS `name`, 0 AS `age`, STRUCT('' AS `city`, '' AS `country`) AS `address`),
    0
  )])
)
SELECT
  `bfcol_0` AS `id`,
  `bfcol_0` AS `id_1`,
  `bfcol_1` AS `people`
FROM `bfcte_0`
ORDER BY
  `bfcol_2` ASC NULLS LAST