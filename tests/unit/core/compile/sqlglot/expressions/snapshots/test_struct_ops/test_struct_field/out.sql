WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRUCT<name STRING, age INT64, address STRUCT<city STRING, country STRING>>, `bfcol_1` INT64>>[STRUCT(
    STRUCT('' AS `name`, 0 AS `age`, STRUCT('' AS `city`, '' AS `country`) AS `address`),
    0
  )])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_0`.`name` AS `bfcol_2`,
    `bfcol_0`.`name` AS `bfcol_3`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `string`,
  `bfcol_3` AS `int`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST