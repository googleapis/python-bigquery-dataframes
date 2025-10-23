WITH `bfcte_1` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` INT64, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS INT64), 0)])
), `bfcte_2` AS (
  SELECT
    `bfcol_0` AS `bfcol_3`,
    `bfcol_1` AS `bfcol_4`,
    `bfcol_2` AS `bfcol_5`
  FROM `bfcte_1`
), `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_6` INT64>>[STRUCT(CAST(NULL AS INT64))])
), `bfcte_3` AS (
  SELECT
    `bfcte_2`.*,
    EXISTS(
      SELECT
        1
      FROM (
        SELECT
          `bfcol_6`
        FROM `bfcte_0`
        GROUP BY
          `bfcol_6`
      ) AS `bft_0`
      WHERE
        COALESCE(`bfcte_2`.`bfcol_4`, 0) = COALESCE(`bft_0`.`bfcol_6`, 0)
        AND COALESCE(`bfcte_2`.`bfcol_4`, 1) = COALESCE(`bft_0`.`bfcol_6`, 1)
    ) AS `bfcol_7`
  FROM `bfcte_2`
)
SELECT
  `bfcol_3` AS `rowindex`,
  `bfcol_7` AS `rowindex_2`
FROM `bfcte_3`
ORDER BY
  `bfcol_5` ASC NULLS LAST