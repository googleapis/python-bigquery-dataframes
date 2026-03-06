WITH `bfcte_0` AS (
  SELECT
  FROM `bfcte_0`
), `bfcte_1` AS (
  SELECT
  FROM `bfcte_1`
), `bfcte_2` AS (
  SELECT
  FROM `bfcte_2`
)
SELECT
FROM (
  SELECT
    `bfcol_30` AS `rowindex`,
    `bfcol_31` AS `rowindex_1`,
    `bfcol_32` AS `int64_col`,
    `bfcol_33` AS `string_col`
  FROM (
    (
      SELECT
        `bfcol_3` AS `bfcol_9`,
        `bfcol_4` AS `bfcol_10`,
        `bfcol_5` AS `bfcol_11`,
        `bfcol_6` AS `bfcol_12`,
        0 AS `bfcol_13`,
        ROW_NUMBER() OVER () - 1 AS `bfcol_14`
      FROM `bfcte_1`
    )
    UNION ALL
    (
      SELECT
        `bfcol_18` AS `bfcol_24`,
        `bfcol_19` AS `bfcol_25`,
        `bfcol_20` AS `bfcol_26`,
        `bfcol_21` AS `bfcol_27`,
        1 AS `bfcol_28`,
        ROW_NUMBER() OVER () - 1 AS `bfcol_29`
      FROM `bfcte_2`
    )
  )
)
ORDER BY
  `bfcol_34` ASC NULLS LAST,
  `bfcol_35` ASC NULLS LAST