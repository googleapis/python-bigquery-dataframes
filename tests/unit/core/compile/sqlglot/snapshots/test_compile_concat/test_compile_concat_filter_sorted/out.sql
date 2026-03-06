WITH `bfcte_0` AS (
  SELECT
    `float64_col` AS `bfcol_23`,
    `int64_col` AS `bfcol_24`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
), `bfcte_1` AS (
  SELECT
    `float64_col` AS `bfcol_2`,
    `int64_col` AS `bfcol_3`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
), `bfcte_2` AS (
  SELECT
    `float64_col` AS `bfcol_34`,
    `int64_too` AS `bfcol_35`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  WHERE
    `bool_col`
), `bfcte_3` AS (
  SELECT
    `float64_col` AS `bfcol_13`,
    `int64_too` AS `bfcol_14`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  WHERE
    `bool_col`
), `bfcte_4` AS (
  SELECT
    `bfcte_0`.*
  FROM `bfcte_0`
), `bfcte_5` AS (
  SELECT
    `bfcte_2`.*
  FROM `bfcte_2`
)
SELECT
  `bfcol_42` AS `float64_col`,
  `bfcol_43` AS `int64_col`
FROM (
  SELECT
    `bfcol_6` AS `bfcol_42`,
    `bfcol_7` AS `bfcol_43`,
    `bfcol_8` AS `bfcol_44`,
    `bfcol_9` AS `bfcol_45`
  FROM (
    (
      SELECT
        `bfcol_2` AS `bfcol_6`,
        `bfcol_3` AS `bfcol_7`,
        0 AS `bfcol_8`,
        ROW_NUMBER() OVER (ORDER BY `bfcol_3` ASC NULLS LAST) - 1 AS `bfcol_9`
      FROM `bfcte_1`
    )
    UNION ALL
    (
      SELECT
        `bfcol_13` AS `bfcol_17`,
        `bfcol_14` AS `bfcol_18`,
        1 AS `bfcol_19`,
        ROW_NUMBER() OVER () - 1 AS `bfcol_20`
      FROM `bfcte_3`
    )
    UNION ALL
    (
      SELECT
        `bfcol_23` AS `bfcol_27`,
        `bfcol_24` AS `bfcol_28`,
        2 AS `bfcol_29`,
        ROW_NUMBER() OVER (ORDER BY `bfcol_24` ASC NULLS LAST) - 1 AS `bfcol_30`
      FROM `bfcte_4`
    )
    UNION ALL
    (
      SELECT
        `bfcol_34` AS `bfcol_38`,
        `bfcol_35` AS `bfcol_39`,
        3 AS `bfcol_40`,
        ROW_NUMBER() OVER () - 1 AS `bfcol_41`
      FROM `bfcte_5`
    )
  )
)
ORDER BY
  `bfcol_44` ASC NULLS LAST,
  `bfcol_45` ASC NULLS LAST