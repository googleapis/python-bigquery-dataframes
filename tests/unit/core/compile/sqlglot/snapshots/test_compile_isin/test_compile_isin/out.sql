SELECT
  `bfcol_2` AS `rowindex`,
  `bfcol_5` AS `int64_col`
FROM (
  SELECT
    *,
    STRUCT(COALESCE(`bfcol_3`, 0) AS `bfpart1_0`, COALESCE(`bfcol_3`, 1) AS `bfpart2_0`) IN (
      (
        SELECT
          STRUCT(COALESCE(`bfcol_4`, 0) AS `bfpart1_0`, COALESCE(`bfcol_4`, 1) AS `bfpart2_0`)
        FROM (
          SELECT
            `int64_too`
          FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
          GROUP BY
            `int64_too`
        )
      )
    ) AS `bfcol_5`
  FROM (
    SELECT
      `rowindex` AS `bfcol_2`,
      `int64_col` AS `bfcol_3`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  )
)