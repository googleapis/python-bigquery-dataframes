SELECT
  `bfcol_2` AS `rowindex`,
  `bfcol_5` AS `rowindex_2`
FROM (
  SELECT
    *,
    `bfcol_3` IN (
      (
        SELECT
          `rowindex_2` AS `bfcol_4`
        FROM (
          SELECT
            `rowindex_2`
          FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
          GROUP BY
            `rowindex_2`
        )
      )
    ) AS `bfcol_5`
  FROM (
    SELECT
      `rowindex` AS `bfcol_2`,
      `rowindex_2` AS `bfcol_3`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  )
)