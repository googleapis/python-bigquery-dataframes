SELECT
  `bfcol_2` AS `rowindex`,
  `bfcol_5` AS `int64_col`
FROM (
  SELECT
    (
      SELECT
        `rowindex` AS `bfcol_2`,
        `int64_col` AS `bfcol_3`
      FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
    ).*,
    EXISTS(
      SELECT
        1
      FROM (
        SELECT
          `int64_too` AS `bfcol_4`
        FROM (
          SELECT
            `int64_too`
          FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
          GROUP BY
            `int64_too`
        )
      ) AS `bft_1`
      WHERE
        COALESCE(
          (
            SELECT
              `rowindex` AS `bfcol_2`,
              `int64_col` AS `bfcol_3`
            FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
          ).`bfcol_3`,
          0
        ) = COALESCE(`bft_1`.`bfcol_4`, 0)
        AND COALESCE(
          (
            SELECT
              `rowindex` AS `bfcol_2`,
              `int64_col` AS `bfcol_3`
            FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
          ).`bfcol_3`,
          1
        ) = COALESCE(`bft_1`.`bfcol_4`, 1)
    ) AS `bfcol_5`
  FROM (
    SELECT
      `rowindex` AS `bfcol_2`,
      `int64_col` AS `bfcol_3`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  )
)