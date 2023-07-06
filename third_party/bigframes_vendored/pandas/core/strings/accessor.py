class StringMethods:
    """
    Vectorized string functions for Series and Index.

    NAs stay NA unless handled otherwise by a particular method.
    Patterned after Python's string methods, with some inspiration from
    R's stringr package.
    """

    def find(self, sub, start: int = 0, end=None):
        """Return lowest indexes in each strings in the Series/Index.

        Each of returned indexes corresponds to the position where the
        substring is fully contained between [start:end]. Return -1 on
        failure. Equivalent to standard :meth:`str.find`.

        Args:
            sub:
                Substring being searched.
            start:
                Left edge index.
            end:
                Right edge index.

        Returns:
            Series or Index of int.
        """

        raise NotImplementedError("abstract method")

    def len(self):
        """Compute the length of each element in the Series/Index.

        The element may be a sequence (such as a string, tuple or list) or a collection
        (such as a dictionary).

        Returns:
            Series or Index of int
            A Series or Index of integer values indicating the length of each
            element in the Series or Index.
        """

        raise NotImplementedError("abstract method")
