# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections
import dataclasses
import functools
import itertools
from typing import (
    Callable,
    cast,
    Dict,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from bigframes.core import (
    agg_expressions,
    expression,
    graphs,
    identifiers,
    nodes,
    subquery_expression,
    window_spec,
)
import bigframes.core.ordered_sets as sets

_MAX_INLINE_COMPLEXITY = 10

T = TypeVar("T")
ExprDomain = Union[window_spec.WindowSpec, Literal["Scalar", "Other"]]


class ExpressionGraph(graphs.DiGraph[nodes.ColumnDef]):
    def __init__(self, column_defs: Sequence[nodes.ColumnDef]):
        # Assumption: All column defs have unique ids
        expr_ids = set(cdef.id for cdef in column_defs)
        self._graph = graphs.DiGraph(
            (expr.id for expr in column_defs),
            (
                (expr.id, child_id)
                for expr in column_defs
                for child_id in expr.expression.column_references
                if child_id in expr_ids
            ),
        )
        self._id_to_cdef = {cdef.id: cdef for cdef in column_defs}

        # TODO: Also prevent inlining expensive or non-deterministic
        # We avoid inlining multi-parent ids, as they would be inlined multiple places, potentially increasing work and/or compiled text size
        self._multi_parent_ids = set(
            id
            for id in self._graph.graph_nodes
            if len(list(self._graph.parents(id))) > 2
        )
        self._free_ids_by_domain: dict[
            ExprDomain, sets.InsertionOrderedSet[identifiers.ColumnId]
        ] = collections.defaultdict(sets.InsertionOrderedSet)

        for id in self._graph.graph_nodes:
            if len(list(self._graph.children(id))) == 0:
                self._mark_free(id)

    @property
    def graph_nodes(self) -> Iterable[nodes.ColumnDef]:
        # should be the same set of ids as self._parents
        return map(self._id_to_cdef.__getitem__, self._graph.graph_nodes)

    @property
    def empty(self):
        return self._graph.empty

    def __len__(self):
        return len(self._graph)

    def parents(self, node: nodes.ColumnDef) -> Iterator[nodes.ColumnDef]:
        yield from map(self._id_to_cdef.__getitem__, self._graph.parents(node.id))

    def children(self, node: nodes.ColumnDef) -> Iterator[nodes.ColumnDef]:
        yield from map(self._id_to_cdef.__getitem__, self._graph.children(node.id))

    def _expr_domain(self, expr: expression.Expression) -> ExprDomain:
        if expr.is_scalar_expr:
            return "Scalar"
        elif isinstance(expr, agg_expressions.WindowExpression):
            return expr.window
        elif isinstance(expr, subquery_expression.SubqueryExpression):
            return "Other"
        else:
            raise ValueError(f"unrecognized  expression {expr}")

    def _mark_free(self, id: identifiers.ColumnId):
        cdef = self._id_to_cdef[id]
        expr = cdef.expression
        # If this expands further, probably generalize a compatibility key
        self._free_ids_by_domain[self._expr_domain(expr)].add(id)

    def _remove_free_mark(self, id: identifiers.ColumnId):
        cdef = self._id_to_cdef[id]
        expr = cdef.expression
        # If this expands further, probably generalize a compatibility key
        if id in self._free_ids_by_domain[self._expr_domain(expr)]:
            self._free_ids_by_domain[self._expr_domain(expr)].remove(id)

    def remove_node(self, node: nodes.ColumnDef) -> None:
        for child in self._children[node]:
            self._parents[child].remove(node)
        for parent in self._parents[node]:
            self._children[parent].remove(node)
            if len(self._children[parent]) == 0:
                self._mark_free(parent.id)
        del self._children[node]
        del self._parents[node]
        self._remove_free_mark(node.id)

    def extract_scalar_exprs(self) -> Sequence[nodes.ColumnDef]:
        results: dict[identifiers.ColumnId, expression.Expression] = dict()
        while (
            True
        ):  # Will converge as each loop either reduces graph size, or fails to find any candidate and breaks
            candidate_ids = list(
                id
                for id in self._free_ids_by_domain["Scalar"]
                if not any(
                    (
                        child in self._multi_parent_ids
                        and id in results.keys()
                        and not is_simple(results[id])
                    )
                    for child in self._graph.children(id)
                )
            )
            if len(candidate_ids) == 0:
                break
            for id in candidate_ids:
                self._graph.remove_node(id)
                new_exprs = {
                    id: self._id_to_cdef[id].expression.bind_refs(
                        results, allow_partial_bindings=True
                    )
                }
                results.update(new_exprs)
        # TODO: We can prune expressions that won't be reused here,
        return tuple(nodes.ColumnDef(expr, id) for id, expr in results.items())

    def extract_window_expr(
        self,
    ) -> Optional[Tuple[Sequence[nodes.ColumnDef], window_spec.WindowSpec]]:
        window = next(
            (
                domain
                for domain in self._free_ids_by_domain
                if domain not in ["Scalar", "Other"]
            ),
            None,
        )
        assert not isinstance(window, str)
        if window:
            window_expr_ids = self._free_ids_by_domain[window]
            window_exprs = (self._id_to_cdef[id] for id in window_expr_ids)
            agg_exprs = tuple(
                nodes.ColumnDef(
                    cast(
                        agg_expressions.WindowExpression, cdef.expression
                    ).analytic_expr,
                    cdef.id,
                )
                for cdef in window_exprs
            )
            for cdef in window_exprs:
                self.remove_node(cdef)
            return (agg_exprs, window)

        return None


def unique_nodes(
    roots: Sequence[expression.Expression],
) -> Generator[expression.Expression, None, None]:
    """Walks the tree for unique nodes"""
    seen = set()
    stack: list[expression.Expression] = list(roots)
    while stack:
        item = stack.pop()
        if item not in seen:
            yield item
            seen.add(item)
            stack.extend(item.children)


def iter_nodes_topo(
    roots: Sequence[expression.Expression],
) -> Generator[expression.Expression, None, None]:
    """Returns nodes in reverse topological order, using Kahn's algorithm."""
    child_to_parents: Dict[
        expression.Expression, list[expression.Expression]
    ] = collections.defaultdict(list)
    out_degree: Dict[expression.Expression, int] = collections.defaultdict(int)

    queue: collections.deque[expression.Expression] = collections.deque()
    for node in unique_nodes(roots):
        num_children = len(node.children)
        out_degree[node] = num_children
        if num_children == 0:
            queue.append(node)
        for child in node.children:
            child_to_parents[child].append(node)

    while queue:
        item = queue.popleft()
        yield item
        parents = child_to_parents.get(item, [])
        for parent in parents:
            out_degree[parent] -= 1
            if out_degree[parent] == 0:
                queue.append(parent)


def reduce_up(
    roots: Sequence[expression.Expression],
    reduction: Callable[[expression.Expression, Tuple[T, ...]], T],
) -> Tuple[T, ...]:
    """Apply a bottom-up reduction to the forest."""
    results: dict[expression.Expression, T] = {}
    for node in list(iter_nodes_topo(roots)):
        # child nodes have already been transformed
        child_results = tuple(results[child] for child in node.children)
        result = reduction(node, child_results)
        results[node] = result

    return tuple(results[root] for root in roots)


def apply_col_exprs_to_plan(
    plan: nodes.BigFrameNode, col_exprs: Sequence[nodes.ColumnDef]
) -> nodes.BigFrameNode:
    target_ids = tuple(named_expr.id for named_expr in col_exprs)

    fragments = fragmentize_expression(col_exprs)
    return push_into_tree(plan, fragments, target_ids)


def apply_agg_exprs_to_plan(
    plan: nodes.BigFrameNode,
    agg_defs: Sequence[nodes.ColumnDef],
    grouping_keys: Sequence[expression.DerefOp],
) -> nodes.BigFrameNode:
    factored_aggs = [factor_aggregation(agg_def) for agg_def in agg_defs]
    all_inputs = list(
        itertools.chain(*(factored_agg.agg_inputs for factored_agg in factored_aggs))
    )
    window_def = window_spec.WindowSpec(grouping_keys=tuple(grouping_keys))
    windowized_inputs = [
        nodes.ColumnDef(windowize(cdef.expression, window_def), cdef.id)
        for cdef in all_inputs
    ]
    plan = apply_col_exprs_to_plan(plan, windowized_inputs)
    all_aggs = list(
        itertools.chain(*(factored_agg.agg_exprs for factored_agg in factored_aggs))
    )
    plan = nodes.AggregateNode(
        plan,
        tuple((cdef.expression, cdef.id) for cdef in all_aggs),  # type: ignore
        by_column_ids=tuple(grouping_keys),
    )

    post_scalar_exprs = tuple(
        (factored_agg.root_scalar_expr for factored_agg in factored_aggs)
    )
    plan = nodes.ProjectionNode(
        plan, tuple((cdef.expression, cdef.id) for cdef in post_scalar_exprs)
    )
    final_ids = itertools.chain(
        (ref.id for ref in grouping_keys), (cdef.id for cdef in post_scalar_exprs)
    )
    plan = nodes.SelectionNode(
        plan, tuple(nodes.AliasedRef.identity(ident) for ident in final_ids)
    )

    return plan


@dataclasses.dataclass(frozen=True, eq=False)
class FactoredExpression:
    root_expr: expression.Expression
    sub_exprs: Tuple[nodes.ColumnDef, ...]


def fragmentize_expression(
    roots: Sequence[nodes.ColumnDef],
) -> Sequence[nodes.ColumnDef]:
    """
    The goal of this functions is to factor out an expression into multiple sub-expressions.
    """
    # TODO: Fragmentize a bit less aggressively
    factored_exprs = reduce_up([root.expression for root in roots], gather_fragments)
    root_exprs = (
        nodes.ColumnDef(factored.root_expr, root.id)
        for factored, root in zip(factored_exprs, roots)
    )
    return (
        *root_exprs,
        *dedupe(
            itertools.chain.from_iterable(
                factored_expr.sub_exprs for factored_expr in factored_exprs
            )
        ),
    )


@dataclasses.dataclass(frozen=True, eq=False)
class FactoredAggregation:
    """
    A three part recomposition of a general aggregating expression.

    1. agg_inputs: This is a set of (*col) -> col transformation that preprocess inputs for the aggregations ops
    2. agg_exprs: This is a set of pure aggregations (eg sum, mean, min, max) ops referencing the outputs of (1)
    3. root_scalar_expr: This is the final set, takes outputs of (2), applies scalar expression to produce final result.
    """

    # pure scalar expression
    root_scalar_expr: nodes.ColumnDef
    # pure agg expression, only refs cols and consts
    agg_exprs: Tuple[nodes.ColumnDef, ...]
    # can be analytic, scalar op, const, col refs
    agg_inputs: Tuple[nodes.ColumnDef, ...]


def windowize(
    root: expression.Expression, window: window_spec.WindowSpec
) -> expression.Expression:
    def windowize_local(expr: expression.Expression):
        if isinstance(expr, agg_expressions.Aggregation):
            if not expr.op.can_be_windowized:
                raise ValueError(f"Op: {expr.op} cannot be windowized.")
            return agg_expressions.WindowExpression(expr, window)
        if isinstance(expr, agg_expressions.WindowExpression):
            raise ValueError(f"Expression {expr} already windowed!")
        return expr

    return root.bottom_up(windowize_local)


def factor_aggregation(root: nodes.ColumnDef) -> FactoredAggregation:
    """
    Factor an aggregation def into three components.
    1. Input column expressions (includes analytic expressions)
    2. The set of underlying primitive aggregations
    3. A final post-aggregate scalar expression
    """
    final_aggs = list(dedupe(find_final_aggregations(root.expression)))
    agg_inputs = list(
        dedupe(itertools.chain.from_iterable(map(find_agg_inputs, final_aggs)))
    )

    agg_input_defs = tuple(
        nodes.ColumnDef(expr, identifiers.ColumnId.unique()) for expr in agg_inputs
    )
    agg_inputs_dict = {
        cdef.expression: expression.DerefOp(cdef.id) for cdef in agg_input_defs
    }

    agg_expr_to_ids = {expr: identifiers.ColumnId.unique() for expr in final_aggs}

    isolated_aggs = tuple(
        nodes.ColumnDef(sub_expressions(expr, agg_inputs_dict), agg_expr_to_ids[expr])
        for expr in final_aggs
    )
    agg_outputs_dict = {
        expr: expression.DerefOp(id) for expr, id in agg_expr_to_ids.items()
    }

    root_scalar_expr = nodes.ColumnDef(
        sub_expressions(root.expression, agg_outputs_dict), root.id  # type: ignore
    )

    return FactoredAggregation(
        root_scalar_expr=root_scalar_expr,
        agg_exprs=isolated_aggs,
        agg_inputs=agg_input_defs,
    )


def sub_expressions(
    root: expression.Expression,
    replacements: Mapping[expression.Expression, expression.Expression],
) -> expression.Expression:
    return root.top_down(lambda x: replacements.get(x, x))


def find_final_aggregations(
    root: expression.Expression,
) -> Iterator[agg_expressions.Aggregation]:
    if isinstance(root, agg_expressions.Aggregation):
        yield root
    elif isinstance(root, expression.OpExpression):
        for child in root.children:
            yield from find_final_aggregations(child)
    elif isinstance(root, expression.ScalarConstantExpression):
        return
    else:
        # eg, window expression, column references not allowed
        raise ValueError(f"Unexpected node: {root}")


def find_agg_inputs(
    root: agg_expressions.Aggregation,
) -> Iterator[expression.Expression]:
    for child in root.children:
        if not isinstance(
            child, (expression.DerefOp, expression.ScalarConstantExpression)
        ):
            yield child


def gather_fragments(
    root: expression.Expression, fragmentized_children: Sequence[FactoredExpression]
) -> FactoredExpression:
    replacements: list[expression.Expression] = []
    named_exprs = []  # root -> leaf dependency order
    for child_result in fragmentized_children:
        child_expr = child_result.root_expr
        is_leaf = isinstance(
            child_expr, (expression.DerefOp, expression.ScalarConstantExpression)
        )
        is_window_agg = isinstance(
            root, agg_expressions.WindowExpression
        ) and isinstance(child_expr, agg_expressions.Aggregation)
        do_inline = is_leaf | is_window_agg
        if not do_inline:
            id = identifiers.ColumnId.unique()
            replacements.append(expression.DerefOp(id))
            named_exprs.append(nodes.ColumnDef(child_result.root_expr, id))
            named_exprs.extend(child_result.sub_exprs)
        else:
            replacements.append(child_result.root_expr)
            named_exprs.extend(child_result.sub_exprs)
    new_root = replace_children(root, replacements)
    return FactoredExpression(new_root, tuple(named_exprs))


def replace_children(
    root: expression.Expression, new_children: Sequence[expression.Expression]
):
    mapping = {root.children[i]: new_children[i] for i in range(len(root.children))}
    return root.transform_children(lambda x: mapping.get(x, x))


def push_into_tree(
    root: nodes.BigFrameNode,
    exprs: Sequence[nodes.ColumnDef],
    target_ids: Sequence[identifiers.ColumnId],
) -> nodes.BigFrameNode:
    curr_root = root
    # id -> id
    graph = ExpressionGraph(exprs)

    while not graph.empty:
        pre_size = len(graph)
        scalar_exprs = graph.extract_scalar_exprs()
        if scalar_exprs:
            curr_root = nodes.ProjectionNode(
                curr_root, tuple((x.expression, x.id) for x in scalar_exprs)
            )
        while result := graph.extract_window_expr():
            defs, window = result
            assert len(defs) > 0
            curr_root = nodes.WindowOpNode(
                curr_root,
                tuple(defs),
                window,
            )
        if len(graph) >= pre_size:
            raise ValueError("graph didn't shrink")
    # TODO: Try to get the ordering right earlier, so can avoid this extra node.
    post_ids = (*root.ids, *target_ids)
    if tuple(curr_root.ids) != post_ids:
        curr_root = nodes.SelectionNode(
            curr_root, tuple(nodes.AliasedRef.identity(id) for id in post_ids)
        )
    return curr_root


@functools.cache
def is_simple(expr: expression.Expression) -> bool:
    count = 0
    for part in expr.walk():
        count += 1
        if count > _MAX_INLINE_COMPLEXITY:
            return False
    return True


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


def grouped(values: Iterable[tuple[K, V]]) -> dict[K, list[V]]:
    result = collections.defaultdict(list)
    for k, v in values:
        result[k].append(v)
    return result


def dedupe(values: Iterable[K]) -> Iterator[K]:
    seen = set()
    for k in values:
        if k not in seen:
            seen.add(k)
            yield k
