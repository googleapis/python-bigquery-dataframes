from __future__ import annotations

from collections import defaultdict
import dataclasses
import dis
import inspect
from typing import Any, Optional

from bigframes.core import expression
import bigframes.operations as ops

LabelType = str

# the instructions immediately after these are implicitly branch targets
_FALLTHROUGH_BRANCHES = set(
    [
        "POP_JUMP_IF_TRUE",
        "POP_JUMP_IF_FALSE",
        "POP_JUMP_IF_NOT_NONE",
        "POP_JUMP_IF_NONE",
    ]
)

_BINARY_OPNAMES = {
    "BINARY_ADD": ops.add_op,
    "BINARY_ASUBTRACT": ops.sub_op,
    "BINARY_MULTIPLY": ops.mul_op,
    "BINARY_TRUE_DIVIDE": ops.div_op,
}


_BINARY_OPARGS = (
    ops.add_op,
    ops.and_op,
    ops.floordiv_op,
    None,  # lshift
    None,  # matrix multiply
    ops.mul_op,
    ops.mod_op,
    ops.or_op,
    ops.pow_op,
    None,  # rshift
    ops.sub_op,
    ops.div_op,
    ops.xor_op,
    # ops 12-24 are unimplemented inplace ops
)

_COMPARE_OPARGS = [
    ops.lt_op,
    ops.le_op,
    ops.eq_null_match_op,
    ops.ne_op,
    ops.gt_op,
    ops.ge_op,
]

# --- CFG and SSA Infrastructure ---
VERSIONING_SEPERATOR = "%"


@dataclasses.dataclass(eq=False)
class Operand:
    """Represents an operand for an instruction, which can be a constant or a variable."""

    value: Any
    is_const: bool = False

    def __repr__(self):
        # Represent constants directly, and variables as is.
        return f"{self.value}" if self.is_const else f"{self.value}"


@dataclasses.dataclass(eq=False)
class Instruction:
    """Represents a single instruction in a basic block using a 3-address-code like format."""

    opname: str
    oparg: Optional[int] = None
    target: Optional[str] = None
    args: list[Operand] = dataclasses.field(default_factory=list)

    def __repr__(self):
        oparg_part = f"({self.oparg})" if self.oparg else ""
        args_part = f"({', '.join(map(str, self.args))})"
        if self.target:
            return f"{self.target} = {self.opname + oparg_part + args_part})"

        return self.opname + oparg_part + args_part


@dataclasses.dataclass(eq=False)
class PhiFunction(Instruction):
    """Represents a phi function, used in SSA form to merge variable versions."""

    def __init__(self, target):
        super().__init__("phi", target=target)

    def __repr__(self):
        return f"{self.target} = phi({', '.join(map(str, self.args))})"


@dataclasses.dataclass(eq=False)
class BasicBlock:
    """Represents a basic block in the control flow graph."""

    label: LabelType
    instructions: list[Instruction] = dataclasses.field(default_factory=list)
    successors: set[BasicBlock] = dataclasses.field(default_factory=set)
    predecessors: set[BasicBlock] = dataclasses.field(default_factory=set)
    # Define dominance structure, which is important for minimizing phi set
    dominators: set[BasicBlock] = dataclasses.field(default_factory=set)
    idom: Optional[BasicBlock] = None
    dominance_frontier: set[BasicBlock] = dataclasses.field(default_factory=set)

    def __repr__(self):
        return f"Block({self.label})"

    def add_instruction(self, instruction):
        self.instructions.append(instruction)


@dataclasses.dataclass(eq=False)
class ControlFlowGraph:
    """Represents the control flow graph of a function."""

    blocks: dict[LabelType, BasicBlock] = dataclasses.field(default_factory=dict)
    entry_block: Optional[BasicBlock] = None

    def get_block(self, label: LabelType) -> BasicBlock:
        if label not in self.blocks:
            self.blocks[label] = BasicBlock(label)
        return self.blocks[label]

    def to_dot(self):
        """Generates a DOT representation of the CFG for visualization."""
        dot = "digraph CFG {\n"
        dot += 'node [shape=box, fontname="Courier"];\n'
        for label, block in sorted(self.blocks.items()):
            instr_str = "\\l".join(str(i) for i in block.instructions) + "\\l"
            dot += f'  {label} [label="{label}:\\l{instr_str}"];\n'
            for succ in sorted(block.successors, key=lambda b: b.label):
                dot += f"  {label} -> {succ.label};\n"
        dot += "}"
        return dot


class SSATranspiler:
    """Transpiles Python bytecode to an SSA CFG."""

    def __init__(self, func):
        self.func = func
        self.bytecode = list(dis.get_instructions(func))
        self.cfg = ControlFlowGraph()
        self.build_cfg()
        self.convert_to_single_exit()
        self.compute_dominators()
        self.compute_dominance_frontiers()
        self.insert_phi_functions()
        self.rename_variables()

    def build_cfg(self):
        """Builds the initial CFG from the bytecode by converting stack operations to a register-like IR."""

        # ok, so jump targets
        jump_targets = {instr.offset for instr in self.bytecode if instr.is_jump_target}
        fallthrough_targets = {
            instr.offset
            for i, instr in enumerate(self.bytecode)
            if (i > 0) and self.bytecode[i - 1].opname in _FALLTHROUGH_BRANCHES
        }
        block_starts = {0} | jump_targets | fallthrough_targets

        # Create all basic blocks
        for offset in block_starts:
            self.cfg.get_block(f"B{offset}")
        self.cfg.entry_block = self.cfg.get_block("B0")

        current_block = self.cfg.entry_block
        stack = []

        for i, instr in enumerate(self.bytecode):
            print(instr)

            if instr.offset in block_starts and instr.offset != 0:
                current_block = self.cfg.get_block(f"B{instr.offset}")

            opname = instr.opname
            arg = instr.argval

            if opname.startswith("LOAD_"):
                if "CONST" in opname:
                    stack.append(Operand(arg, is_const=True))
                else:  # LOAD_FAST, etc.
                    stack.append(Operand(arg, is_const=False))

            elif opname.startswith("STORE_"):  # STORE_FAST
                value_operand = stack.pop()
                # Create an assignment instruction. This is a variable definition.
                new_instr = Instruction("ASSIGN", target=arg)
                new_instr.args = [value_operand]
                current_block.add_instruction(new_instr)

            elif opname.startswith("BINARY_") or opname == "COMPARE_OP":
                right, left = stack.pop(), stack.pop()
                # Operation produces a temporary value, which we push back for the next instruction.
                temp_target = f"t{instr.offset}"
                new_instr = Instruction(opname, oparg=instr.arg, target=temp_target)
                # For compare, we also store the operation (e.g., '>', '<')
                # if opname == 'COMPARE_OP':
                #    new_instr.opname = f"COMPARE_OP_{arg}"
                new_instr.args = [left, right]
                current_block.add_instruction(new_instr)
                stack.append(Operand(temp_target, is_const=False))

            elif opname == "RETURN_VALUE":
                new_instr = Instruction("RETURN")
                new_instr.args = [stack.pop()]
                current_block.add_instruction(new_instr)

            elif opname == "POP_JUMP_IF_FALSE":
                cond_operand = stack.pop()
                branch_instr = Instruction("BRANCH_IF_FALSE")
                branch_instr.args = [cond_operand]
                current_block.add_instruction(branch_instr)

                target_block = self.cfg.get_block(f"B{arg}")
                fallthrough_block = self.cfg.get_block(
                    f"B{self.bytecode[i + 1].offset}"
                )

                # Successors: False path first, then True path
                current_block.successors.add(target_block)  # False path
                current_block.successors.add(fallthrough_block)  # True path
                target_block.predecessors.add(current_block)
                fallthrough_block.predecessors.add(current_block)

            elif opname in ("JUMP_FORWARD", "JUMP_ABSOLUTE"):
                target_block = self.cfg.get_block(f"B{arg}")
                current_block.successors.add(target_block)
                target_block.predecessors.add(current_block)

            elif opname == "POP_TOP":
                stack.pop()

            elif opname == "RESUME":
                # literal no-op
                continue
            else:
                raise ValueError(f"Unrecognized operation: {instr}")

    def convert_to_single_exit(self):
        """
        Modifies the CFG to have a single exit block.
        All RETURN instructions are replaced with assignments to a special
        return variable, and those blocks now jump to a single exit block.
        """
        return_blocks = []
        for block in self.cfg.blocks.values():
            for i, instr in enumerate(block.instructions):
                if instr.opname == "RETURN":
                    return_blocks.append((block, instr, i))

        if not return_blocks:
            return

        exit_block = self.cfg.get_block("B_EXIT")
        return_var_name = "_return_value_"

        for block, return_instr, instr_index in return_blocks:
            return_operand = return_instr.args[0]

            assign_instr = Instruction("ASSIGN", target=return_var_name)
            assign_instr.args = [return_operand]
            block.instructions[instr_index] = assign_instr

            assert len(block.successors) == 0
            # block.successors.clear() # Should be clear already???
            block.successors.add(exit_block)
            exit_block.predecessors.add(block)

        final_return_instr = Instruction("RETURN")
        final_return_instr.args = [Operand(return_var_name, is_const=False)]
        exit_block.add_instruction(final_return_instr)

    def compute_dominators(self):
        """Computes the dominators for each basic block using the standard iterative algorithm."""
        all_blocks = set(self.cfg.blocks.values())
        assert self.cfg.entry_block is not None
        self.cfg.entry_block.dominators = {self.cfg.entry_block}

        for block in self.cfg.blocks.values():
            if block != self.cfg.entry_block:
                block.dominators = all_blocks.copy()

        changed = True
        while changed:
            changed = False
            for block_lbl in sorted(self.cfg.blocks.keys()):
                block = self.cfg.blocks[block_lbl]
                if block == self.cfg.entry_block:
                    continue
                if not block.predecessors:
                    continue

                pred_doms = [p.dominators for p in block.predecessors]
                new_doms = {block} | set.intersection(*pred_doms)

                if new_doms != block.dominators:
                    block.dominators = new_doms
                    changed = True

        for block in self.cfg.blocks.values():
            immediate_dominators = {d for d in block.dominators if d != block}
            idom = next(
                (
                    d
                    for d in immediate_dominators
                    if all(
                        dom not in d.dominators
                        for dom in immediate_dominators
                        if dom != d
                    )
                ),
                None,
            )
            block.idom = idom

    def compute_dominance_frontiers(self):
        """Computes the dominance frontier for each basic block."""
        for b in self.cfg.blocks.values():
            if len(b.predecessors) > 1:
                for p in b.predecessors:
                    runner = p
                    while runner != b.idom:
                        runner.dominance_frontier.add(b)
                        if runner.idom is None:
                            break
                        runner = runner.idom

    def insert_phi_functions(self):
        """Inserts phi functions where needed for variables with multiple definitions."""
        defs: dict[str, set[BasicBlock]] = defaultdict(set)
        variables = set()

        arg_names = inspect.signature(self.func).parameters.keys()
        for name in arg_names:
            assert self.cfg.entry_block is not None
            defs[name].add(self.cfg.entry_block)
            variables.add(name)

        for block in self.cfg.blocks.values():
            for instr in block.instructions:
                if instr.opname == "ASSIGN":
                    assert instr.target is not None
                    defs[instr.target].add(block)
                    variables.add(instr.target)

        for var in variables:
            def_sites = defs[var]
            worklist = list(def_sites)

            while worklist:
                block = worklist.pop(0)
                for d_frontier_block in block.dominance_frontier:
                    if not any(
                        isinstance(i, PhiFunction) and i.target == var
                        for i in d_frontier_block.instructions
                    ):
                        phi = PhiFunction(var)
                        d_frontier_block.instructions.insert(0, phi)
                        if d_frontier_block not in def_sites:
                            def_sites.add(d_frontier_block)
                            worklist.append(d_frontier_block)

    def rename_variables(self):
        """
        Performs a pass over the CFG to rename all variables, putting the graph into SSA form.
        """
        counters = defaultdict(int)
        stacks = defaultdict(list)

        arg_names = inspect.signature(self.func).parameters.keys()
        for name in arg_names:
            stacks[name].append(0)
            counters[name] = 1

        def rename_recursive(block: BasicBlock):
            pushed_on_stack = []

            # Do we need to do def and ref serially??

            # rename definitions
            for instr in block.instructions:
                if instr.target:
                    var = instr.target
                    version = counters[var]
                    counters[var] += 1
                    stacks[var].append(version)
                    pushed_on_stack.append(var)
                    instr.target = f"{var}{VERSIONING_SEPERATOR}{version}"

            # rename references
            for instr in block.instructions:
                if not isinstance(instr, PhiFunction):
                    renamed_args = []
                    for arg in instr.args:
                        if not arg.is_const:
                            if not stacks[arg.value]:
                                raise Exception(
                                    f"Compiler error: variable '{arg.value}' used before definition."
                                )
                            version = stacks[arg.value][-1]
                            renamed_args.append(
                                Operand(
                                    f"{arg.value}{VERSIONING_SEPERATOR}{version}",
                                    is_const=False,
                                )
                            )
                        else:
                            renamed_args.append(arg)
                    instr.args = renamed_args

            # Peek ahead and modify downstream phi functions
            for succ in sorted(block.successors, key=lambda b: b.label):
                for instr in succ.instructions:
                    if isinstance(instr, PhiFunction):
                        assert instr.target is not None
                        var = instr.target.split(VERSIONING_SEPERATOR)[0]
                        if not stacks[var]:
                            versioned_operand = Operand("UNDEFINED", is_const=True)
                        else:
                            version = stacks[var][-1]
                            versioned_operand = Operand(
                                f"{var}{VERSIONING_SEPERATOR}{version}", is_const=False
                            )
                        instr.args.append(versioned_operand)

            # and now fully process children in the dominance tree
            # TODO: This is inefficient for many blocks, we should have an explicit dominance tree
            for child in sorted(
                (b for b in self.cfg.blocks.values() if b.idom == block),
                key=lambda b: b.label,
            ):
                rename_recursive(child)

            # undefine
            for var in pushed_on_stack:
                stacks[var].pop()

        assert self.cfg.entry_block is not None
        rename_recursive(self.cfg.entry_block)

    def to_sql_expr(self) -> expression.Expression:
        """Converts the SSA CFG into a single SQL expression tree."""
        memo: dict[str, expression.Expression] = {}  # SSA var name -> Expression

        # Pre-populate with function arguments as free variables
        arg_names = inspect.signature(self.func).parameters.keys()
        for name in arg_names:
            memo[f"{name}{VERSIONING_SEPERATOR}0"] = expression.free_var(name)

        # Topologically sort blocks to process definitions before uses
        sorted_blocks = self._topological_sort()

        for block in sorted_blocks:
            # TODO: LOAD, CALL, TO_BOOL,
            for instr in block.instructions:
                if instr.opname == "ASSIGN":
                    assert instr.target is not None
                    # Simple alias, point to the expression of the source
                    source_expr = self._operand_to_expr(instr.args[0], memo)
                    memo[instr.target] = source_expr

                elif isinstance(instr, PhiFunction):
                    assert instr.target is not None

                    # This is the conditional logic
                    # Find the branching block that is the idom of this phi's block
                    branch_block = block.idom
                    assert branch_block is not None
                    branch_instr = next(
                        (
                            i
                            for i in branch_block.instructions
                            if i.opname == "BRANCH_IF_FALSE"
                        ),
                        None,
                    )

                    if branch_instr:
                        predicate_expr = self._operand_to_expr(
                            branch_instr.args[0], memo
                        )

                        # Determine true/false path based on successor order
                        # preds = sorted(block.predecessors, key=lambda b: b.label)

                        # This logic assumes a simple if/else diamond shape
                        true_path_val = self._operand_to_expr(
                            instr.args[0], memo
                        )  # Fallthrough
                        false_path_val = self._operand_to_expr(
                            instr.args[1], memo
                        )  # Jump target

                        memo[instr.target] = ops.where_op.as_expr(
                            true_path_val, predicate_expr, false_path_val
                        )
                    else:
                        # Handle more complex cases like CASE WHEN here if needed
                        raise ValueError(f"Unhandled branch instruction {instr}")
                        pass

                elif instr.opname == "RETURN":
                    # This is the final expression
                    return self._operand_to_expr(instr.args[0], memo)

                elif instr.target:  # All other ops that define a variable
                    op_args = [self._operand_to_expr(arg, memo) for arg in instr.args]
                    op_name = instr.opname
                    # Map compare op to something the Ops class can handle
                    if op_name == "BINARY_OP":  # fused verions
                        assert instr.oparg is not None
                        sql_op = _BINARY_OPARGS[instr.oparg]
                        assert sql_op is not None

                    elif op_name.startswith("BINARY_"):
                        sql_op = _BINARY_OPNAMES[op_name]

                    elif op_name.startswith("COMPARE_OP"):
                        assert instr.oparg is not None
                        # are these consistent across python versions??
                        sql_op = _COMPARE_OPARGS[instr.oparg >> 4]

                    memo[instr.target] = sql_op.as_expr(*op_args)  # type: ignore
                else:  # usually jump instructions, which maybe blocks should track separately?
                    pass
                    # raise Exception(f"Unhandled instruction {instr}")

        # print(memo)
        # if "_return_value__0" in memo:
        #    return memo["_return_value__0"]
        raise Exception("Could not find RETURN instruction in single exit block.")

    def _operand_to_expr(self, operand: Operand, memo: dict) -> expression.Expression:
        """Helper to convert an Operand to an Expression."""
        if operand.is_const:
            return expression.const(operand.value)
        else:
            return memo[operand.value]

    def _topological_sort(self) -> list[BasicBlock]:
        """Performs a topological sort of the basic blocks."""
        result: list[BasicBlock] = []
        in_degree = {
            label: len(block.predecessors) for label, block in self.cfg.blocks.items()
        }
        queue = [self.cfg.entry_block]

        while queue:
            block = queue.pop(0)
            assert block is not None
            result.append(block)

            for successor in sorted(block.successors, key=lambda b: b.label):
                in_degree[successor.label] -= 1
                if in_degree[successor.label] == 0:
                    queue.append(successor)

        return result
