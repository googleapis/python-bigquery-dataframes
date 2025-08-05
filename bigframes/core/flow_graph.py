from __future__ import annotations

from collections import defaultdict
import dataclasses
import dis
import functools
import inspect
from typing import Any, Literal, Mapping, Optional, Sequence, Union

from bigframes.core import expression, py_exprs
import bigframes.dtypes
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
    # These are the inplace variants, but should work same at bytecode level for the relevant types?
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
)

_COMPARE_OPARGS = [
    ops.lt_op,
    ops.le_op,
    ops.eq_null_match_op,
    ops.ne_op,
    ops.gt_op,
    ops.ge_op,
]

PREDICATE_T = Literal["IS NONE", "IS TRUE"]
VERSIONING_SEPERATOR = "%"


# Operand ontology here is rough, early vs late resolution?
@dataclasses.dataclass(frozen=True)
class Operand:
    """Represents an operand for an instruction, which can be a constant or a variable."""

    @property
    def is_const(self) -> bool:
        return False

    def rename_refs(self, renamings: Mapping[str, str]) -> Operand:
        return self


@dataclasses.dataclass(frozen=True)
class ConstArg(Operand):  # represents non-arguments: could be from closure, module, etc
    value: Any

    def __repr__(self):
        return f"{self.value}"

    @property
    def is_const(self):
        return True


# Maybe create a separate class for temp args?
@dataclasses.dataclass(frozen=True)
class VariableRef(Operand):  # represents a reference to a variable
    name: str

    def __post_init__(self):
        assert isinstance(self.name, str)

    def __repr__(self):
        return self.name

    @property
    def is_const(self):
        return False

    def rename_refs(self, renamings: Mapping[str, str]) -> Operand:
        return VariableRef(renamings[self.name]) if self.name in renamings else self


@dataclasses.dataclass(frozen=True)
class Undefined(
    Operand
):  # represents variable state in control flow paths where the variable is undefined
    def __repr__(self):
        return "UNDEFINED"

    @property
    def is_const(self):
        return True

    def rename_refs(self, renamings: Mapping[str, str]) -> Operand:
        return self


# TODO: Maybe type all the instructions? Eg LOAD, ASSIGN, OP, ATTR, CALL?
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

    def references(self) -> set[str]:
        return {item.name for item in self.args if isinstance(item, VariableRef)}


@dataclasses.dataclass(eq=False)
class PhiFunction(Instruction):
    """Represents a phi function, used in SSA form to merge variable versions."""

    def __init__(self, target):
        super().__init__("phi", target=target)
        # blocks holds references to the predecessor blocks for each argument in args.
        self.blocks: list[BasicBlock] = []

    def __repr__(self):
        # After renaming, blocks and args are parallel arrays
        if self.blocks:
            arg_str = ", ".join(
                f"{block.label}:{op}" for block, op in zip(self.blocks, self.args)
            )
            return f"{self.target} = phi({arg_str})"
        return f"{self.target} = phi({', '.join(map(str, self.args))})"


# Block Terminators: Branch, Goto or Return
@dataclasses.dataclass(frozen=True)
class BinaryBranchCondition:
    predicate: PREDICATE_T
    target: Operand
    true_case: BasicBlock
    false_case: BasicBlock

    @property
    def successors(self) -> Sequence[BasicBlock]:
        return (self.true_case, self.false_case)

    def rename_refs(self, renamings: Mapping[str, str]) -> BinaryBranchCondition:
        return BinaryBranchCondition(
            self.predicate,
            self.target.rename_refs(renamings),
            self.true_case,
            self.false_case,
        )

    def references(self) -> set[str]:
        return {item.name for item in [self.target] if isinstance(item, VariableRef)}

    def __repr__(self):
        return f"BRANCH {self.target} {self.predicate} ? {self.true_case.label} : {self.false_case.label}"


@dataclasses.dataclass(frozen=True)
class Goto:
    to: BasicBlock

    @property
    def successors(self) -> Sequence[BasicBlock]:
        return (self.to,)

    def rename_refs(self, renamings: Mapping[str, str]) -> Goto:
        return self

    def references(self) -> set[str]:
        return set()

    def __repr__(self):
        return f"GOTO {self.to.label}"


@dataclasses.dataclass(frozen=True)
class Return:
    target: Operand

    @property
    def successors(self) -> Sequence[BasicBlock]:
        return ()

    def rename_refs(self, renamings: Mapping[str, str]) -> Return:
        return Return(self.target.rename_refs(renamings))

    def references(self) -> set[str]:
        return {item.name for item in [self.target] if isinstance(item, VariableRef)}

    def __repr__(self):
        return f"RETURN {self.target}"


@dataclasses.dataclass(eq=False)
class BasicBlock:
    """Represents a basic block in the control flow graph."""

    label: LabelType
    instructions: list[Instruction] = dataclasses.field(default_factory=list)
    predecessors: set[BasicBlock] = dataclasses.field(default_factory=set)
    then: Union[BinaryBranchCondition, Goto, Return, None] = None

    # Define dominance structure, which is important for minimizing phi set
    dominators: set[BasicBlock] = dataclasses.field(default_factory=set)
    idom: Optional[BasicBlock] = None
    dominance_frontier: set[BasicBlock] = dataclasses.field(default_factory=set)

    def __repr__(self):
        return f"Block({self.label})"

    def add_instruction(self, instruction):
        self.instructions.append(instruction)

    @property
    def successors(self) -> Sequence[BasicBlock]:
        if self.then is None:
            raise ValueError("Block isn't fully defined yet")
        else:
            return self.then.successors

    def rename_refs(self, renamings: Mapping[str, str]) -> None:
        """Renames all references to variables. Useful when converting to SSA"""
        for instr in self.instructions:
            if not isinstance(instr, PhiFunction):
                instr.args = list(map(lambda x: x.rename_refs(renamings), instr.args))
        if self.then is not None:
            self.then = self.then.rename_refs(renamings)


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
            if block.then:
                instr_str += "-" * 30 + "\\l"
                instr_str += str(block.then) + "\\l"
            dot += f'  {label} [label="{label}:\\l{instr_str}"];\n'
            for succ in sorted(block.successors, key=lambda b: b.label):
                dot += f"  {label} -> {succ.label};\n"
            if block.idom is not None:
                dot += (
                    f'  {block.idom.label} -> {label} [color=red, label="dominates"];\n'
                )
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
        # https://en.wikipedia.org/wiki/Dominator_(graph_theory)
        self.compute_dominators()
        self.compute_dominance_frontiers()
        self.insert_phi_functions()
        # Converts to static single-assignment form (SSA for short)
        # https://en.wikipedia.org/wiki/Static_single-assignment_form
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
        stack: list[Operand] = []

        # General philosophy here is we try to nomralize a bit between python versions, but mostly leave instructions intact
        # Also, mostly be strict about recognizing stuff?
        for instr in self.bytecode:
            print(instr)

        for i, instr in enumerate(self.bytecode):
            if instr.offset in block_starts and instr.offset != 0:
                current_block = self.cfg.get_block(f"B{instr.offset}")

            opname = instr.opname

            if opname.startswith("LOAD_"):
                if "CONST" in opname:
                    stack.append(ConstArg(instr.argval))
                elif opname == "LOAD_GLOBAL":
                    if instr.argval in self.func.__globals__:
                        stack.append(ConstArg(self.func.__globals__[instr.argval]))
                    else:
                        # It must be a builtin. The __builtins__ can be a dict or a module.
                        builtins = self.func.__globals__["__builtins__"]
                        if isinstance(builtins, dict):
                            val = builtins[instr.argval]
                        else:  # It's the builtins module
                            val = getattr(builtins, instr.argval)
                        stack.append(ConstArg(val))
                elif opname in {"LOAD_METHOD", "LOAD_ATTR"}:
                    module_or_object = stack.pop()
                    temp_target = f"t{instr.offset}"
                    new_instr = Instruction(
                        "LOAD_ATTR", oparg=instr.arg, target=temp_target
                    )
                    new_instr.args = [module_or_object, ConstArg(instr.argval)]
                    current_block.add_instruction(new_instr)
                    stack.append(VariableRef(temp_target))
                else:  # LOAD_FAST, etc.
                    if instr.argval in self.func.__code__.co_varnames:
                        stack.append(VariableRef(instr.argval))
                    elif instr.argval in self.func.__code__.co_freevars:
                        stack.append(ConstArg(self._closure_vars[instr.argval]))
                    else:
                        raise ValueError(f"Can't identify source of load: {instr}")

            elif opname.startswith("STORE_"):  # STORE_FAST
                # TODO: Maybe block storing to closure vars? Users may expect info passing between invocations.
                value_operand = stack.pop()
                # Create an assignment instruction. This is a variable definition.
                new_instr = Instruction("ASSIGN", target=instr.argval)
                new_instr.args = [value_operand]
                current_block.add_instruction(new_instr)
            elif opname.startswith("BINARY_") or opname == "COMPARE_OP":
                # We might want to just normalize down here?
                right, left = stack.pop(), stack.pop()
                # Operation produces a temporary value, which we push back for the next instruction.
                temp_target = f"t{instr.offset}"
                new_instr = Instruction(opname, oparg=instr.arg, target=temp_target)
                new_instr.args = [left, right]
                current_block.add_instruction(new_instr)
                stack.append(VariableRef(temp_target))

            elif opname.startswith("CALL"):
                # TODO: More CALL variations
                if (
                    opname == "CALL"
                ):  # 3.11+ stack = callable, self/NULL, posarg1, posarg2, ...
                    temp_target = f"t{instr.offset}"
                    nargs = instr.arg
                    assert nargs is not None
                    args = []
                    for _ in range(nargs):
                        args.append(stack.pop())
                    function = stack.pop()  # expecting actual python callable
                    new_instr = Instruction("CALL", target=temp_target)
                    new_instr.args = [function, *args]
                    current_block.add_instruction(new_instr)
                    stack.append(VariableRef(temp_target))
                else:
                    raise ValueError(f"Unrecognized operation: {instr}")
            # Terminal cases
            elif opname == "RETURN_VALUE":
                current_block.then = Return(stack.pop())
            elif opname.startswith("POP_JUMP"):
                cond_operand = stack.pop()
                jump_block = self.cfg.get_block(f"B{instr.argval}")
                fallthrough_block = self.cfg.get_block(
                    f"B{self.bytecode[i + 1].offset}"
                )
                predicate: Literal["IS TRUE", "IS NONE"] = "IS TRUE"
                if opname == "POP_JUMP_IF_TRUE":
                    true_case = jump_block
                    false_case = fallthrough_block
                elif opname == "POP_JUMP_IF_FALSE":
                    false_case = jump_block
                    true_case = fallthrough_block
                elif opname == "POP_JUMP_IF_NONE":
                    predicate = "IS NONE"
                    true_case = jump_block
                    false_case = fallthrough_block
                elif opname == "POP_JUMP_IF_NOT_NONE":
                    predicate = "IS NONE"
                    false_case = jump_block
                    true_case = fallthrough_block
                else:
                    raise ValueError(f"unrecognized instruction: {opname}")

                current_block.then = BinaryBranchCondition(
                    predicate, cond_operand, true_case, false_case
                )
                jump_block.predecessors.add(current_block)
                fallthrough_block.predecessors.add(current_block)

            elif opname in ("JUMP_FORWARD", "JUMP_ABSOLUTE"):
                target_block = self.cfg.get_block(f"B{instr.argval}")
                current_block.then = Goto(target_block)
                target_block.predecessors.add(current_block)
            elif opname == "POP_TOP":
                stack.pop()

            elif opname in {"COPY_FREE_VARS"}:
                # we don't need to copy this in, we can just directly dereference later
                continue
            elif opname == "RESUME":
                # literal no-op
                continue
            else:
                raise ValueError(f"Unrecognized operation: {instr}")

            # Finally, this might end the block, in that case, goto next instruction
            if current_block.then is None:
                if (len(self.bytecode) > i + 1) and (
                    self.bytecode[i + 1].offset in block_starts
                ):
                    next_block = self.cfg.get_block(f"B{self.bytecode[i + 1].offset}")
                    current_block.then = Goto(next_block)
                    next_block.predecessors.add(current_block)

    @functools.cached_property
    def _closure_vars(self):
        """
        Returns a dictionary of a function's closed-over variables and their values.
        """
        if not self.func.__closure__:
            return {}

        # Get names and cell objects from the function
        var_names = self.func.__code__.co_freevars
        closure_cells = self.func.__closure__

        # Zip them together and get the value from each cell
        return {
            name: cell.cell_contents for name, cell in zip(var_names, closure_cells)
        }

    def convert_to_single_exit(self):
        """
        Modifies the CFG to have a single exit block.
        All RETURN instructions are replaced with assignments to a special
        return variable, and those blocks now jump to a single exit block.
        """
        return_blocks: list[BasicBlock] = []
        for block in self.cfg.blocks.values():
            if isinstance(block.then, Return):
                return_blocks.append(block)

        if not return_blocks:
            return

        exit_block = self.cfg.get_block("B_EXIT")
        return_var_name = "_return_value_"

        for block in return_blocks:
            assert isinstance(block.then, Return)
            return_operand = block.then.target

            assign_instr = Instruction("ASSIGN", target=return_var_name)
            assign_instr.args = [return_operand]
            block.instructions.append(assign_instr)

            assert len(block.successors) == 0
            block.then = Goto(exit_block)
            exit_block.predecessors.add(block)

        exit_block.then = Return(VariableRef(return_var_name))

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

        # o(n**2) algo, not great
        for block in self.cfg.blocks.values():

            strict_dominators = block.dominators.difference([block])
            # select the immediate dominator d (sus) such that no other immediate dominator dominates it
            idom = (
                next(  # we want the dominator that doesn't dominate any other dominator
                    (
                        d
                        for d in strict_dominators
                        if all(
                            d not in other_dom.dominators
                            for other_dom in strict_dominators.difference([d])
                        )
                    ),
                    None,
                )
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
        """
        Inserts phi functions where needed for variables with multiple definitions.

        Note: This must be done **before** converting to ssa.
        """
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

        # OK, so what does this state represent?

        # current counters for each variable
        counters = defaultdict(int)

        stacks: dict[str, list[int]] = defaultdict(list)

        arg_names = inspect.signature(self.func).parameters.keys()
        for name in arg_names:
            stacks[name].append(0)
            counters[name] = 1

        def versioned_name(name: str, version: int) -> str:
            return f"{name}{VERSIONING_SEPERATOR}{version}"

        def rename_recursive(block: BasicBlock):
            pushed_on_stack = []

            # Do we need to do def and ref serially??

            # rename definitions
            for instr in block.instructions:
                if not isinstance(instr, PhiFunction):
                    for name in instr.references():
                        if len(stacks[name]) < 1:
                            raise ValueError(f"Name {name} reffed before decl")
                    # TODO: Aghhhhh
                    renames = {
                        name: versioned_name(name, stacks[name][-1])
                        for name in instr.references()
                    }
                    instr.args = list(map(lambda x: x.rename_refs(renames), instr.args))
                if instr.target:
                    var = instr.target
                    version = counters[var]
                    counters[var] += 1
                    stacks[var].append(version)
                    pushed_on_stack.append(var)
                    instr.target = versioned_name(var, version)

            # rename references
            if block.then is not None:
                renames = {
                    name: versioned_name(name, stacks[name][-1])
                    for name in block.then.references()
                }
                block.then = block.then.rename_refs(renames)

            # Peek ahead and modify downstream phi functions
            for succ in sorted(block.successors, key=lambda b: b.label):
                for instr in succ.instructions:
                    if isinstance(instr, PhiFunction):
                        assert instr.target is not None
                        var = instr.target.split(VERSIONING_SEPERATOR)[0]
                        if not stacks[var]:
                            versioned_operand: Operand = Undefined()
                        else:
                            version = stacks[var][-1]
                            versioned_operand = VariableRef(
                                f"{var}{VERSIONING_SEPERATOR}{version}"
                            )
                        instr.blocks.append(block)
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

    # TODO: A second mode that takes a single-arg function, and treats LOAD_ATTR, BINARY_SUBSCR as the main way to get columns
    # Maybe also possible to spit out multiple expressions instead to minimize expression size?
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
                    # zip the blocks and args back together for the resolver
                    zipped_args = list(zip(instr.blocks, instr.args))
                    assert block.idom is not None
                    memo[instr.target] = self._resolve_phi_to_expr(
                        zipped_args,
                        branch_block=block.idom,
                        merge_block=block,
                        memo=memo,
                    )
                elif instr.opname == "LOAD_ATTR":
                    assert instr.target is not None
                    # So there are basically a few main cases we want to handle here
                    # 1. Loading a module method such as np.add , may or may not be callable (eg math.pi vs math.sqrt())
                    # 2. Getting an attribute on a const (eg "hello".upper()). May or may not be callable (eg str.upper() )
                    # 3. Getting an attribuge on a variable (eg x.abs()) (May or may not be callable)
                    # When do we know if it is callable? Eh, we can generally decide right now.
                    # OK, so what are the options
                    # 1. Throw an attr op on the expression - fairly clean conceptually, though we will have to rewrite later (either at call)
                    obj, attr = instr.args
                    assert isinstance(attr, ConstArg)  # No dynamic attribute support
                    memo[instr.target] = py_exprs.GetAttr(
                        self._operand_to_expr(obj, memo), attr.value
                    )
                elif instr.opname == "CALL":
                    assert instr.target is not None
                    (callable, *call_args) = instr.args
                    call_expr = py_exprs.Call(
                        self._operand_to_expr(callable, memo),
                        tuple(self._operand_to_expr(arg, memo) for arg in call_args),
                    )
                    # we are resolving a little bit early rn, maybe should defer??
                    memo[instr.target] = call_expr  # py_exprs.resolve_call(call_expr)
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
                    # pass
                    raise Exception(f"Unhandled instruction {instr}")

            if isinstance(block.then, Return):
                unresolved = self._operand_to_expr(block.then.target, memo)
                resolved = py_exprs.resolve_py_exprs(unresolved)
                print(print_expr_tree(resolved))
                return resolved

        raise Exception("Could not find RETURN instruction in single exit block.")

    def _operand_to_expr(self, operand: Operand, memo: dict) -> expression.Expression:
        """Helper to convert an Operand to an Expression."""
        if isinstance(operand, ConstArg):
            # We are still being pretty permissive here, and just stuffing things in the expression
            # Later steps will try to resolve and translate these and may fail, thats fine though
            # we don't necessarily have all the context right now to know what is acceptable.
            # TODO: Also builtins like abs, str etc need to work
            if inspect.ismodule(operand.value):
                return py_exprs.Module(operand.value)
            else:
                return py_exprs.PyObject(operand.value)
        else:
            if isinstance(operand, Undefined):
                return py_exprs.PyObject(None)
            assert isinstance(operand, VariableRef)
            if operand.name not in memo:
                raise ValueError(
                    f"Variable {operand.name} referenced before def. Defined vars: {memo.keys()}"
                )
            return memo[operand.name]

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

    def _resolve_phi_to_expr(
        self,
        args: list[
            tuple[BasicBlock, Operand]
        ],  # the inputs to the phi function and where they come from
        branch_block: BasicBlock,
        merge_block: BasicBlock,
        memo: dict,
    ) -> expression.Expression:
        if isinstance(branch_block.then, BinaryBranchCondition):
            branch = branch_block.then
            if branch.predicate == "IS TRUE":
                pred_expr = ops.AsTypeOp(bigframes.dtypes.BOOL_DTYPE).as_expr(
                    self._operand_to_expr(branch.target, memo)
                )
            else:
                pred_expr = ops.isnull_op.as_expr(
                    self._operand_to_expr(branch.target, memo)
                )

            if branch.true_case == merge_block:
                true_expr = next(
                    self._operand_to_expr(op, memo)
                    for block, op in args
                    if block == branch_block
                )
            else:  # recursive call
                true_expr = self._resolve_phi_to_expr(
                    args, branch.true_case, merge_block, memo
                )

            if branch.false_case == merge_block:
                false_expr = next(
                    self._operand_to_expr(op, memo)
                    for block, op in args
                    if block == branch_block
                )
            else:  # recursive call
                false_expr = self._resolve_phi_to_expr(
                    args, branch.false_case, merge_block, memo
                )

            return ops.where_op.as_expr(true_expr, pred_expr, false_expr)
        elif isinstance(branch_block.then, Goto):
            if branch_block.then.to == merge_block:
                return next(
                    self._operand_to_expr(op, memo)
                    for block, op in args
                    if block == branch_block
                )
            return self._resolve_phi_to_expr(
                args, branch_block.then.to, merge_block, memo
            )
        raise ValueError("Invalid branch block")

    def _is_dominated_by(
        self, block: BasicBlock, potential_dominator: BasicBlock
    ) -> bool:
        """Checks if `block` is dominated by `potential_dominator`."""
        return potential_dominator in block.dominators


def print_expr_tree(expr: expression.Expression, indent=0):
    # Print the current node's value with indentation
    str = " " * indent + print_expr_node(expr) + "\n"
    # Recursively call the function on each child node
    for child in expr.children:
        str += print_expr_tree(child, indent + 4)  # Increase indentation for children
    return str


def print_expr_node(expr: expression.Expression) -> str:
    if isinstance(expr, expression.OpExpression):
        return f"op: {expr.op}"
    elif isinstance(expr, py_exprs.GetAttr):
        return f"attr: {expr.attr}"
    else:
        return str(expr)
