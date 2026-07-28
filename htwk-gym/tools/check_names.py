"""Catch undefined names before they reach the GPU.

`python -m py_compile` and `ast.parse` only check syntax, so a NameError sits
quietly until the code actually executes that branch. That is exactly how
`CATEGORY_PATH` shipped into eval_goal_pose.summarize() on 2026-07-27: the file
parsed fine, the smoke path never touched it, and the re-eval died 100 seconds
into its first candidate on a busy shared server.

This walks each function, collects what is genuinely in scope (module globals,
imports, parameters, assignments, comprehension targets, walrus, except-as,
with-as, global/nonlocal declarations, class attributes) and flags loads of
anything else. Deliberately conservative: it reports only names that resolve
nowhere, so a hit is almost always real.

Usage:
    python tools/check_names.py                # every tracked .py
    python tools/check_names.py eval_goal_pose.py
"""

import ast
import builtins
import os
import sys


BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__"}


class ScopeCollector(ast.NodeVisitor):
    """Names bound anywhere in a function body (Python has no block scope)."""

    def __init__(self):
        self.bound = set()

    def _bind(self, target):
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                self.bound.add(node.id)
            elif isinstance(node, (ast.Starred, ast.Tuple, ast.List)):
                continue

    def visit_Assign(self, node):
        for t in node.targets:
            self._bind(t)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._bind(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._bind(node.target)
        self.generic_visit(node)

    def visit_NamedExpr(self, node):       # walrus
        self._bind(node.target)
        self.generic_visit(node)

    def visit_For(self, node):
        self._bind(node.target)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_comprehension(self, node):
        self._bind(node.target)
        self.generic_visit(node)

    def visit_withitem(self, node):
        if node.optional_vars is not None:
            self._bind(node.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for a in node.names:
            self.bound.add(a.asname or a.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for a in node.names:
            self.bound.add(a.asname or a.name)
        self.generic_visit(node)

    def visit_Global(self, node):
        self.bound.update(node.names)

    def visit_Nonlocal(self, node):
        self.bound.update(node.names)

    def visit_FunctionDef(self, node):
        self.bound.add(node.name)          # nested def is a binding; don't descend
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.bound.add(node.name)

    def visit_Lambda(self, node):
        pass                               # handled separately


def params_of(node):
    a = node.args
    out = [p.arg for p in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)]
    if a.vararg:
        out.append(a.vararg.arg)
    if a.kwarg:
        out.append(a.kwarg.arg)
    return set(out)


def module_globals(tree):
    g = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            g.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        g.add(x.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            for x in ast.walk(node.target):
                if isinstance(x, ast.Name):
                    g.add(x.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                g.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                g.add(a.asname or a.name)
        elif isinstance(node, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
            # conditionally-defined module-level names still count as defined
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    g.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        for x in ast.walk(t):
                            if isinstance(x, ast.Name):
                                g.add(x.id)
                elif isinstance(sub, ast.Import):
                    for a in sub.names:
                        g.add(a.asname or a.name.split(".")[0])
                elif isinstance(sub, ast.ImportFrom):
                    for a in sub.names:
                        g.add(a.asname or a.name)
    return g


def check_format_calls(tree):
    """`"{a} {b}".format(a=1)` -- a KeyError that only fires when the line runs.

    check_names catches undefined NAMES; this catches undefined format
    PLACEHOLDERS, which is a different failure with the same shape. It cost an
    overnight G-batch launch on 2026-07-27: make_v7_arms printed a command with
    {task} but never passed task=, so config generation died and the smoke gate
    correctly refused to train anything.

    Only checks all-keyword .format() on a literal string -- the case where the
    answer is unambiguous. Positional or *args/**kwargs forms are skipped rather
    than guessed at.
    """
    import string
    problems = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "format"):
            continue
        # the literal may be a chain of implicitly-concatenated strings
        target = node.func.value
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            template = target.value
        elif isinstance(target, ast.BinOp) and isinstance(target.op, ast.Add):
            parts = [n.value for n in ast.walk(target)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            template = "".join(parts)
        else:
            continue
        if node.args or any(k.arg is None for k in node.keywords):
            continue                      # positional or **kwargs: cannot judge
        if not node.keywords:
            continue
        given = {k.arg for k in node.keywords}
        needed = set()
        try:
            for _, field, _, _ in string.Formatter().parse(template):
                if field:
                    needed.add(field.split(".")[0].split("[")[0])
        except ValueError:
            continue
        if not needed:
            continue
        missing = {f for f in needed if f and not f.isdigit()} - given
        if missing:
            problems.append((node.lineno, sorted(missing)))
    return problems


def check_imports(path, root):
    """`from utils.runner import get_task_class` -- but is it actually there?

    check_names validates names WITHIN a file and check_format_calls validates
    format placeholders; neither looks across module boundaries, so a plausible
    but wrong import sails through both. tools/diag_reset.py shipped with
    `from envs import get_task_class` (it lives in utils.runner) and the failure
    surfaced only after Isaac Gym had finished loading on the training server --
    the second time an import-level mistake cost a full GPU round trip.

    Only local modules are checked: if the module resolves to a file under the
    repo it is parsed and its module-level names compared, otherwise it is a
    stdlib or site-packages import and left alone. A target containing
    `from x import *` is skipped, since anything could be re-exported.
    """
    problems = []
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except SyntaxError:
        return problems
    cache = {}

    def resolve(mod):
        if mod in cache:
            return cache[mod]
        rel = mod.replace(".", os.sep)
        for cand in (os.path.join(root, rel + ".py"),
                     os.path.join(root, rel, "__init__.py")):
            if os.path.exists(cand):
                try:
                    t = ast.parse(open(cand, encoding="utf-8").read())
                except SyntaxError:
                    cache[mod] = None
                    return None
                if any(isinstance(n, ast.ImportFrom)
                       and any(a.name == "*" for a in n.names) for n in ast.walk(t)):
                    cache[mod] = None          # re-exports unknowable
                    return None
                names = module_globals(t)
                # submodules are importable by name too
                pkg = os.path.join(root, rel)
                if os.path.isdir(pkg):
                    names |= {x[:-3] for x in os.listdir(pkg) if x.endswith(".py")}
                    names |= {x for x in os.listdir(pkg)
                              if os.path.isdir(os.path.join(pkg, x))}
                cache[mod] = names
                return names
        cache[mod] = None
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        if not node.module or any(a.name == "*" for a in node.names):
            continue
        names = resolve(node.module)
        if names is None:
            continue
        for a in node.names:
            if a.name not in names:
                problems.append((node.lineno, node.module, a.name))
    return problems


def check(path, star_import_seen=False):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    # `from x import *` makes any name potentially defined -- stay silent rather
    # than emit noise we would learn to ignore.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            star_import_seen = True
    g = module_globals(tree) | BUILTINS
    problems = []

    NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

    def own_nodes(fn):
        """Nodes belonging to THIS function, not to functions nested in it.

        Without this the parent sees a nested function's locals as undefined
        (and vice versa), which is pure noise -- a checker that cries wolf is one
        we stop reading, which defeats the point of having it.
        """
        stack = list(fn.body)
        while stack:
            node = stack.pop()
            if isinstance(node, NESTED):
                continue          # neither yield it nor descend: not our scope
            yield node
            for child in ast.iter_child_nodes(node):
                stack.append(child)

    def nested_of(fn):
        stack = list(fn.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                yield node
                continue
            for child in ast.iter_child_nodes(node):
                stack.append(child)

    def walk_fn(fn, enclosing):
        if isinstance(fn, ast.Lambda):
            scope = enclosing | params_of(fn)
            body = [fn.body]
            name = "<lambda>"
        else:
            sc = ScopeCollector()
            for stmt in fn.body:
                sc.visit(stmt)
            scope = enclosing | params_of(fn) | sc.bound
            body = None
            name = fn.name

        if body is not None:                       # lambda: single expression
            for node in body:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                        if sub.id not in scope:
                            problems.append((sub.lineno, name, sub.id))
        else:
            for node in own_nodes(fn):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id not in scope:
                        problems.append((node.lineno, name, node.id))
            for child in nested_of(fn):
                walk_fn(child, scope)               # closures see the enclosing scope

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            walk_fn(node, g)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    walk_fn(sub, g)
    return problems, star_import_seen


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    targets = sys.argv[1:]
    if not targets:
        targets = []
        for d in (".", "tools", "envs/K1", "utils"):
            full = os.path.join(root, d)
            if os.path.isdir(full):
                targets += [os.path.join(d, f) for f in sorted(os.listdir(full))
                            if f.endswith(".py")]
    bad = 0
    for rel in targets:
        p = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if not os.path.exists(p):
            continue
        try:
            problems, star = check(p)
        except SyntaxError as e:
            print("SYNTAX  {}:{} {}".format(rel, e.lineno, e.msg))
            bad += 1
            continue
        for lineno, mod, name in check_imports(p, root):
            print("IMPORT     {}:{}  '{}' 안에 '{}' 없음".format(rel, lineno, mod, name))
            bad += 1
        for lineno, fields in check_format_calls(ast.parse(open(p, encoding="utf-8").read())):
            print("FORMAT     {}:{}  .format() 인자 누락 -> {}".format(rel, lineno, ", ".join(fields)))
            bad += 1
        for lineno, fname, name in problems:
            note = "  (파일에 `import *`가 있어 오탐일 수 있음)" if star else ""
            print("UNDEFINED  {}:{}  in {}()  -> {}{}".format(rel, lineno, fname, name, note))
            bad += 1
    print("\n{} 개 파일 검사, 문제 {}건".format(len(targets), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
