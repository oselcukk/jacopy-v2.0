"""Tests for jacopy.brackets.lie."""

import pytest

from jacopy.algorithms.collect_terms import collect_terms
from jacopy.algorithms.simplify import simplify
from jacopy.brackets.base import BracketApply
from jacopy.brackets.lie import LieBracket, lie
from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry


def _expand_all_brackets(e, registry):
    """Recursively expand every BracketApply in the tree."""
    if hasattr(e, "children") and e.children:
        new = tuple(_expand_all_brackets(c, registry) for c in e.children)
        if isinstance(e, BracketApply):
            return BracketApply(e.bracket, new[0], new[1]).expand(registry)
        return type(e)(*new)
    return e


@pytest.fixture
def reg():
    r = PropertyRegistry()
    # Lie bracket conventionally takes vector fields; we model them as
    # degree-0 symbols so the obstruction helpers can compute parities.
    for s in ("X", "Y", "Z"):
        r.declare(Symbol(s), Graded(degree=0))
    return r


class TestExpansion:
    def test_basic(self):
        X, Y = Symbol("X"), Symbol("Y")
        assert lie(X, Y).expand() == Sum(Product(X, Y), Neg(Product(Y, X)))

    def test_bracket_with_self_collapses(self):
        """[X, X] = X*X − X*X → 0 after collect_terms."""
        X = Symbol("X")
        assert collect_terms(lie(X, X).expand()) == Integer(0)


class TestAxioms:
    def test_antisymmetry_holds_on_vector_pair(self, reg):
        """[X, Y] + [Y, X] → 0 once each [·,·] is expanded and the
        outer Neg-over-Sum is distributed by canonicalize."""
        X, Y = Symbol("X"), Symbol("Y")
        obs = lie.graded_antisymmetry_obstruction(X, Y, reg)
        # Expand every BracketApply in place, then simplify. The
        # canonicalize pass (invoked inside simplify) is the one that
        # pushes Neg through Sum so cancellation reaches every term.
        expanded = _expand_all_brackets(obs, reg)
        assert simplify(expanded) == Integer(0)

    def test_jacobi_holds_on_vector_triple(self, reg):
        """Plan-required test: ``[[X,Y],Z] + [[Y,Z],X] + [[Z,X],Y] = 0``.

        The Lie bracket's Jacobi identity is a theorem of associativity
       , for vector-field symbols whose product is associative and
        non-commutative, expanding every commutator layer and
        collecting like terms must leave nothing behind.
        """
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        obs = lie.graded_jacobi_obstruction(X, Y, Z, reg)
        expanded = _expand_all_brackets(obs, reg)
        assert simplify(expanded) == Integer(0)

    def test_module_singleton_is_lie(self):
        """The `lie` re-export is a LieBracket with default name."""
        assert isinstance(lie, LieBracket)
        assert lie.name == "[·,·]"
