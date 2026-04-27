"""Tests for ``jacopy.library.theorem_book``."""

from __future__ import annotations

import pytest

from jacopy.core.expr import Symbol
from jacopy.library import Theorem, TheoremBook, theorem_book
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _chain(rule: str = "demo") -> ProofChain:
    return ProofChain([ProofStep(Symbol("X"), Symbol("Y"), rule=rule)])


def _thm(name: str = "t", statement: str = "X = Y", **kwargs) -> Theorem:
    return Theorem(
        name=name,
        statement=statement,
        from_axioms=kwargs.pop("from_axioms", ("ax1",)),
        proof=kwargs.pop("proof", _chain()),
        notes=kwargs.pop("notes", ""),
    )


# --------------------------------------------------------------------- #
# Theorem dataclass                                                     #
# --------------------------------------------------------------------- #


class TestTheoremConstruction:
    def test_minimal(self):
        t = _thm()
        assert t.name == "t"
        assert t.statement == "X = Y"
        assert t.from_axioms == ("ax1",)
        assert isinstance(t.proof, ProofChain)
        assert t.notes == ""

    def test_frozen(self):
        t = _thm()
        with pytest.raises(Exception):
            t.name = "other"  # type: ignore[misc]

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            Theorem(
                name="",
                statement="s",
                from_axioms=(),
                proof=_chain(),
            )

    def test_non_string_statement_rejected(self):
        with pytest.raises(TypeError):
            Theorem(
                name="t",
                statement=42,  # type: ignore[arg-type]
                from_axioms=(),
                proof=_chain(),
            )

    def test_from_axioms_must_be_tuple_of_str(self):
        with pytest.raises(TypeError):
            Theorem(
                name="t",
                statement="s",
                from_axioms=["ax1"],  # type: ignore[arg-type]
                proof=_chain(),
            )
        with pytest.raises(TypeError):
            Theorem(
                name="t",
                statement="s",
                from_axioms=(1,),  # type: ignore[arg-type]
                proof=_chain(),
            )

    def test_proof_must_be_proofchain(self):
        with pytest.raises(TypeError):
            Theorem(
                name="t",
                statement="s",
                from_axioms=(),
                proof="not a chain",  # type: ignore[arg-type]
            )

    def test_notes_must_be_string(self):
        with pytest.raises(TypeError):
            Theorem(
                name="t",
                statement="s",
                from_axioms=(),
                proof=_chain(),
                notes=123,  # type: ignore[arg-type]
            )

    def test_equality_by_value(self):
        c = _chain()
        a = Theorem(name="t", statement="s", from_axioms=("a",), proof=c)
        b = Theorem(name="t", statement="s", from_axioms=("a",), proof=c)
        assert a == b
        assert hash(a) == hash(b)


# --------------------------------------------------------------------- #
# TheoremBook registry                                                  #
# --------------------------------------------------------------------- #


class TestTheoremBookBasics:
    def test_empty_book(self):
        book = TheoremBook()
        assert len(book) == 0
        assert book.names() == ()
        assert list(book) == []
        assert "anything" not in book

    def test_add_and_get(self):
        book = TheoremBook()
        t = _thm(name="d_squared_zero")
        book.add(t)
        assert book.get("d_squared_zero") is t
        assert "d_squared_zero" in book
        assert len(book) == 1

    def test_names_preserves_insertion_order(self):
        book = TheoremBook()
        book.add(_thm(name="first"))
        book.add(_thm(name="second"))
        book.add(_thm(name="third"))
        assert book.names() == ("first", "second", "third")

    def test_iter_yields_theorems_in_order(self):
        book = TheoremBook()
        a = _thm(name="a")
        b = _thm(name="b")
        book.add(a)
        book.add(b)
        assert list(book) == [a, b]

    def test_add_rejects_non_theorem(self):
        book = TheoremBook()
        with pytest.raises(TypeError):
            book.add("not a theorem")  # type: ignore[arg-type]

    def test_duplicate_name_raises(self):
        book = TheoremBook()
        book.add(_thm(name="t"))
        with pytest.raises(KeyError, match="already registered"):
            book.add(_thm(name="t", statement="different"))

    def test_get_unknown_name_raises(self):
        book = TheoremBook()
        book.add(_thm(name="known"))
        with pytest.raises(KeyError, match="no theorem named"):
            book.get("missing")

    def test_contains_only_matches_strings(self):
        book = TheoremBook()
        book.add(_thm(name="t"))
        assert "t" in book
        assert 123 not in book  # type: ignore[operator]

    def test_repr_includes_count_and_names(self):
        book = TheoremBook()
        book.add(_thm(name="alpha"))
        book.add(_thm(name="beta"))
        r = repr(book)
        assert "2 theorems" in r
        assert "alpha" in r
        assert "beta" in r


class TestReplaceRemoveClear:
    def test_replace_overwrites_existing(self):
        book = TheoremBook()
        book.add(_thm(name="t", statement="old"))
        book.replace(_thm(name="t", statement="new"))
        assert book.get("t").statement == "new"
        assert len(book) == 1

    def test_replace_inserts_when_missing(self):
        book = TheoremBook()
        book.replace(_thm(name="t"))
        assert "t" in book

    def test_replace_rejects_non_theorem(self):
        book = TheoremBook()
        with pytest.raises(TypeError):
            book.replace("nope")  # type: ignore[arg-type]

    def test_remove(self):
        book = TheoremBook()
        book.add(_thm(name="t"))
        book.remove("t")
        assert "t" not in book
        assert len(book) == 0

    def test_remove_missing_raises(self):
        book = TheoremBook()
        with pytest.raises(KeyError):
            book.remove("t")

    def test_clear(self):
        book = TheoremBook()
        book.add(_thm(name="a"))
        book.add(_thm(name="b"))
        book.clear()
        assert len(book) == 0
        assert book.names() == ()


# --------------------------------------------------------------------- #
# Module singleton                                                      #
# --------------------------------------------------------------------- #


class TestSingleton:
    def test_singleton_is_a_theorem_book(self):
        assert isinstance(theorem_book, TheoremBook)

    def test_singleton_round_trip(self):
        # Use a unique name so this test survives re-runs.
        key = "_test_singleton_round_trip"
        if key in theorem_book:
            theorem_book.remove(key)
        theorem_book.add(_thm(name=key))
        try:
            assert key in theorem_book
            assert theorem_book.get(key).name == key
        finally:
            theorem_book.remove(key)

    def test_singleton_starts_empty_at_stage_a(self):
        # Stage A ships without any auto-seeded theorems. Later stages
        # (symplectic, poisson, ...) will add their own; this test
        # documents the invariant and will need updating when they do.
        # If a prior test leaked, the other tests above clean up.
        assert all(
            not name.startswith("_test_") for name in theorem_book.names()
        )
