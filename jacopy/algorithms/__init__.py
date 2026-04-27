"""Algorithms layer: rewriting, canonical form, graded sorting."""

from jacopy.algorithms.base import Algorithm, StepResult
from jacopy.algorithms.canonicalize import (
    canonical_hash,
    canonicalize,
    semantically_equal,
)
from jacopy.algorithms.collect_terms import CollectTerms, collect_terms
from jacopy.algorithms.distribute import Distribute, distribute
from jacopy.algorithms.flatten import Flatten, flatten
from jacopy.algorithms.product_rule import ProductRule, product_rule
from jacopy.algorithms.rewrite import (
    Rule,
    apply_bottomup,
    apply_once_at_root,
    apply_topdown,
    normalize,
)
from jacopy.algorithms.simplify import simplify
from jacopy.algorithms.sort_product import apply_sign, sort_product

__all__ = [
    # base
    "Algorithm",
    "StepResult",
    # rewrite
    "Rule",
    "apply_once_at_root",
    "apply_bottomup",
    "apply_topdown",
    "normalize",
    # flatten
    "flatten",
    "Flatten",
    # distribute
    "distribute",
    "Distribute",
    # canonicalize
    "canonicalize",
    "canonical_hash",
    "semantically_equal",
    # collect_terms
    "collect_terms",
    "CollectTerms",
    # sort_product
    "sort_product",
    "apply_sign",
    # product_rule
    "product_rule",
    "ProductRule",
    # simplify
    "simplify",
]
