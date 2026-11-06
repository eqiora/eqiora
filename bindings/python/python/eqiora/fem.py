"""Closed finite-element spatial policies with executable Eqiora consumers."""

from ._eqiora import GaussLegendre, SimplexDuffyGaussLegendre, MiniP1, P1, Q1, Q2, ScopedSpatialPolicy, TetrahedralEdge, TetrahedralFace

__all__ = ["GaussLegendre", "SimplexDuffyGaussLegendre", "MiniP1", "P1", "Q1", "Q2", "ScopedSpatialPolicy", "TetrahedralEdge", "TetrahedralFace"]
