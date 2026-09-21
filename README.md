# Implicit PDE inversion

**Recover an unknown coefficient field of a PDE from sparse, indirect
observations — by representing the field as a coordinate network and training
it so the physics it implies reproduces what was actually measured.**

![sweeping 26 cross sections](animations/section_scan.gif)

## The inverse problem

Many measurement problems have the same shape. A field you care about is buried
inside a PDE and cannot be observed directly; what you *can* observe is a
downstream consequence of it, on a boundary, sparsely. Recovering the field
means inverting the PDE.

This repository is one instance of that: the unknown is the bed elevation of a
channel, and the observable is the velocity of the fluid at the free surface.
The same structure appears in seismic imaging — see the companion repositories
below, where the unknown is a wave-speed model and the observable is a
seismogram at the surface.

## The method

Two ingredients, both reusable across the applications:

**1. The unknown is a network, not a grid.** The field is written as a small
coordinate network `x -> m(x)` and the *weights* are the optimisation
variables. This regularises by construction — the network cannot represent
grid-scale noise — and decouples the number of unknowns from the mesh.

**2. The PDE is the loss.** Rather than fit the field to a reference, the
residual of the governing equation is minimised alongside the data misfit, so
the recovered field is one the physics admits, not merely one that interpolates
the measurements.

Here the governing relation is a depth-averaged momentum balance of
Shiono–Knight type; in the seismic repositories it is the acoustic or elastic
wave equation, and the gradient comes from an adjoint-state solve instead of
autodiff. The outer structure is identical.

## Why the problem is hard, and what fixes it

The inverse problem is **ill-posed**: the physics alone does not determine the
answer. The interesting result here is how little extra information is needed
to close it.

| constraint added | what it determines |
|---|---|
| PDE residual only | the shape of the field, but not its scale |
| \+ one integral quantity (total flux) | the scale |
| \+ one or two point measurements | the residual ambiguity |

So a handful of measurements substitutes for a dense survey. That ratio —
sparse constraints in, full field out — is what makes the approach worth
something operationally.

## Validation strategy

Three independent checks, of the kind an inverse solution needs before anyone
should trust it:

- **Against a conventional forward model**, at locations where the true field
  was surveyed.
- **Across two different operating conditions**, where the recovered field is
  physically the same object and must come out the same both times.
- **Across three independent observation sources**, where the answer must not
  depend on which instrument produced the input.

## Status

The animation here was rendered for the web from result files. The quantitative
figures belong to a manuscript in preparation and are held back until it is
published; they will be added then.

`code/make_gifs.py` renders the animation. Its input CSVs are not in this
repository.

## Companion repositories — same method, different PDE

| | unknown field | governing PDE | gradient |
|---|---|---|---|
| **this repo** | bed elevation | depth-averaged momentum | autodiff |
| [implicit-elastic-fwi](https://github.com/anhle156/implicit-elastic-fwi) | vp, vs, rho | elastic wave equation | hand-written adjoint |
| [implicit-acoustic-fwi](https://github.com/anhle156/implicit-acoustic-fwi) | vp | acoustic wave equation | adjoint-state |

---

Anh Le · [github.com/anhle156](https://github.com/anhle156)
