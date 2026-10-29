"""Berquist-Sherman style adjustment for a change in settlement speed.

The classic Berquist-Sherman (1977) paid adjustment restates historical paid
losses to the *current* disposal rate using closed claim counts.  Schedule P
in the CAS database has no claim counts, so this module uses incurred losses
as the exposure base instead:

    P_adj[i, j] = I[i, j] * r_j,   r_j = P/I ratio at age j on the latest diagonal

Every historical cell is restated to the paid-to-incurred ratio that the
most recent accident year showed at that age, and the latest diagonal is
left untouched.  Paid chain ladder on the restated triangle then projects
with development factors consistent with today's payment speed.

The proxy assumes case reserve adequacy is stable, so that a moving P/I
ratio reflects settlement speed.

**Result: without counts the adjustment adds nothing.**  Because every
column is a constant multiple r_j of the incurred column, the adjusted paid
factors are f^I_j * r_{j+1} / r_j, and the latest paid value is I * r_a for
the same accident year that set r_a.  The projection telescopes to

    U_adj = I_latest * CDF^I(a) * r_last  =  U_CL,incurred * r_last

and the paid-to-incurred tail 1 / r_last turns that back into the incurred
chain ladder ultimate *exactly*.  A settlement adjustment that uses incurred
losses as its yardstick can only reproduce the incurred method; the
independent information in Berquist-Sherman comes from claim counts.
(``tests/test_extensions.py`` checks the identity.)
"""

from __future__ import annotations

import numpy as np

from .triangle import Triangle


def settlement_adjusted_paid(paid: Triangle, incurred: Triangle) -> Triangle:
    n = paid.n
    P, I = paid.values, incurred.values
    r = np.array([P[n - 1 - j, j] / I[n - 1 - j, j] for j in range(n)])  # latest-diagonal ratio by age
    adj = I * r[None, :]
    for i in range(n):  # keep the actual latest diagonal
        adj[i, n - 1 - i] = P[i, n - 1 - i]
    return Triangle.from_square(adj, paid.origins, "paid (settlement-adjusted)")
