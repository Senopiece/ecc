# Theoretical choice of PR implementation

This document compares time and space bounds, not measured runtimes. The production
implementation contains one carrier-generation algorithm and one phase-repair
algorithm. No timing-based dispatch or alternative implementations are retained.
See [the protocol](shift_invariant_pr.md) for the encoding and decoding equations.

## Model and workload

- n: component degree; M=2^n-1: multiplicative period.
- R=phi(M)/n: number of inequivalent primitive carriers.
- b: number of active blocks, including the pilot; D=bn: full binary dimension.
- E: number of transmissions sharing a field and carrier family.
- L=S_max+N_max: prepared encoder prefix length.
- omega(M): number of distinct prime divisors of M.

Bounds below use a word-RAM model: a field element, exponent and its intermediate
integer products fit in a constant number of machine words. Space is in words
unless specified otherwise. For arbitrary-size integers, multiplication, division
and GCD are not constant-time; these bounds must then include their bit costs.
The implementation deliberately supports small degrees 2..20. The current layouts
use n=11 or 13 and E=1,000,000, so reusable O(2^n) storage is appropriate.

## Generating the nested carrier family

A carrier is represented by a primitive root a^d. Require gcd(d,M)=1 and retain the
smallest exponent in each doubling orbit modulo M. Each such orbit has n elements.
Its members correspond to one primitive minimal polynomial; there are R orbits.
Ascending representatives give nested families without constructing polynomial
coefficients or later recovering their root exponents.

The following bounds assume the distinct prime factors of M are already available.
The output itself occupies Theta(R) words when the complete family is retained.

| Method | Time for a complete family | Extra working memory, excluding output |
|---|---|---|
| Sieve eligible exponents, then mark doubling orbits | O(M + M sum_(p divides M)(1/p) + phi(M)) | O(M) |
| Scan exponents using GCD, with visited orbit marks | O(M log M + phi(M)) upper bound | O(M) |
| No visited array: test GCD and whether each orbit representative is minimal | O(M log M + phi(M)n) upper bound | O(1) |
| Enumerate monic odd polynomials and test primitiveness | O(M P(n)) | Space of one polynomial test |

P(n) is the cost of a primitive-polynomial test; its value depends on polynomial
arithmetic. For the simple shift/XOR arithmetic and sequential irreducibility tests
used to find our single field modulus, a conservative bound is
O(n^3 + omega(M)n^2) word operations per candidate. This is not an optimal bound for
specialized polynomial-enumeration algorithms. Enumerating all primitive polynomials
also does not directly provide their exponents d in the chosen common field; a root
identification step would still be needed.

The sieve clears multiples of each distinct prime divisor in O(M/p) work, scans M
entries, and marks each eligible exponent once. Its two temporary boolean arrays
occupy O(M) bytes in this implementation. Finding the prime divisors by trial division
costs O(sqrt(M)) integer operations once, with O(omega(M)) output memory.

**Choice: sieve plus doubling-orbit representatives.** We need a reusable complete
nested family, and the chosen field arithmetic already requires O(M) space. Thus
an O(M) temporary sieve does not change the total space order, while eliminating
repeated GCD tests and polynomial materialization. This is a workload-based choice,
not a claim of universal optimality: requesting only a few carriers at a much larger
n can favor a streaming enumeration with less memory. Specialized necklace generators
are another design space, not implemented here.

Only one primitive modulus is sought to define the common field. If A candidate
polynomials are examined before finding it, that setup costs O(A P(n)); the worst
case A is O(M). The implementation stops immediately at the first primitive modulus.
It does not enumerate all primitive polynomials.

If polynomial coefficients were required, multiplying n conjugate linear factors
would add O(n^2) table-field operations per carrier, O(R n^2) in total, and O(n)
temporary words. Encoding and phase repair need only d, so this work is omitted.

## Field tables and amortization

The implementation builds exp/log tables in O(M) time. Parity takes O(M) time;
the current straightforward trace construction takes O(nM). Together these use O(M)
words and O(nM) setup time, shared by every encoder, decoder and trial of that degree.
The setup amortizes to O(nM/E) per transmission, excluding the one-time modulus
search, factorization and carrier enumeration. Caching is per degree, not per trial.

The exponential dependence on n is explicit: O(M)=O(2^n). This is not a suitable
memory strategy for degree 32. A large-degree protocol would instead need polynomial
field arithmetic and a different setup/memory tradeoff. No such alternate runtime
path is included in this small-degree implementation.

## Recovering and removing phase

Let q' be the transformed pilot on carrier d_p, whose original value was 1.
Compute the integer modular inverse r=d_p^(-1) mod M once per layout.
It costs O(log M) Euclidean iterations in the word model. This is an inverse of
an exponent modulo M, not the multiplicative field inverse of q'.

With shared tables:

```
s = log[q'] * r mod M
m_i = exp[(log[m'_i] - s*d_i) mod M]    # nonzero data blocks
```

A zero data block stays zero; a zero pilot makes phase recovery undefined.
The recovered s is S modulo M, which is sufficient for every carrier.
One pilot lookup is shared by all b blocks. Each block needs a constant number of
integer operations and table accesses, so field-block correction takes Theta(b).
Packing/unpacking the binary coordinate representation separately costs O(D).

For comparison, let mu(n) be the cost of one field multiplication and I(n) the
cost of one field inversion without exp/log tables. With shift/XOR multiplication,
mu(n)=O(n) word operations; Fermat inversion takes O(n mu(n)), while a polynomial
extended-Euclidean implementation has its own I(n) bound.

| Approach | Phase-repair time per message | Shared table memory |
|---|---|---|
| exp/log tables | Theta(b), plus O(D) coordinate conversion | O(M) |
| Recover a^S=q'^r, invert once, then raise a^(-S) to each d_i | O(n mu(n) + I(n) + sum_i log(d_i) mu(n)) | None required |
| Recover a^S, form each carrier phase and invert it separately | O(n mu(n) + sum_i log(d_i) mu(n) + b I(n)) | None required |

Here log(d_i) is bounded by O(n). Without tables, each field element still occupies
O(1) words in the stated model; block storage costs O(b), plus the workspace of the
chosen arithmetic. Computing q'^r avoids solving a discrete logarithm in the
non-table alternatives, but still requires field exponentiation.

**Choice: exp/log correction.** The tables already serve encoding and are reused
across E transmissions. Phase repair adds no asymptotic shared memory and reaches
Theta(b), the output-size lower bound for b corrected field elements. The complete
binary-message path is O(D), since all decoded bits must be read/written. Retaining
inverse d_i for data carriers would not improve this: correction needs the forward
d_i multiplied by the negative phase. Only the pilot exponent's inverse is required.

## Generator preparation and OSD

Encoder masks cost O(DL) setup time and O(bL) words. Each trial evaluates only the
requested N columns in O(bN) time; the removed prefix is not encoded anew. The
unshifted OSD generator occupies O(D ceil(N_max/64)) uint64 words. These arrays are
shared across trials and compatible layouts, not rebuilt by each protocol instance.

For Q test patterns, the implemented most-reliable-basis OSD uses:

- O(N log N) reliability sorting;
- O(D^2 ceil((N+D)/64) + DN) elimination work;
- O(32N) work to build byte-weight lookup tables;
- O(Q ceil(N/8)) candidate-scoring work;
- O(D ceil((N+D)/64) + N + Q) words of scratch/pattern storage.

The constant 64 describes this packed implementation. Q is capped explicitly;
full order-t enumeration has sum_(j=0)^t binomial(D,j) patterns. Order 1 therefore
has Q=D+1, while order 2 can have quadratic candidate count. Early score termination
improves particular inputs but does not change these worst-case bounds.

## Implementation decision

Keep one path: **sieved Frobenius representatives + shared exp/log tables + packed
OSD**. Field and generator preparation are cached; the notebook has a separate
Numba compilation pass. The experiment notebook measures transmission outcomes,
not competing implementations. No empirical timing comparison is needed for this
choice, and no alternative algorithm code or benchmark notebook is retained.
