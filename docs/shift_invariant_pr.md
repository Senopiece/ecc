# Unknown-prefix invariant PR protocol

The notebook is [x_ecc/shift_invariant_pr.ipynb](../notebooks/x_ecc/shift_invariant_pr.ipynb).
The implementation uses one carrier-generation method and one phase-repair method,
chosen in the [theoretical time and memory analysis](shift_invariant_pr_methods.md).
The construction recovers the original K-bit message from a noisy length-N window
whose starting offset S is unknown to the decoder. It combines independent primitive
carriers in one binary extension field and reserves one field block to recover phase.
This multi-carrier construction is experimental; the cited papers motivate PR encoding
and OSD, rather than establish performance guarantees for this particular extension.

## Degree, zero padding and transmitted length

Let C=32 and t=log2(CK). Take the three integers closest to t, breaking equal-distance
ties toward the smaller integer. Degrees must be at least 2 in this implementation.
For each candidate k, let p(k)=(-K) mod k. Choose the lexicographic minimum

```
(p(k) + k, p(k), k).
```

Here k (also called n in the field implementation) is the component degree, not the
payload length. The first objective minimizes zero padding plus the phase block.

| K | Candidates | Selected k | Padding p | Full dimension D=K+p+k | Data + pilot blocks |
|---:|:---|---:|---:|---:|---:|
| 32 | 9, 10, 11 | 11 | 1 | 44 | 3 + 1 |
| 64 | 10, 11, 12 | 11 | 2 | 77 | 6 + 1 |
| 128 | 11, 12, 13 | 13 | 2 | 143 | 10 + 1 |
| 256 | 12, 13, 14 | 13 | 4 | 273 | 20 + 1 |

Append p zero bits to the payload, split into k-bit blocks, then append the known
field element 1 as a pilot. Its coordinate bits are **[1, 0, ..., 0]**.
No padding parity equations or additional noiseless observations are supplied to OSD.
After phase repair, truncate to K bits without inspecting the discarded padding.
A future protocol may check it, but this implementation deliberately does not.

For sampled overhead h, transmit **N=ceil((1+h)(K+p+k))** symbols. Thus the channel
pays for the padding as well as the pilot. The table stores sampled `Overhead`, integer
`N`, full dimension `D`, and `RealizedOverhead=N/D-1` to expose ceiling effects.

## Carrier family and bit convention

Work in F=GF(2^k), represented in the polynomial basis of one primitive modulus.
Let a be the residue class of x, of multiplicative order M=2^k-1. Message coordinates
are little-endian: bit j is the coefficient of a^j in the field element.

Choose ascending exponents d coprime to M, keeping the smallest representative from
each Frobenius orbit {d,2d,4d,...} mod M. The resulting a^d have distinct primitive
minimal polynomials. There are phi(M)/k carriers: 176 for k=11 and 630 for k=13.
Using successive prefixes of this list gives the requested nested carrier families.
The first carrier is a. All K values of the same degree share this family.

The encoder uses the **trace pairing**, not an unqualified coefficient dot product:

```
y_t = sum_i Tr(m_i * a^(d_i*t)) mod 2.
G[(i*k+j), t] = Tr(a^j * a^(d_i*t)).
```

This fixes the coordinate convention so that the shift acts on the message field
elements themselves. An arbitrary coordinate dot product would instead introduce
the dual basis into the phase-repair formula. The ordinary PR trace representation
and primitive generator viewpoint are described in
[Primitive Rateless Codes (2021)](https://arxiv.org/abs/2107.05774).

Prepare trace masks through `S_max+N_max` once. Encoding a trial reads only columns
`S:S+N`; it does not evaluate or allocate the removed symbols. Each component-column
is a k-bit mask; evaluating a bit uses a cached binary parity lookup. OSD receives
the same family's unshifted generator, packed across channel coordinates into uint64
words. Encoding and decoding do not independently regenerate carriers or prefixes.

## OSD and phase repair

For channel input LLRs, OSD sorts coordinates by descending absolute reliability,
greedily selects an independent most reliable basis (MRB), and performs packed GF(2)
Gaussian elimination. Row operations also transform a packed identity matrix so
the decoded bits are returned in the original field-coordinate basis.

The baseline candidate matches channel hard decisions on the MRB. Candidates flip
up to `OSD_ORDER` MRB decisions, re-encode, and minimize weighted Hamming distance
`sum(abs(LLR[j]) for mismatching coordinates j)`. The implementation supports orders
0, 1 and 2; defaults are order 1, the full MRB, and a cap of 4096 candidates including
the baseline. A positive `OSD_SEARCH_BITS` restricts flips to the least reliable
part of the MRB. Capped order-2 enumeration is lexicographic in these reliability
indices, not an exhaustive order-2 search when the cap is reached.

The reliability-basis approach follows conventional OSD as discussed in
[Liang and Ma (2024)](https://arxiv.org/html/2401.16709v1).
The more PR-specific
[Low-complexity Ordered Statistic Decoder for Primitive Rateless Codes (ICC 2025)](https://zenodo.org/records/16911224)
uses self-dual bases to simplify high-rate single-carrier PR decoding. Our multi-carrier
code uses general packed elimination; it does **not** claim to implement that paper's
specialized self-dual-basis decoder or its complexity bound.

Since

```
y_(S+t) = sum_i Tr((m_i * a^(S*d_i)) * a^(t*d_i)),
```

the unshifted OSD decoder returns transformed blocks `m'_i=m_i*a^(S*d_i)` when
successful. Let d_p be the pilot carrier exponent and q' the decoded pilot block.
The original pilot is 1. Precompute `inverse_d_p = d_p^(-1) mod M` once per layout:

```
s_mod = log_a(q') * inverse_d_p mod M
m_i = exp_a((log_a(m'_i) - s_mod*d_i) mod M)  # nonzero blocks
```

Zero data blocks remain zero. A zero decoded pilot or a rank-deficient generator
returns failure (`None` in the single-observation interface). The modular inverse
of d_p is unrelated to the multiplicative field inverse of q'. The former is computed
once using integer modular inversion; neither inverses nor discrete logarithm
searches are computed per data block. `log_a` is a precomputed lookup table.

Only S modulo M can be identified. This suffices to undo all carrier phases even
when S exceeds a period. The decoder never receives the actual S, the true message,
or a padding check. A nonzero pilot does not certify correctness: `valid` only means
OSD had full rank and phase repair was defined; `succ` means exact K-bit recovery.

## Experiment and extension interface

Defaults: 1,000,000 trials; K uniform in (32,64,128,256); overhead uniform in (0.1,0.5);
SNR uniform in (-3,3) dB; S uniform over integers 0..4095 inclusive; random independent
payloads; uniform protocol choice. The finite S range is a configurable default.
Es=1, SNR=Es/N0, noise variance=1/(2*10^(SNR/10)), and LLR=2*received/variance.

`core/protocols` contains prepared encoder/decoder pairs. Add a spec to `PROTOCOLS`
with a unique `name` and `prepare(message_sizes, max_shift=..., max_overhead=...)`.
It returns `{K: prepared}`. A prepared instance exposes `dimension`, `max_length`
and `simulate(messages, lengths, shifts, snr_db, standard_normal_noise)`, returning
one boolean success and validity flag per trial. Internally its decoder must not
use message truth or the sampled shift. These are only available to encoding and
the simulation's final equality comparison. The current prepared protocol also
provides independent `.encode(message, length=..., shift=...)` and `.decode(llr)`.

`application/protocol_experiment.py` samples parameters and assembles records.
`presentation/protocol_plots.py` renders K-Overhead, SNR-K and Overhead-SNR heatmaps with protocol choice
and independent Swap X/Y toggles. K is a discrete axis. Each view filters the third variable or selects ALL:
SNR bins for K-Overhead, Overhead bins for SNR-K, and individual K values for
Overhead-SNR. Continuous filter intervals use the corresponding plot bin edges,
with the right endpoint included only in the final bin. ALL pools success counts and observation
counts, rather than averaging per-K percentages. Empty bins are NaN, not zero success.

Numba compiles the entire numerical batch with the GIL released. Joblib threads
share immutable banks, avoiding per-process copies and repeated preparation. The
notebook has an explicit compilation cell before timing. Chunk-indexed seeds make
trial data independent of worker count and task completion order. Changing batch
size or the protocol list changes the random stream. Each invocation computes all trials and returns a table in notebook memory;
no experiment files are written or loaded.

## Cost and scope

Writing b=D/k and L=S_max+N_max, trace masks cost O(DL) one-time work and O(bL)
machine words. Each encoding costs O(bN). Packed generator storage costs
O(D*ceil(N_max/64)) words. Per-observation OSD elimination costs at most
O(D^2*ceil((N+D)/64) + DN) word operations, plus sorting O(N log N).
Q candidate scores cost O(Q*ceil(N/8)); byte-weight table preparation costs O(32N).
Phase repair costs O(b), plus O(D) unpacking, with shared exp/log tables.

These field tables intentionally target small degrees (supported range 2..20).
They consume O(2^k) memory; this strategy is not suitable for degree 32.
That limitation is specific to this protocol, not the existing sparse LFSR encoder.
See [theoretical analysis](shift_invariant_pr_methods.md) for complexity and
the tradeoff against polynomial enumeration and field exponentiation.
