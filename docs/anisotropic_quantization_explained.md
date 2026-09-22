# The Idea Behind ScaNN: Why Compressing Vectors "Wrong" On Purpose Makes Search Better

*A plain-language walkthrough of ["Accelerating Large-Scale Inference with
Anisotropic Vector Quantization"](https://arxiv.org/abs/1908.10396) (Guo,
Sun, Lindgren, Geng, Simcha, Chern & Kumar, ICML 2020) — the paper behind
ScaNN's `anisotropic_quantization_threshold` and `avq` parameters. If you
just want to know which knob to turn, see
[docs/api_reference.md](api_reference.md); this page is about why the knob
exists at all.*

## The problem: finding a needle in 1.2 million haystacks, fast

Say you've built a recommendation system, or a semantic search engine, or a
RAG pipeline. Under the hood, it's almost certainly doing the same thing:
turning items (products, documents, chunks of text) into vectors, turning a
query into a vector too, and then asking "which of my million-plus stored
vectors has the highest dot product with this query vector?" That's
Maximum Inner Product Search (MIPS), and it's the workhorse behind
recommender systems, extreme classification, and vector databases in general.

The naive approach — compute the dot product against every single stored
vector, keep the biggest — works fine for a thousand vectors. It falls over
completely at a million, and is out of the question at a billion. So
production systems compress ("quantize") the stored vectors into a much
smaller representation, and search against the compressed versions instead.
ScaNN's `.score_ah(...)` — Asymmetric Hashing — is exactly this: instead of
storing each vector as, say, 128 floats (512 bytes), you store it as a
handful of small codebook lookups (as little as a couple of bytes). This
paper is about *how* to do that compression well.

## The old way, and why it's subtly wrong for search

The standard way to compress a set of vectors is **product quantization**:
chop each vector into a few chunks, run k-means on each chunk position
across the whole dataset, and replace every chunk with the ID of its nearest
learned "codeword." This is a well-studied, decades-old technique, and it
works by minimizing *reconstruction error* — how far, on average, is a
compressed vector from the original one it's standing in for?

That sounds like exactly the right thing to minimize. But the paper makes an
observation that, once you hear it, feels obvious: **reconstruction error
treats every vector's compression mistakes as equally important, but for
search, they aren't.** If a stored vector `x` has a low dot product with your
query `q` — it was never going to show up in your top results anyway — it
genuinely doesn't matter if its compressed version is a little wrong. But if
`x` has a *high* dot product with `q` — it's a real contender for the top
spot — then a compression error there can flip the ranking and cause you to
miss it entirely.

Standard reconstruction-error quantization doesn't know this. It spends the
exact same compression "effort" protecting the accuracy of vectors that
were never going to matter as it does on the vectors that actually decide
your search results. The paper's Figure 1 illustrates this nicely: for a
given database point `x`, its dot product with query `q1` (large) matters
far more to get right than its dot product with `q2` or `q3` (small) — yet
classic PQ can't distinguish between those cases at all.

## The fix: not all compression error is created equal

Here's the paper's central trick. Take any stored vector `x` and its
compressed approximation `x_hat`. The difference between them — the error
introduced by compression — can be split into two geometric pieces:

- **Parallel error**: the part of the error that points in the *same
  direction* as `x` itself. This changes `x`'s effective length/magnitude.
- **Perpendicular error**: the part of the error at right angles to `x`.
  This mostly rotates `x` slightly without changing how "big" it is.

Why does this split matter? Because a dot product `<q, x>` is, geometrically,
about how much `x` lines up with `q` and how long `x` is. **Error parallel to
`x` directly biases that dot product for basically every possible query**
— it makes `x` look uniformly bigger or smaller than it really is, which
throws off its rank against every other candidate. **Error perpendicular to
`x`, on the other hand, mostly averages out** across the many different
directions a query could come from — for some queries it makes the estimated
dot product slightly too high, for others slightly too low, and it rarely
causes a wrong ranking near the top.

The conclusion: **if you're going to accept some quantization error (and you
have to — that's the whole point of compression), you should push it toward
the perpendicular direction and protect the parallel direction.** That's the
"anisotropic" in anisotropic vector quantization — *anisotropic* just means
"not the same in every direction," as opposed to standard reconstruction
loss, which is *isotropic* (treats every direction equally).

## The threshold: how much do you protect?

The paper turns this into a concrete loss function with one tunable knob,
which the authors call `T` (this is exactly the
`anisotropic_quantization_threshold` parameter in ScaNN's `.score_ah()`,
`.reorder()`, and `.tree(avq=...)`). Conceptually:

- Datapoint/query pairs with a dot product **above** the threshold `T` get
  the "protect parallel error" treatment described above.
- Pairs below the threshold are treated closer to ordinary reconstruction
  loss, since they were unlikely to be a top result anyway and it's not
  worth spending accuracy on them.

Where you set `T` (relative to a vector's own norm) controls *how hard* the
loss leans into protecting parallel error versus treating all error equally.
`T = 0` recovers standard, direction-agnostic reconstruction loss (parallel
and perpendicular error weighted equally) — anisotropic quantization is
"turned off." As `T` grows, the loss cares more and more about parallel
error specifically, and less about perpendicular error, up to a limit where
it stops penalizing perpendicular error almost entirely. In ScaNN this is a
value to *sweep and tune empirically* rather than a single correct number —
the paper's own experiments (see below) land on `T = 0.2` for their
benchmark, but the right setting depends on your data and accuracy/speed
target, and `T = NaN`/unset in the Python API disables anisotropic
quantization entirely (plain reconstruction loss).

## From loss function to algorithm

A loss function alone doesn't compress anything — you need an algorithm that
optimizes it. The paper shows this new anisotropic loss slots into the same
family of algorithms used for ordinary vector quantization and product
quantization:

- It's a small, closed-form modification of **k-means**. Instead of setting
  each codeword to the plain average of the points assigned to it (as
  standard k-means/PQ training does), the update rule becomes a *weighted*
  average that leans away from the parallel-error direction. The rest of the
  algorithm — assign each point to its nearest codeword, update codewords,
  repeat until convergence — is unchanged, and it's proven to converge for
  the same reason ordinary k-means does (the loss can only decrease or stay
  flat each iteration).
- Because it's essentially "k-means with a different update rule," it drops
  into **product quantization** the same way ordinary k-means does — quantize
  each chunk of the vector independently, just using the anisotropic update
  instead of a plain average. This is precisely what ScaNN's `.score_ah(...)`
  does under the hood, and it's why AVQ (`.tree(avq=...)`) can apply the same
  idea to the coarse partition centroids used by `.tree(...)`.

In other words: this isn't a wholesale replacement for the quantization
machinery the field already had — it's a better *objective* for that
machinery to optimize, and it turns out to be a cheap, almost drop-in change.

## Does it actually help?

Yes, and by a meaningful margin. The paper's headline experiment uses
Glove-1.2M — 1.2 million 100-dimensional word-embedding vectors, a standard
public benchmark. A few results worth knowing:

- **At a fixed compressed size** (a fixed number of bits per stored vector),
  switching from ordinary reconstruction-loss quantization to
  anisotropic-loss quantization measurably improved Recall@10 — how often
  the true top-10 nearest neighbors actually showed up in the retrieved set
  — as long as the threshold `T` was chosen reasonably (their sweep found `T
  = 0.2` best for this dataset).
- It also directly improved the **accuracy of the estimated dot product
  itself** for the true top-1 match — useful anywhere you need the actual
  score, not just the ranking (e.g. softmax approximations that consume raw
  logits, which the paper calls out explicitly as a downstream use case).
- On the public [ann-benchmarks.com](http://ann-benchmarks.com) leaderboard
  methodology — which measures queries-per-second at a given recall target,
  not just an isolated recall number — the resulting system (partitioning +
  anisotropically-quantized AH + rescoring, i.e. exactly the
  `.tree()` + `.score_ah()` + `.reorder()` pipeline documented in
  [docs/api_reference.md](api_reference.md)) came out ahead of every other
  publicly benchmarked library on Glove-1.2M at the time, including FAISS
  and hnswlib.
- The gains weren't specific to one dataset or to inner-product quantization
  specifically — the paper shows the same anisotropic loss improves results
  when substituted into an unrelated technique, binary quantization, with a
  one-line change to that method's loss function.

## The takeaway for using ScaNN

You don't need to understand the proofs to use this — that's what the
`anisotropic_quantization_threshold` parameter is for. But now you know what
you're actually turning up or down when you set it: you're telling ScaNN how
aggressively to protect the compression accuracy of vectors that are likely
to matter for the ranking, at the cost of being a little more careless about
the ones that aren't. Leave it at the default (disabled) and you get
ordinary, direction-agnostic compression; sweep it starting around `0.2` (as
in [docs/api_reference.md](api_reference.md) and
[docs/algorithms.md](algorithms.md)) and you're using the actual idea this
paper introduced.
