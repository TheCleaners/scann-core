# Part 1: A first index

Script: [`code/part1_first_index.py`](code/part1_first_index.py)

## Loading the data

```python
from tutorial_data import load_glove

dataset, queries, true_neighbors = load_glove()
```

```
dataset (1183514, 100) float32, queries (10000, 100)
```

`dataset` is the collection to search: one row per vector, a float32 numpy
array. ScaNN takes its data in exactly this form. `queries` are the vectors
we'll search *for*, and `true_neighbors[i]` lists the rows of `dataset`
closest to `queries[i]`, closest first.

## Building an index

```python
import scann

searcher = (scann.scann_ops_pybind.builder(dataset, 10, "dot_product")
            .score_brute_force()
            .build())
```

`builder(dataset, num_neighbors, distance_measure)` starts a configuration:

* `10` is how many neighbours a search returns by default.
* `"dot_product"` sets how closeness is measured. A higher dot product means
  closer. The alternative is `"squared_l2"`, squared Euclidean distance,
  where lower means closer.

The builder methods in between describe *how* to search.
`.score_brute_force()` is the simplest option: compare the query against
every single vector, exactly. `.build()` then creates the searcher. For
brute force there is nothing to train, so it takes 0.04 s.

### Dot product, cosine, and why we normalized

GloVe's ground truth uses *cosine similarity*. That measures the angle
between vectors and ignores their length. For unit-length vectors, cosine
similarity *is* the dot product. `load_glove()` normalizes every vector, so
`"dot_product"` finds exactly the neighbours the ground truth lists.

If you forget to normalize, dot-product search favours long vectors: a long
vector pointing roughly the right way outscores a short one pointing exactly
the right way. Sometimes that is what you want, for example with
recommendation embeddings where length encodes popularity. Often it isn't.
**Know which similarity your embeddings were trained for.**

## Searching

```python
neighbors, distances = searcher.search(queries[0])
```

```
neighbors: [  97478  262700  846101  671078  232287  727732  544474 1133489  723915
  660281]
distances: [0.5696 0.5676 0.5669 0.5666 0.5624 0.5602 0.5504 0.5499 0.548  0.5478]
```

`neighbors` are row numbers in `dataset`, best first. `distances` are the
actual scores; for dot product that's the dot product itself, so higher is
better and the list is descending. (Internally ScaNN negates dot products so
that smaller is always better; the API converts back.)

This is exact search, so we can check it by hand:

```python
scores = dataset @ queries[0]
np.argsort(-scores)[:10]
```

```
by hand:   [  97478  262700  846101  671078  232287  727732  544474 1133489  723915
  660281]
           [0.5696 0.5676 0.5669 0.5666 0.5624 0.5602 0.5504 0.5499 0.548  0.5478]
```

Identical. The default of 10 neighbours can be overridden per query:

```python
neighbors, _ = searcher.search(queries[0], final_num_neighbors=3)
```

```
top 3: [ 97478 262700 846101]
```

## Many queries at once

```python
neighbors, distances = searcher.search_batched(queries)
```

```
batched: (10000, 10) in 14.56 s
same as the ground truth for query 0: True
```

`search_batched` takes a 2-D array of queries and returns one row of results
per query.

14.6 seconds for 10,000 queries works out to about 1.5 ms each, on a single
thread. That's 10,000 × 1.18 million × 100 ≈ 1.2 trillion multiply-adds,
so it isn't slow for what it does. But it scales with the dataset: ten times the data means ten times
the wait. The rest of the tutorial is about not doing all that work.

**Next:** before speeding anything up, we need a way to measure what we
lose. [Part 2: Measuring](02-measuring.md).
