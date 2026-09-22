"""Part 7 setup: export GloVe as .npy files, and save an index built in
Python, for the C++ and Rust programs to use."""

import os

import numpy as np
import scann

from tutorial_data import data_dir, load_glove

dataset, queries, true_neighbors = load_glove()
out = data_dir()
np.save(os.path.join(out, "glove_train.npy"), dataset)
np.save(os.path.join(out, "glove_test.npy"), queries)
np.save(os.path.join(out, "glove_neighbors.npy"),
        true_neighbors.astype(np.uint32))

searcher = (scann.scann_ops_pybind.builder(dataset, 10, "dot_product")
            .tree(num_leaves=2000, num_leaves_to_search=100,
                  training_sample_size=250000)
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(100)
            .build())
index_dir = os.path.join(out, "glove-index")
os.makedirs(index_dir, exist_ok=True)
searcher.serialize(index_dir, relative_path=True)
searcher.set_num_threads(os.cpu_count())
found, _ = searcher.search_batched_parallel(queries)
hits = sum(np.intersect1d(f, t[:10]).size for f, t in zip(found, true_neighbors))
print(f"wrote .npy files and {index_dir}; "
      f"recall@10 from Python: {hits / found.size:.4f}")
