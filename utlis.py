# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import threading
import warnings
import numpy as np
import pandas as pd
import random as rn
import numpy.random as np_rn
from collections import namedtuple

from tensorflow.keras.utils import Sequence

FEATURE_ATTR_NAME = "feature"
TARGET_ATTR_NAME = "target"
TYPE_ATTR_NAME = "label"
UNKNOWN_TARGET_ATTRIBUTE = "-1"
NODE_TYPE_DEFAULT = "default"
EDGE_TYPE_DEFAULT = "default"

SOURCE = "source"
TARGET = "target"
WEIGHT = "weight"

NEO4J_ID_PROPERTY = "ID"
NEO4J_FEATURES_PROPERTY = "features"

RandomState = namedtuple("RandomState", "random, numpy")


def _global_state():
    return RandomState(rn, np_rn)

def _seeded_state(s):
    return RandomState(rn.Random(s), np_rn.RandomState(s))

_rs = _global_state()

def random_state(seed):
    """
    Create a RandomState using the provided seed. If seed is None, return the global RandomState.

    Args:
        seed (int, optional): random seed

    Returns:
        RandomState object
    """
    if seed is None:
        return _rs
    else:
        return _seeded_state(seed)

class Generator(abc.ABC):
    """
    A generator supports creating sequences for input into graph machine learning algorithms via the `flow` method.
    """

    @abc.abstractmethod
    def num_batch_dims(self):
        """
        Returns the number of batch dimensions in returned tensors (_not_ the batch size itself).

        For instance, for full batch methods like GCN, the feature has shape ``1 × number of nodes ×
        feature size``, where the 1 is a "dummy" batch dimension and ``number of nodes`` is the real
        batch size (every node in the graph).
        """
        ...

    @abc.abstractmethod
    def flow(self, *args, **kwargs):
        """
        Create a Keras Sequence or similar input, appropriate for a graph machine learning model.
        """
        ...

    def default_corrupt_input_index_groups(self):
        """
        Optionally returns the indices of input tensors that can be shuffled for
        :class:`.CorruptedGenerator` to use in :class:`.DeepGraphInfomax`.

        If this isn't overridden, this method returns None, indicating that the generator doesn't
        have a default or "canonical" set of indices that can be corrupted for Deep Graph Infomax.
        """
        return None

class KGTripleGenerator(Generator):
    """
    A data generator for working with triple-based knowledge graph models, like ComplEx.

    This requires a StellarGraph that contains all nodes/entities and every edge/relation type that
    will be trained or predicted upon. The graph does not need to contain the edges/triples that are
    used for training or prediction.

    .. seealso::

       Models using this generator: :class:`.ComplEx`, :class:`.DistMult`, :class:`.RotatE`, :class:`.RotE`, :class:`.RotH`.

       Example using this generator (see individual models for more): `link prediction with ComplEx <https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/complex-link-prediction.html>`__.

    Args:
        G (StellarGraph): the graph containing all nodes, and all edge types.

        batch_size (int): the size of the batches to generate
    """

    def __init__(self, G, batch_size):
        self.G = G

        if not isinstance(batch_size, int):
            raise TypeError(
                f"batch_size: expected int, found {type(batch_size).__name__}"
            )

        self.batch_size = batch_size

    def num_batch_dims(self):
        return 1

    def flow(
        self,
        edges,
        negative_samples=None,
        sample_strategy="uniform",
        shuffle=False,
        seed=None,
    ):
        """
        Create a Keras Sequence yielding the edges/triples in ``edges``, potentially with some negative
        edges.

        The negative edges are sampled using the "local closed world assumption", where a
        source/subject or a target/object is randomly mutated.

        Args:
            edges: the edges/triples to feed into a knowledge graph model.
            negative_samples (int, optional): the number of negative samples to generate for each positive edge.

            sample_strategy (str, optional): the sampling strategy to use for negative sampling, if ``negative_samples`` is not None. Supported values:

                ``uniform``

                  Uniform sampling, where a negative edge is created from a positive edge in
                  ``edges`` by replacing the source or destination entity with a uniformly sampled
                  random entity in the graph (without verifying if the edge exists in the graph: for
                  sparse graphs, this is unlikely). Each element in a batch is labelled as 1
                  (positive) or 0 (negative). An appropriate loss function is
                  :class:`tensorflow.keras.losses.BinaryCrossentropy` (probably with
                  ``from_logits=True``).

                ``self-adversarial``

                  Self-adversarial sampling from [1], where each edge is sampled in the same manner
                  as ``uniform`` sampling. Each element in a batch is labelled as 1 (positive) or an
                  integer in ``[0, -batch_size)`` (negative). An appropriate loss function is
                  :class:`stellargraph.losses.SelfAdversarialNegativeSampling`.

                  [1] Z. Sun, Z.-H. Deng, J.-Y. Nie, and J. Tang, “RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space,” `arXiv:1902.10197 <http://arxiv.org/abs/1902.10197>`_, Feb. 2019.

        Returns:
            A Keras sequence that can be passed to the ``fit`` and ``predict`` method of knowledge-graph models.
        """
        if isinstance(edges, pd.DataFrame):
            sources = edges[SOURCE]
            rels = edges[TYPE_ATTR_NAME]
            targets = edges[TARGET]
        else:
            raise TypeError(
                f"edges: expected pandas.DataFrame; found {type(edges).__name__}"
            )

        if negative_samples is not None:
            require_integer_in_range(negative_samples, "negative_samples", min_val=0)

        supported_strategies = ["uniform", "self-adversarial"]
        if sample_strategy not in supported_strategies:
            raise ValueError(
                f"sample_strategy: expected one of {comma_sep(supported_strategies)}, found {sample_strategy!r}"
            )

        source_ilocs = self.G.node_ids_to_ilocs(sources)
        rel_ilocs = self.G.edge_type_names_to_ilocs(rels)
        target_ilocs = self.G.node_ids_to_ilocs(targets)

        return KGTripleSequence(
            max_node_iloc=self.G.number_of_nodes(),
            source_ilocs=source_ilocs,
            rel_ilocs=rel_ilocs,
            target_ilocs=target_ilocs,
            batch_size=self.batch_size,
            shuffle=shuffle,
            negative_samples=negative_samples,
            sample_strategy=sample_strategy,
            seed=seed,
        )

class KGTripleSequence(Sequence):
    def __init__(
        self,
        *,
        max_node_iloc,
        source_ilocs,
        rel_ilocs,
        target_ilocs,
        batch_size,
        shuffle,
        negative_samples,
        sample_strategy,
        seed,
    ):
        self.max_node_iloc = max_node_iloc

        num_edges = len(source_ilocs)
        self.indices = np.arange(num_edges, dtype=np.min_scalar_type(num_edges))

        self.source_ilocs = np.asarray(source_ilocs)
        self.rel_ilocs = np.asarray(rel_ilocs)
        self.target_ilocs = np.asarray(target_ilocs)

        self.negative_samples = negative_samples
        self.sample_strategy = sample_strategy

        self.batch_size = batch_size
        self.seed = seed

        self.shuffle = shuffle

        _, self._global_rs = random_state(seed)
        self._batch_sampler = SeededPerBatch(
            np.random.RandomState, self._global_rs.randint(2 ** 32, dtype=np.uint32)
        )

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_num):
        start = self.batch_size * batch_num
        end = start + self.batch_size

        indices = self.indices[start:end]

        s_iloc = self.source_ilocs[indices]
        r_iloc = self.rel_ilocs[indices]
        o_iloc = self.target_ilocs[indices]
        positive_count = len(s_iloc)
        targets = None

        if self.negative_samples is not None:
            s_iloc = np.tile(s_iloc, 1 + self.negative_samples)
            r_iloc = np.tile(r_iloc, 1 + self.negative_samples)
            o_iloc = np.tile(o_iloc, 1 + self.negative_samples)

            negative_count = self.negative_samples * positive_count
            assert len(s_iloc) == positive_count + negative_count

            rng = self._batch_sampler[batch_num]

            # FIXME (#882): this sampling may be able to be optimised to a slice-write
            change_source = rng.random(size=negative_count) < 0.5
            source_changes = change_source.sum()

            new_nodes = rng.randint(self.max_node_iloc, size=negative_count)

            s_iloc[positive_count:][change_source] = new_nodes[:source_changes]
            o_iloc[positive_count:][~change_source] = new_nodes[source_changes:]

            if self.sample_strategy == "uniform":
                targets = np.repeat(
                    np.array([1, 0], dtype=np.float32), [positive_count, negative_count]
                )
            elif self.sample_strategy == "self-adversarial":
                # the negative samples are labelled with an arbitrary within-batch integer <= 0, based on
                # which positive edge they came from.
                targets = np.tile(
                    np.arange(0, -positive_count, -1), 1 + self.negative_samples
                )
                # the positive examples are labelled with 1
                targets[:positive_count] = 1
            else:
                raise ValueError(f"unknown sample_strategy: {self.sample_strategy!r}")

            assert len(targets) == len(s_iloc)

        assert len(s_iloc) == len(r_iloc) == len(o_iloc)

        if targets is None:
            return ((s_iloc, r_iloc, o_iloc),)

        return (s_iloc, r_iloc, o_iloc), targets

    def on_epoch_end(self):
        if self.shuffle:
            self._global_rs.shuffle(self.indices)

def comma_sep(values, limit=20, stringify=repr):
    """
    Print up to ``limit`` values, comma separated.

    Args:
        values (list): the values to print
        limit (optional, int): the maximum number of values to print (None for no limit)
        stringify (callable): a function to use to convert values to strings
    """
    return separated(values, limit=limit, stringify=stringify, sep=", ")

def separated(values, *, limit, stringify, sep):
    """
    Print up to ``limit`` values with a separator.

    Args:
        values (list): the values to print
        limit (optional, int): the maximum number of values to print (None for no limit)
        stringify (callable): a function to use to convert values to strings
        sep (str): the separator to use between elements (and the "... (NNN more)" continuation)
    """
    count = len(values)
    if limit is not None and count > limit:
        values = values[:limit]
        continuation = f"{sep}... ({count - limit} more)" if count > limit else ""
    else:
        continuation = ""

    rendered = sep.join(stringify(x) for x in values)
    return rendered + continuation

class SeededPerBatch:
    """
    Internal utility class for managing a random state per batch number in a multi-threaded
    environment.

    """

    def __init__(self, create_with_seed, seed):
        self._create_with_seed = create_with_seed
        self._walkers = []
        self._lock = threading.Lock()
        self._rs, _ = random_state(seed)

    def __getitem__(self, batch_num):
        self._lock.acquire()
        try:
            return self._walkers[batch_num]
        except IndexError:
            # always create a new seeded sampler in ascending order of batch number
            # this ensures seeds are deterministic even when batches are run in parallel
            self._walkers.extend(
                self._create_with_seed(self._rs.randrange(2 ** 32))
                for _ in range(len(self._walkers), batch_num + 1)
            )
            return self._walkers[batch_num]
        finally:
            self._lock.release()

def require_integer_in_range(x, variable_name, min_val=-np.inf, max_val=np.inf):
    """
    A function to verify that a variable is an integer in a specified closed range.
    Args:
        x: the variable to check
        variable_name (str): the name of the variable to print out in error messages
        min_val: the minimum value that `x` can attain
        min_val: the maximum value that `x` can attain
    """

    if not isinstance(x, int):
        raise TypeError(f"{variable_name}: expected int, found {type(x).__name__}")

    if x < min_val or x > max_val:

        if min_val == -np.inf:
            region = f"<= {max_val}"
        elif max_val == np.inf:
            region = f">= {min_val}"
        else:
            region = f"in the range [{min_val}, {max_val}]"

        raise ValueError(f"{variable_name}: expected integer {region}, found {x}")
