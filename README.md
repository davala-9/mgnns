# MGNNs

This repository implements monotonic graph neural networks (MGNNs) over multi-relational graphs, together with
algorithms that extract Datalog rules from a trained model: both rules that explain individual predictions and
(attempts at) a whole program equivalent to the model. This project is licensed under the Apache License 2.0 – see
the LICENSE file for details.

## Installation

The code requires [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric),
plus a few smaller packages:

```bash
pip install -r requirements.txt
```

It requires Python 3.10 or later, and has been tested with Python 3.13, PyTorch 2.12 and PyTorch Geometric 2.8.

All commands below are run from the repository root, as modules (`python -m src.run.<script>`), so that the
`src.` imports resolve. Each script supports `--help`.

## Directory structure

```
.
├── configs        # Experiment configuration files (see "Configuration" below)
├── data           # Dataset folders (see "Data format" below)
├── experiments    # One output folder per experiment run
├── src            # The implementation (see "Code map" below)
└── tests          # Unit tests, mirroring the layout of src
```

## Data format

Each dataset is a single folder with the following files:

```
dataset_name
├── predicates.csv
├── train_graph.tsv
├── train_pos.tsv
├── valid_graph.tsv
├── valid_pos.tsv
├── valid_neg.tsv
├── test_graph.tsv
├── test_pos.tsv
└── test_neg.tsv
```

- `predicates.csv` lists the signature, one predicate per line, as `[predicate name],[arity]` where the arity is 1 or 2.
  The order of the lines matters: it fixes the order of the model's feature positions and edge colours.
- `train_graph.tsv`, `valid_graph.tsv` and `test_graph.tsv` are the input graphs fed into the model for training,
  validation and testing.
- `train_pos.tsv`, `valid_pos.tsv` and `test_pos.tsv` are the positive examples for training, validation and testing.
- `valid_neg.tsv` and `test_neg.tsv` are the negative examples for validation and testing. There are no negative
  examples for training: every fact not in `train_pos.tsv` is treated as negative during training.

Each line of a tsv file is one fact `[subject]\t[relation]\t[object]`. Unary facts use
`http://www.w3.org/1999/02/22-rdf-syntax-ns#type` as the relation, with the unary predicate as the object. Every
predicate used (the relation of a binary fact, or the object of a unary fact) **must** appear in `predicates.csv` with
the matching arity. There is no list of entity names: the system works in the inductive setting, so validation and
test files may mention entities never seen during training.

The scripts in `src/utils/` (`split_dataset.py`, `convert_examples_from_rgcn_format.py`,
`extract_predicates_from_graph_and_examples.py`, `generate_negatives_for_classification.py`) are one-off helpers that
were used to build some of the datasets in `data/` from other formats.

## Configuration

An experiment is described by a YAML file, e.g. `configs/AD1_maxmax.yaml`. Every key is required:

```yaml
data_dir: ./data/node_classification/AD1  # the dataset folder
exp_dir: ./experiments                    # where the experiment's output folder is created
use_dummies: false                        # (iclr22 only) add dummy constants to the training graph to discourage false positives
encoding_scheme: canonical                # canonical | iclr22
agg_function_1: max                       # aggregation in layer 1: max | sum
agg_function_2: max                       # aggregation in layer 2: max | sum
derivation_threshold: 0.000000001         # threshold (theta in the papers), between 0 and 1, above which a fact is derived
non_negative_weights: true                # clamp the weight matrices to be non-negative after each training step (monotonicity)
clamping: 0                               # [CURRENTLY UNSUPPORTED] must be non-negative
```

`encoding_scheme: canonical` uses the canonical encoding directly. `iclr22` uses the encoding from our ICLR 2022
paper [1], which also supports binary target predicates.

## Running an experiment

```bash
python -m src.run.run_experiment configs/AD1_maxmax.yaml
```

This trains a model, evaluates it on the validation and test data, extracts a program from it, and explains its
top-scoring test predictions. `run_all.sh` does this for every file in `configs/`. Options:

- `--load-model [experiment folder]` skips training and loads the model and encoders from a previous experiment. The
  loaded model's parameters are not checked against the current configuration, so make sure they match.
- `--skip-program` skips the (slow) program extraction.

Each run creates a folder in `exp_dir` named after the dataset folder and the start time, containing:

```
experiment_name
├── checkpoints                  # model snapshots saved during training
├── [config file].yaml           # a copy of the configuration used
├── external_encoder.tsv
├── internal_encoder.tsv
├── model.pt
├── valid_metrics.txt
├── test_metrics.txt
├── predicted_triples.tsv
├── predicted_triples_scored.tsv
├── program.txt
└── explanations.txt
```

- `external_encoder.tsv` and `internal_encoder.tsv` are the two encoders (see "Code map" below).
  `checkpoints`, `model.pt` and the encoder files are only written when the model is trained, not when it is loaded.
- `valid_metrics.txt` and `test_metrics.txt` give precision, recall, accuracy and F1 at a range of thresholds, plus
  the area under the precision-recall curve.
- `predicted_triples.tsv` lists the facts the model derives on the test graph, from highest to lowest score.
  `predicted_triples_scored.tsv` is the same with each fact's score (its value in the model's last layer).
- `program.txt` is the extracted program, one rule per head predicate and subtree found (see `full_program.py`).
- `explanations.txt` lists the highest-scoring test predictions (`N_FACTS_TO_EXPLAIN` in `run_experiment.py`),
  each followed by a rule that explains it.

### Rule syntax

Rules are written as `head :- body .`, with atoms `<predicate>[?X]` (unary) or `<predicate>[?X,?Y]` (binary):

```
<A>[?X0] :- <S>[?X1,?X0], <A>[?X1], <R>[?X2,?X0] .
```

`src/datalog/apply_rules.py` can parse such rules and apply them to a set of facts.

### Extracting rules from a saved model

To run only the program extraction on a model from a previous experiment:

```bash
python -m src.run.extract_rules [experiment folder] [threshold] [output folder] [canonical|iclr22] [--predicate P]
```

`--predicate` restricts extraction to rules whose head is `P`.

### The ADNI pipeline

`src/adni/` is a separate extraction method for the ADNI brain-graph datasets (`data/node_classification/AD1`,
`AD2`), which uses the fact that their graphs have a fixed shape. The signature must include the unary predicates
`node`, `node_0` ... `node_{d-1}` and `positive`, and the binary predicates `part_of` and `1.0` ... `{max}.0`. A
candidate rule body is a d x d upper-triangular matrix whose cell (i, j) holds the colour of the edge between
regions i and j (or 0 for no edge). To search for a sound rule for `positive`:

```bash
python -m src.run.extract_adni_rules [experiment folder] [threshold] [output folder]
```

This writes the rule to `[output folder]/program.txt`.

## Code map

The model:

- `src/model/cd_graph.py`: `CDGraph`, a (col,d)-graph: node features plus coloured edges.
- `src/model/gnn_architectures.py`: `GNN`, the two-layer model, with accessors for its matrices (`matrix_A`,
  `matrix_B`), biases, activations and aggregations.
- `src/model/gnn_transformation.py`: applies the full transformation described in the papers: external encoding,
  internal encoding, model, and the two decodings.

Encodings. We use the framework from our paper [2]: an external (non-canonical) encoder maps a dataset to a
(col,d)-dataset, and an internal encoder, which is always the canonical one, maps that to a (col,d)-graph. With
`encoding_scheme: canonical` the external encoder is the identity.

- `src/encodings/canonical.py`: `CanonicalEncoderDecoder`, the internal encoder.
- `src/encodings/noncanonical/`: the external encoders (`IdentityEncoderDecoder`, `ICLREncoderDecoder`). Besides
  encoding and decoding facts, they "unfold" rules over the canonical signature back into rules over the data
  signature.

Rule extraction (`src/rule_extraction/`):

- `tree_shaped_conjunction.py`: `TreeShapedConjunction`, the tree-shaped rule bodies that extraction works with,
  and `CompactSubTree`, a compact representation of one of their subtrees.
- `full_program.py`: `EquivalentProgramExtractor`. For each head predicate it builds the largest relevant rule
  body (`compute_tree_for`), then searches its subtrees for the minimal sound ones.
- `lattice_search.py`: the generic search over subtrees used by the above, with pluggable exploration orders
  (`Frontier`s) and stopping rules (`ResultPolicy`s).
- `typed_constraint.py`: `TypedConstraint`, structural knowledge about an encoding (e.g. which edges can exist
  between which kinds of node), used to prune the search.
- `fact_explanation.py`: `FactExplainer`, which explains one prediction: it builds a rule body grounded in the
  input graph, then shrinks it with Optimisations 1–3 (`rule_optimisation_*.py`).

Everything else: `src/config/` (reads the YAML configuration), `src/run/` (the scripts above, training and metrics),
`src/datalog/` (parsing, writing and applying rules), `src/utils/` (`BitSet` and small helpers).

## Tests

```bash
python -m pytest
```

## References

[1] David Tena Cucala, Bernardo Cuenca Grau, Egor V. Kostylev, Boris Motik. Explainable GNN-Based Models over
Multi-Relational Graphs. ICLR 2022.

[2] TODO
