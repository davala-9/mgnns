import time

from src.rule_extraction.lattice_search import (
    BFSFrontier,
    PriorityFrontier,
    SinglePathFrontier,
    GreedyBestSuccessorFrontier,
    FirstResultPolicy,
    AllMinimalPolicy,
    search,
)

# BFSFrontier/DFSFrontier are also exercised via tests/rule_extraction/test_rule_optimisation_3.py, which
# imports them through rule_optimisation_3.py's re-export. This file focuses on what's new here:
# PriorityFrontier, SinglePathFrontier's single-path commitment, GreedyBestSuccessorFrontier, and the
# search()/policy machinery.


class FakeSubtree:
    def __init__(self, name, successors=None, sound=False, superset_of=()):
        self.name = name
        self._successors = successors or []
        self.sound = sound
        self._superset_of = set(superset_of)  # names of other FakeSubtrees this one dominates
        self.soundness_calls = 0

    def get_successors(self, base_tree):
        return list(self._successors)

    def is_superset_of(self, other):
        return other.name == self.name or other.name in self._superset_of

    def __eq__(self, other):
        return isinstance(other, FakeSubtree) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"FakeSubtree({self.name!r})"


class FakeBaseTree:
    def __init__(self, initial_subtree):
        self.initial_compact = initial_subtree


def counting_check_soundness(node):
    node.soundness_calls += 1
    return node.sound


class TestSinglePathFrontier:
    def test_commits_to_first_pushed_successor_and_ignores_the_rest(self):
        # Regression test: when a node has several successors, the frontier must keep the FIRST one
        # pushed (mirroring next(node.get_successors(base_tree), None)), not the last. A previous
        # version overwrote self._item on every push, so the search silently followed the last
        # successor instead -- e.g. picking a still-unsound node over an earlier, already-sound one.
        frontier = SinglePathFrontier()
        frontier.push("first")
        frontier.push("second")
        frontier.push("third")
        assert frontier.pop() == "first"

    def test_pop_resets_state_so_the_next_node_is_free_to_push(self):
        frontier = SinglePathFrontier()
        frontier.push("first")
        frontier.pop()
        assert frontier.is_empty()
        frontier.push("next-node")
        assert frontier.pop() == "next-node"

    def test_search_follows_the_first_successor_at_each_step(self):
        target = FakeSubtree("target", sound=True)
        other = FakeSubtree("other", sound=False)
        root = FakeSubtree("root", successors=[target, other], sound=False)
        results = list(search(FakeBaseTree(root), counting_check_soundness, SinglePathFrontier(),
                              FirstResultPolicy()))
        assert results == [target]


class TestGreedyBestSuccessorFrontier:
    def test_keeps_the_highest_scored_item_pushed_since_the_last_pop(self):
        frontier = GreedyBestSuccessorFrontier(score_fn=lambda x: x)
        frontier.push(3)
        frontier.push(7)  # new best
        frontier.push(5)  # lower than the current best -- discarded
        assert frontier.pop() == 7

    def test_pop_resets_state_so_the_next_node_starts_a_fresh_comparison(self):
        frontier = GreedyBestSuccessorFrontier(score_fn=lambda x: x)
        frontier.push(10)
        frontier.pop()
        assert frontier.is_empty()
        frontier.push(1)  # not discarded just because it's lower than the previous round's best
        assert frontier.pop() == 1

    def test_search_always_follows_the_best_successor_never_backtracking_to_the_rest(self):
        # A node with three successors of differing quality: the search must commit to the best one
        # (worst_first_choice, despite being pushed first) and never fall back to a discarded sibling,
        # even though that sibling (better_but_discarded) would itself have led to a sound node.
        discarded_dead_end = FakeSubtree("discarded_dead_end", sound=False)
        better_but_discarded = FakeSubtree("better_but_discarded", successors=[discarded_dead_end], sound=False)
        sound_target = FakeSubtree("sound_target", sound=True)
        best_first_choice = FakeSubtree("best_first_choice", successors=[sound_target], sound=False)
        root = FakeSubtree("root", successors=[better_but_discarded, best_first_choice], sound=False)

        score = {root: 0, better_but_discarded: 5, best_first_choice: 10, sound_target: 0, discarded_dead_end: 0}
        results = list(search(FakeBaseTree(root), counting_check_soundness,
                              GreedyBestSuccessorFrontier(score_fn=score.__getitem__), FirstResultPolicy()))

        assert results == [sound_target]
        assert better_but_discarded.soundness_calls == 0  # never even checked, let alone expanded


class TestPriorityFrontier:
    def test_pops_highest_score_first(self):
        frontier = PriorityFrontier(score_fn=lambda x: x)
        for x in [3, 1, 4, 1, 5]:
            frontier.push(x)
        popped = []
        while not frontier.is_empty():
            popped.append(frontier.pop())
        assert popped == [5, 4, 3, 1, 1]

    def test_ties_broken_fifo(self):
        frontier = PriorityFrontier(score_fn=lambda x: 0)  # every item scores the same
        for x in ["a", "b", "c"]:
            frontier.push(x)
        assert [frontier.pop(), frontier.pop(), frontier.pop()] == ["a", "b", "c"]

    def test_is_empty(self):
        frontier = PriorityFrontier(score_fn=lambda x: x)
        assert frontier.is_empty()
        frontier.push(1)
        assert not frontier.is_empty()


class TestFirstResultPolicy:
    def test_returns_only_first_sound_node_found(self):
        target = FakeSubtree("target", sound=True)
        other = FakeSubtree("other", sound=True)
        root = FakeSubtree("root", successors=[target, other], sound=False)
        results = list(search(FakeBaseTree(root), counting_check_soundness, BFSFrontier(), FirstResultPolicy()))
        assert results == [target]
        assert other.soundness_calls == 0  # search stopped before ever reaching it

    def test_deduplicates_across_a_cycle(self):
        root = FakeSubtree("root", sound=False)
        child = FakeSubtree("child", successors=[root], sound=False)
        root._successors = [child]
        results = list(search(FakeBaseTree(root), counting_check_soundness, BFSFrontier(), FirstResultPolicy()))
        assert results == []
        assert root.soundness_calls == 1
        assert child.soundness_calls == 1

    def test_deadline_in_the_past_yields_nothing(self):
        root = FakeSubtree("root", sound=True)
        results = list(search(FakeBaseTree(root), counting_check_soundness, BFSFrontier(), FirstResultPolicy(),
                              deadline=time.monotonic() - 1))
        assert results == []
        assert root.soundness_calls == 0


class TestAllMinimalPolicy:
    def test_collects_every_inclusion_minimal_sound_node(self):
        a = FakeSubtree("a", sound=True)
        d = FakeSubtree("d", sound=True)
        e = FakeSubtree("e", sound=False)
        c = FakeSubtree("c", successors=[e], sound=True, superset_of={"a"})
        b = FakeSubtree("b", successors=[c, d], sound=False)
        root = FakeSubtree("root", successors=[a, b], sound=False)

        results = list(search(FakeBaseTree(root), counting_check_soundness, BFSFrontier(), AllMinimalPolicy()))

        assert results == [a, d]
        # c is dominated by a (monotonicity: a superset of a sound node is sound too, but redundant)
        # -- it must be pruned WITHOUT spending a soundness check or being expanded further.
        assert c.soundness_calls == 0
        assert e.soundness_calls == 0

    def test_sound_nodes_are_never_expanded_further(self):
        grandchild = FakeSubtree("grandchild", sound=False)
        child = FakeSubtree("child", successors=[grandchild], sound=True)
        root = FakeSubtree("root", successors=[child], sound=False)
        list(search(FakeBaseTree(root), counting_check_soundness, BFSFrontier(), AllMinimalPolicy()))
        assert grandchild.soundness_calls == 0  # child was sound, so its successors are redundant
