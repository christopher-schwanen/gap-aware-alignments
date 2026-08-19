from __future__ import annotations

from copy import deepcopy
from typing import Iterable

# A marking is a sorted tuple of (place_id, count) with count > 0.
Marking = tuple[tuple[str, int], ...]


class PetriNet:
    """Lightweight accepting Petri net wrapper around the Rust4PM dict.

    Attributes:
        places: set of place ids.
        transition_labels: mapping ``transition_id -> label`` (``None`` = silent).
        initial_marking / final_marking: :data:`Marking` values.
        preset / postset: for every transition id, the multiset of input / output
            places as ``{place_id: weight}``.
    """

    def __init__(self, net: dict) -> None:
        self.places: set[str] = set(net["places"].keys())
        self.transition_labels: dict[str, str | None] = {
            tid: t.get("label") for tid, t in net["transitions"].items()
        }

        self.preset: dict[str, dict[str, int]] = {tid: {} for tid in self.transition_labels}
        self.postset: dict[str, dict[str, int]] = {tid: {} for tid in self.transition_labels}
        for arc in net["arcs"]:
            source, target = arc["from_to"]["nodes"]
            weight = arc.get("weight", 1)
            if arc["from_to"]["type"] == "PlaceTransition":
                self.preset[target][source] = self.preset[target].get(source, 0) + weight
            else:  # TransitionPlace
                self.postset[source][target] = self.postset[source].get(target, 0) + weight

        self.initial_marking: Marking = _as_marking(net["initial_marking"])
        final_markings = net["final_markings"]
        if len(final_markings) != 1:
            raise ValueError(
                f"Expected exactly one final marking, got {len(final_markings)}."
            )
        self.final_marking: Marking = _as_marking(final_markings[0])

    # -- Petri net semantics -------------------------------------------------

    def enabled_transitions(self, marking: Marking) -> list[str]:
        """Return the ids of all transitions enabled in ``marking``."""
        tokens = dict(marking)
        enabled = []
        for tid, pre in self.preset.items():
            if all(tokens.get(place, 0) >= weight for place, weight in pre.items()):
                enabled.append(tid)
        return enabled

    def execute(self, transition: str, marking: Marking) -> Marking:
        """Fire ``transition`` in ``marking`` and return the resulting marking."""
        tokens = dict(marking)
        for place, weight in self.preset[transition].items():
            tokens[place] = tokens.get(place, 0) - weight
        for place, weight in self.postset[transition].items():
            tokens[place] = tokens.get(place, 0) + weight
        return _normalize(tokens)

    # -- Structural checks ---------------------------------------------------

    def is_workflow_net(self) -> bool:
        """Check the essential WF-net property: a unique source and sink place.

        A workflow net has exactly one place without incoming arcs (source) and
        exactly one place without outgoing arcs (sink).  The initial marking must
        mark the source with a single token and the final marking the sink with a
        single token.  This is the property the reachability-graph construction
        relies on.
        """
        place_has_input = {p: False for p in self.places}
        place_has_output = {p: False for p in self.places}
        for pre in self.preset.values():
            for place in pre:
                place_has_output[place] = True
        for post in self.postset.values():
            for place in post:
                place_has_input[place] = True

        sources = [p for p in self.places if not place_has_input[p]]
        sinks = [p for p in self.places if not place_has_output[p]]
        if len(sources) != 1 or len(sinks) != 1:
            return False
        return (
            self.initial_marking == ((sources[0], 1),)
            and self.final_marking == ((sinks[0], 1),)
        )


ARTIFICIAL_END_TRANSITION_NAME = "END"
ARTIFICIAL_END_TRANSITION_LABEL = "END"
ARTIFICIAL_END_PLACE_NAME = "FINAL"


def add_artificial_end_transition(net: PetriNet) -> PetriNet:
    """Return a copy of ``net`` with an artificial end transition appended.

    The end transition consumes the tokens of the original final marking and
    produces a single token in a fresh ``FINAL`` place, which becomes the new
    final marking.  This mirrors ``utils.add_artificial_end_transition`` of the
    original pm4py-based implementation and gives every accepting run a common,
    explicitly labeled terminating move.
    """
    net = deepcopy(net)
    net.places.add(ARTIFICIAL_END_PLACE_NAME)
    net.transition_labels[ARTIFICIAL_END_TRANSITION_NAME] = ARTIFICIAL_END_TRANSITION_LABEL
    net.preset[ARTIFICIAL_END_TRANSITION_NAME] = {place: count for place, count in net.final_marking}
    net.postset[ARTIFICIAL_END_TRANSITION_NAME] = {ARTIFICIAL_END_PLACE_NAME: 1}
    net.final_marking = ((ARTIFICIAL_END_PLACE_NAME, 1),)
    return net


def _as_marking(marking: dict) -> Marking:
    return _normalize({place: int(count) for place, count in marking.items()})


def _normalize(tokens: dict[str, int]) -> Marking:
    return tuple(sorted((place, count) for place, count in tokens.items() if count > 0))


def import_petri_net(pnml_path: str) -> PetriNet:
    """Import an accepting Petri net from a PNML file using Rust4PM."""
    from r4pm import petri_net as r4pm_petri_net

    return PetriNet(r4pm_petri_net.import_pnml(str(pnml_path)))
