from __future__ import annotations
from typing import TYPE_CHECKING

import networkx as nx
from pm4py.algo.analysis.workflow_net.algorithm import apply as is_workflow_net
from pm4py.objects.petri_net.semantics import enabled_transitions, execute

from gap_aware_alignments.utils import (add_artificial_end_transition, ARTIFICIAL_END_TRANSITION_NAME,
                                        ARTIFICIAL_END_TRANSITION_LABEL)

if TYPE_CHECKING:
    from pm4py import PetriNet, Marking


class ReachabilityGraph(nx.MultiDiGraph):
    def __init__(self, accepting_petri_net: tuple[PetriNet, Marking, Marking]) -> None:
        if not is_workflow_net(accepting_petri_net[0]):
            raise ValueError("The Petri net is not a workflow net")

        accepting_petri_net = add_artificial_end_transition(accepting_petri_net)
        super().__init__()
        self.transition_labels = {transition.name: transition.label
                                  for transition in accepting_petri_net[0].transitions}
        self.marking_map = {accepting_petri_net[1]: 0, accepting_petri_net[2]: 1}
        self.initial_state = 0
        self.final_state = 1
        self.add_node(0, marking=accepting_petri_net[1], enabled=set())
        self.add_node(1, marking=accepting_petri_net[2], enabled=set())
        open_list = [0]

        def get_arcs_from_enabled_transitions(petri_net: PetriNet,
                                              marking: Marking,
                                              visited_states: set[Marking],
                                              firing_sequence: tuple[str, ...] | None = None,
                                              ) -> set[tuple[Marking, Marking, tuple[str, ...], str]]:
            if firing_sequence is None:
                firing_sequence = tuple()
            visited_states.add(marking)
            arcs = set()
            for transition in enabled_transitions(petri_net, marking):
                next_marking = execute(transition, petri_net, marking)
                if transition.label is not None:
                    arcs.add((marking, next_marking, firing_sequence + (transition.name,), transition.label))
                elif next_marking not in visited_states:
                    arcs.update(get_arcs_from_enabled_transitions(petri_net, next_marking, visited_states,
                                                                  firing_sequence=firing_sequence + (transition.name,)))
            return arcs

        while open_list:
            current_node = open_list.pop()
            current_marking = self.nodes[current_node]['marking']
            for arc in get_arcs_from_enabled_transitions(accepting_petri_net[0], current_marking, set()):
                post_marking = arc[1]
                if post_marking not in self.marking_map:
                    self.marking_map[post_marking] = self.number_of_nodes()
                    self.add_node(self.marking_map[post_marking], marking=post_marking, enabled=set())
                    open_list.append(self.marking_map[post_marking])
                if arc[3] is not ARTIFICIAL_END_TRANSITION_LABEL:
                    self.nodes[current_node]['enabled'].add(arc[3])
                self.add_edge(current_node, self.marking_map[post_marking], firing_sequence=arc[2], label=arc[3],
                              cost=0 if arc[2][-1] == ARTIFICIAL_END_TRANSITION_NAME else 1)
        self.best_worst_cost = nx.dijkstra_path_length(self, self.initial_state, self.final_state, weight='cost')
        self.all_pairs_shortest_path = dict(nx.all_pairs_shortest_path(self))
