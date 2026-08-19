from __future__ import annotations

import networkx as nx

from gap_aware_alignments.petri_net import (
    ARTIFICIAL_END_TRANSITION_LABEL,
    ARTIFICIAL_END_TRANSITION_NAME,
    Marking,
    PetriNet,
    add_artificial_end_transition,
)


class ReachabilityGraph(nx.MultiDiGraph):
    def __init__(self, accepting_petri_net: PetriNet) -> None:
        if not accepting_petri_net.is_workflow_net():
            raise ValueError("The Petri net is not a workflow net")

        net = add_artificial_end_transition(accepting_petri_net)
        super().__init__()
        self.transition_labels = dict(net.transition_labels)
        self.marking_map: dict[Marking, int] = {net.initial_marking: 0, net.final_marking: 1}
        self.initial_state = 0
        self.final_state = 1
        self.add_node(0, marking=net.initial_marking, enabled=set())
        self.add_node(1, marking=net.final_marking, enabled=set())
        open_list = [0]

        def get_arcs_from_enabled_transitions(
            marking: Marking,
            visited_states: set[Marking],
            firing_sequence: tuple[str, ...] | None = None,
        ) -> set[tuple[Marking, Marking, tuple[str, ...], str]]:
            if firing_sequence is None:
                firing_sequence = tuple()
            visited_states.add(marking)
            arcs = set()
            for transition in net.enabled_transitions(marking):
                next_marking = net.execute(transition, marking)
                label = net.transition_labels[transition]
                if label is not None:
                    arcs.add((marking, next_marking, firing_sequence + (transition,), label))
                elif next_marking not in visited_states:
                    arcs.update(
                        get_arcs_from_enabled_transitions(
                            next_marking,
                            visited_states,
                            firing_sequence=firing_sequence + (transition,),
                        )
                    )
            return arcs

        while open_list:
            current_node = open_list.pop()
            current_marking = self.nodes[current_node]["marking"]
            for arc in get_arcs_from_enabled_transitions(current_marking, set()):
                post_marking = arc[1]
                if post_marking not in self.marking_map:
                    self.marking_map[post_marking] = self.number_of_nodes()
                    self.add_node(self.marking_map[post_marking], marking=post_marking, enabled=set())
                    open_list.append(self.marking_map[post_marking])
                if arc[3] is not ARTIFICIAL_END_TRANSITION_LABEL:
                    self.nodes[current_node]["enabled"].add(arc[3])
                self.add_edge(
                    current_node,
                    self.marking_map[post_marking],
                    firing_sequence=arc[2],
                    label=arc[3],
                    cost=0 if arc[2][-1] == ARTIFICIAL_END_TRANSITION_NAME else 1,
                )
        self.best_worst_cost = nx.dijkstra_path_length(
            self, self.initial_state, self.final_state, weight="cost"
        )
        self.all_pairs_shortest_path = dict(nx.all_pairs_shortest_path(self))
