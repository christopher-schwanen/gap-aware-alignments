from __future__ import annotations
from abc import ABC, abstractmethod
from math import log
from typing import TYPE_CHECKING

import networkx as nx

from gap_aware_alignments.utils import ARTIFICIAL_END_TRANSITION_LABEL

if TYPE_CHECKING:
    from pm4py.objects.log.obj import Trace
    from pm4py.util.typing import AlignmentResult

    from gap_aware_alignments.reachability_graph import ReachabilityGraph


class CostModel(ABC):
    """Base class for gap-aware cost models."""

    @abstractmethod
    def __call__(self, gap_length: int) -> float:
        """
        Return the cost of a gap of the given length.
        """
        ...


class AffineGapCostModel(CostModel):
    def __init__(self, gap_opening_cost: float, gap_extension_cost: float) -> None:
        self.gap_opening_cost = gap_opening_cost
        self.gap_extension_cost = gap_extension_cost

    def __call__(self, gap_length: int) -> float:
        if gap_length <= 0:
            return 0
        return self.gap_opening_cost + (gap_length) * self.gap_extension_cost


class LogGapCostModel(CostModel):
    def __init__(self, gap_opening_cost: float, gap_extension_cost: float) -> None:
        self.gap_opening_cost = gap_opening_cost
        self.gap_extension_cost = gap_extension_cost

    def __call__(self, gap_length: int) -> float:
        if gap_length <= 0:
            return 0
        return self.gap_opening_cost + log(gap_length + 1) * self.gap_extension_cost


class PowerGapCostModel(CostModel):
    def __init__(self, gap_opening_cost: float, gap_extension_cost: float, power: float) -> None:
        self.gap_opening_cost = gap_opening_cost
        self.gap_extension_cost = gap_extension_cost
        self.power = power

    def __call__(self, gap_length: int) -> float:
        if gap_length <= 0:
            return 0
        return self.gap_opening_cost + (gap_length ** self.power) * self.gap_extension_cost


class AlignmentStateGraph(nx.MultiDiGraph):
    def __init__(self,
                 reachability_graph: ReachabilityGraph,
                 trace: Trace,
                 affine_gap_cost_model: AffineGapCostModel,
                 log_gap_cost_model: LogGapCostModel,
                 power_gap_cost_model: PowerGapCostModel,
                 ) -> None:
        super().__init__()
        self.trace = trace
        self.best_worst_cost = len(trace) + reachability_graph.best_worst_cost
        self.initial_state = (0, reachability_graph.initial_state)
        self.final_state = (len(trace) + 1, reachability_graph.final_state)
        self.transition_labels = reachability_graph.transition_labels

        # Add initial and final states
        self.add_node(self.initial_state, marking=reachability_graph.nodes[reachability_graph.initial_state]['marking'])
        self.add_node(self.final_state, marking=reachability_graph.nodes[reachability_graph.final_state]['marking'])

        gap_starts: list[tuple[int, int]] = [self.initial_state]
        for idx in range(len(trace)):
            new_gap_starts: list[tuple[int, int]] = []
            for edge in reachability_graph.edges(data=True):
                if edge[2]['label'] == trace[idx].get('concept:name'):
                    self.add_node((idx, edge[0]), marking=reachability_graph.nodes[edge[0]]['marking'])
                    # Add arcs corresponding to minimal gaps
                    for gap_start in gap_starts:
                        if shortest_path := reachability_graph.all_pairs_shortest_path[gap_start[1]].get(edge[0]):
                            log_cost = idx - gap_start[0]
                            log_labels = [trace[i].get('concept:name') for i in range(gap_start[0], idx)]
                            model_cost = len(shortest_path) - 1
                            model_sequences = [reachability_graph.edges[u, v, 0]['firing_sequence']
                                               for u, v in nx.utils.pairwise(shortest_path)]
                            self.add_edge(gap_start, (idx, edge[0]),
                                          firing_sequence=[()] * log_cost + model_sequences,
                                          label=log_labels + [None] * model_cost,
                                          affine_gap_cost=affine_gap_cost_model(log_cost + model_cost),
                                          log_gap_cost=log_gap_cost_model(log_cost + model_cost),
                                          power_gap_cost=power_gap_cost_model(log_cost + model_cost),
                                          classic_cost=log_cost + model_cost,
                                          type='gap')
                    self.add_node((idx + 1, edge[1]), marking=reachability_graph.nodes[edge[1]]['marking'])
                    new_gap_starts.append((idx + 1, edge[1]))
                    # Add arcs corresponding to synchronous moves
                    self.add_edge((idx, edge[0]), (idx + 1, edge[1]),
                                  firing_sequence=edge[2]['firing_sequence'],
                                  label=trace[idx].get('concept:name'),
                                  affine_gap_cost=0,
                                  log_gap_cost=0,
                                  power_gap_cost=0,
                                  classic_cost=0,
                                  type='sync')
            gap_starts.extend(new_gap_starts)

        # Handling of artificial end transitions
        for edge in reachability_graph.edges(data=True):
            if edge[2]['label'] == ARTIFICIAL_END_TRANSITION_LABEL:
                self.add_node((len(trace), edge[0]), marking=reachability_graph.nodes[edge[0]]['marking'])
                # Add arcs corresponding to minimal gaps
                for gap_start in gap_starts:
                    if shortest_path := reachability_graph.all_pairs_shortest_path[gap_start[1]].get(edge[0]):
                        log_cost = len(trace) - gap_start[0]
                        log_labels = [trace[i].get('concept:name') for i in range(gap_start[0], len(trace))]
                        model_cost = len(shortest_path) - 1
                        model_sequences = [reachability_graph.edges[u, v, 0]['firing_sequence']
                                           for u, v in nx.utils.pairwise(shortest_path)]
                        self.add_edge(gap_start, (len(trace), edge[0]),
                                      firing_sequence=[()] * log_cost + model_sequences,
                                      label=log_labels + [None] * model_cost,
                                      affine_gap_cost=affine_gap_cost_model(log_cost + model_cost),
                                      log_gap_cost=log_gap_cost_model(log_cost + model_cost),
                                      power_gap_cost=power_gap_cost_model(log_cost + model_cost),
                                      classic_cost=log_cost + model_cost,
                                      type='gap')
                self.add_node((len(trace) + 1, edge[1]), marking=reachability_graph.nodes[edge[1]]['marking'])
                # Add arcs corresponding to synchronous moves
                self.add_edge((len(trace), edge[0]), (len(trace) + 1, edge[1]),
                              firing_sequence=edge[2]['firing_sequence'],
                              label=trace[idx].get('concept:name'),
                              affine_gap_cost=0,
                              log_gap_cost=0,
                              power_gap_cost=0,
                              classic_cost=0,
                              type='sync')

        print(f"Constructed alignment state graph with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges.")

    def get_optimal_alignment_costs(self) -> tuple[float, float, float, float]:
        return (nx.dijkstra_path_length(self, self.initial_state, self.final_state, weight='classic_cost'),
                nx.dijkstra_path_length(self, self.initial_state, self.final_state, weight='affine_gap_cost'),
                nx.dijkstra_path_length(self, self.initial_state, self.final_state, weight='log_gap_cost'),
                nx.dijkstra_path_length(self, self.initial_state, self.final_state, weight='power_gap_cost'))

    def _path_to_alignment(self, path: list, weight: str, cost: float) -> 'AlignmentResult':
        moves = []
        for u, v in nx.utils.pairwise(path):
            if v == self.final_state:
                continue
            best_edge = min(self[u][v].values(), key=lambda e: e[weight])
            if best_edge['type'] == 'sync':
                log_label = best_edge['label']
                firing_seq = best_edge['firing_sequence']
                model_label = self.transition_labels.get(firing_seq[-1], firing_seq[-1])
                moves.append((log_label, model_label))
            else:  # gap
                for log_label, firing_seq in zip(best_edge['label'], best_edge['firing_sequence']):
                    if log_label is not None:
                        moves.append((log_label, '>>'))
                    else:
                        model_label = self.transition_labels.get(firing_seq[-1], firing_seq[-1])
                        moves.append(('>>',  model_label))
        return {'alignment': moves, 'cost': cost}

    def get_optimal_alignments(self) -> tuple['AlignmentResult', 'AlignmentResult', 'AlignmentResult', 'AlignmentResult']:
        weights = ['classic_cost', 'affine_gap_cost', 'log_gap_cost', 'power_gap_cost']
        costs = self.get_optimal_alignment_costs()
        return tuple(
            self._path_to_alignment(
                nx.dijkstra_path(self, self.initial_state, self.final_state, weight=w),
                w,
                c,
            )
            for w, c in zip(weights, costs)
        )


def align(trace: Trace,
          reachability_graph: ReachabilityGraph,
          gap_opening_cost: float = 1,
          gap_extension_cost: float = 0.5,
          gap_power: float = 0.7,
          ) -> tuple[tuple[float, float, float, float], tuple['AlignmentResult', 'AlignmentResult', 'AlignmentResult', 'AlignmentResult']]:
    affine_gap_cost_model = AffineGapCostModel(gap_opening_cost=gap_opening_cost, gap_extension_cost=gap_extension_cost)
    log_gap_cost_model = LogGapCostModel(gap_opening_cost=gap_opening_cost, gap_extension_cost=gap_extension_cost)
    power_gap_cost_model = PowerGapCostModel(gap_opening_cost=gap_opening_cost,
                                             gap_extension_cost=gap_extension_cost,
                                             power=gap_power)
    graph = AlignmentStateGraph(reachability_graph, trace, affine_gap_cost_model, log_gap_cost_model, power_gap_cost_model)
    return graph.get_optimal_alignment_costs(), graph.get_optimal_alignments()
