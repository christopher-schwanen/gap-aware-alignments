from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING

from pm4py.objects.petri_net.utils.petri_utils import add_place, add_transition, add_arc_from_to, place_set_as_marking

if TYPE_CHECKING:
    from pm4py import PetriNet, Marking


ARTIFICIAL_END_TRANSITION_NAME = "END"
ARTIFICIAL_END_TRANSITION_LABEL = "END"
ARTIFICIAL_END_PLACE_NAME = "FINAL"


def add_artificial_end_transition(accepting_petri_net: tuple[PetriNet, Marking, Marking]
                                  ) -> tuple[PetriNet, Marking, Marking]:
    accepting_petri_net = deepcopy(accepting_petri_net)
    artificial_end_transition = add_transition(accepting_petri_net[0],
                                               name=ARTIFICIAL_END_TRANSITION_NAME,
                                               label=ARTIFICIAL_END_TRANSITION_LABEL)
    for place in accepting_petri_net[2]:
        add_arc_from_to(place, artificial_end_transition, accepting_petri_net[0])
    artificial_final_place = add_place(accepting_petri_net[0], name=ARTIFICIAL_END_PLACE_NAME)
    add_arc_from_to(artificial_end_transition, artificial_final_place, accepting_petri_net[0])
    return accepting_petri_net[0], accepting_petri_net[1], place_set_as_marking([artificial_final_place])
