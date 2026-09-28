"""AnsatzSpec: variational parameters, offsets and fixed angles, and their JSON form."""

import json

import numpy as np
import pytest

from ansatz_search.circuit.ansatz import AnsatzBlock, AnsatzSpec, ParamRef


def test_an_offset_is_part_of_the_angle_not_a_parameter():
    spec = AnsatzSpec(1, [AnsatzBlock("rz", (0,), (ParamRef(0, np.pi),)), AnsatzBlock("rx", (0,), (ParamRef(0),))])
    assert spec.num_params == 1 and spec.blocks[0].params[0] != spec.blocks[1].params[0]


def test_fixed_angles_are_not_parameters():
    spec = AnsatzSpec(2, [AnsatzBlock("rx", (0,), (np.pi / 2,)), AnsatzBlock("crz", (0, 1), (ParamRef(0),)),
                          AnsatzBlock("rzz", (0, 1), (np.float64(0.3),))])
    assert spec.num_params == 1
    assert spec.blocks[0].params == (np.pi / 2,) and spec.blocks[0].param_refs == ()
    assert type(spec.blocks[2].params[0]) is float


def test_an_int_angle_is_ambiguous():
    with pytest.raises(TypeError, match="ParamRef or a float"):
        AnsatzBlock("rx", (0,), (1,))


def test_json_round_trip_keeps_indices_and_fixed_angles_apart():
    spec = AnsatzSpec(2, [AnsatzBlock("rx", (0,), (1.0,)), AnsatzBlock("ry", (1,), (ParamRef(1),)),
                          AnsatzBlock("cx", (0, 1)), AnsatzBlock("rz", (0,), (ParamRef(0, np.pi),))])
    data = json.loads(json.dumps(spec.to_dict()))
    assert [block[2] for block in data["blocks"]] == [[1.0], [1], [], [[0, np.pi]]]
    assert AnsatzSpec.from_dict(data) == spec
