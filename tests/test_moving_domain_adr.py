from examples.moving_domain_adr_example import main


def test_moving_domain_adr_example_advances_a_real_step():
    result = main(h=0.18, step_count=1)
    assert result["node_count"] > 100
    assert result["relative_l2_error"] < 5.0e-3
    assert result["step_count"] == 1
