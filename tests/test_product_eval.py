def test_replayed_provider_real_product_workflow():
    from eval.product.agent_eval import offline_workflow
    report = offline_workflow()
    assert report["summary"]["cases"] == 12
    assert report["summary"]["success_rate"] == 1, report
    assert report["llm_quality_measured"] is False


def test_routing_eval_accepts_equivalent_days_and_weeks_but_preserves_other_arguments():
    from eval.product.agent_eval import routing_args_match
    expected = {'target_column': 'revenue', 'horizon': 2, 'horizon_unit': 'week'}
    assert routing_args_match({'target_column': 'revenue', 'horizon': 14, 'horizon_unit': 'day'}, expected)
    assert not routing_args_match({'target_column': 'cost', 'horizon': 14, 'horizon_unit': 'day'}, expected)
    assert not routing_args_match({'target_column': 'revenue', 'horizon': 2, 'horizon_unit': 'day'}, expected)
