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


def test_scoped_kb_reply_without_evidence_is_refused_even_if_model_invents_an_endpoint():
    from eval.product.agent_eval import isolated_workspace
    from app.core.agent import GenesisAgent
    from app.core.agent_store import AgentStore, AgentConfig
    from app.db.database import GenesisDB
    from unittest.mock import patch
    from types import SimpleNamespace
    import json
    with isolated_workspace():
        db = GenesisDB()
        uid = db.create_user('grounding@example.test','not-used')['id']
        bot = AgentConfig('Grounded','Use only uploaded documents.')
        bot.author = uid
        AgentStore(db=db).create_agent(bot)
        db.add_kb_document(bot.id,uid,'policy.txt','Returns accepted for fourteen days.',['Returns accepted for fourteen days.'])
        agent = GenesisAgent(db=db,agent_override={'agent_id':bot.id,'system_prompt':bot.system_prompt})
        answer = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'tool':None,'message':'Use DELETE /api/v1/background-jobs/{id}'}),tool_calls=None))],usage=None)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_:answer)))
        with patch('app.core.groq_client.provider_plan',return_value=[('replay','fixture',client)]):
            response = agent.process('How do I cancel a background job?','g',user_id=uid)
        assert response.data['grounding']=='insufficient_evidence'
        assert '/api/v1/' not in response.message
