from experiments.e54_acceptance import decision


def fixture(passed=True):
    return {'research_guard_passed':passed,'folds':[{'fold':i,'rates':{'clean':{'passed':True},'q75':{'passed':True}}} for i in range(3)]}


def replays():
    return [{'fold':i,'passed':True} for i in range(3)]


def test_relative_gain_cannot_hide_absolute_gate_failure():
    arm=fixture();arm['folds'][1]['rates']['q75']['passed']=False
    assert not decision(arm,replays())['eligible_for_next_research_stage']


def test_requires_all_three_artifact_replays():
    assert not decision(fixture(),replays()[:2])['eligible_for_next_research_stage']
    failed=replays();failed[2]['passed']=False
    assert not decision(fixture(),failed)['eligible_for_next_research_stage']
    assert not decision(fixture(),[replays()[0]]*3)['eligible_for_next_research_stage']
    assert decision(fixture(),replays())['eligible_for_next_research_stage']


def test_absolute_success_cannot_hide_ai_preservation_failure():
    assert not decision(fixture(False),replays())['eligible_for_next_research_stage']
