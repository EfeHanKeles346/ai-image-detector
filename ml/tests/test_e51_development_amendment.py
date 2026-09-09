from experiments.e51_development_amendment import real_clusters


def test_internal_scene_grouping_keeps_all_members_and_transitive_links():
    rows = [{'parent_id':str(i)} for i in range(4)]
    matches = [{'train_label':0,'cal_label':0,'train_parent':str(a),'cal_parent':str(b)} for a,b in [(1,2),(0,1)]]
    groups = real_clusters(rows,matches)
    assert groups=={'0':'0','1':'0','2':'0','3':'3'}
