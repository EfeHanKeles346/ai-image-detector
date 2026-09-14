import copy
import numpy as np
import pytest
from PIL import Image
from experiments.e130_patch_drift_audit import population, encoded_maps


def fixtures():
    rows=[{'index':i,'role':'TRAIN_RESEARCH_PILOT','seed':119000+i,'rendering':'center_square_resized_512'} for i in range(16)]
    cases=[{'index':i,'seed':119000+i,'branch':'fp16_sdpa','passed':True,
            'checks':dict.fromkeys(['background_exact','masked_pixels_changed','nonblank_output','safety_check_negative'],True)} for i in range(16)]
    return {'rows':rows},{'state':'E129_context_generation_complete','rows':cases}


def test_failed_generation_remains_in_complete_diagnostic_population():
    prepared,report=fixtures()
    report['rows'][7]['checks']['safety_check_negative']=False
    report['rows'][7]['passed']=False
    rows,cases=population(prepared,report)
    assert len(rows)==len(cases)==16
    assert sum(r['passed'] for r in cases)==15


@pytest.mark.parametrize('kind',['subset','reordered','role','seed','claims_pass','missing_check','integer_check'])
def test_bad_population_fails_closed(kind):
    prepared,report=fixtures()
    if kind=='subset': report['rows'].pop()
    if kind=='reordered': report['rows'].reverse()
    if kind=='role': prepared['rows'][0]['role']='FINAL'
    if kind=='seed': report['rows'][0]['seed']=0
    if kind=='claims_pass': report['rows'][0]['checks']['background_exact']=False
    if kind=='missing_check': report['rows'][0]['checks'].pop('safety_check_negative')
    if kind=='integer_check': report['rows'][0]['checks']['safety_check_negative']=1
    with pytest.raises(ValueError): population(prepared,report)


def test_encoder_uses_whole_image_triplet_and_spatial_tokens_without_mask():
    import torch
    class Model:
        def forward_intermediates(self,value,**kwargs):
            assert value.shape==(3,3,448,448)
            assert kwargs=={'indices':[10],'norm':True,'return_prefix_tokens':False,'intermediates_only':True}
            torch.testing.assert_close(value[0],value[2])
            return [torch.ones((3,384,32,32))]
    grid,tokens,error,pixel=encoded_maps(Model(),torch,Image.new('RGB',(512,512),(100,120,140)),
                                  torch.device('cpu'),torch.zeros((1,3,1,1)),torch.ones((1,3,1,1)))
    assert grid.shape==(32,32) and tokens.shape==(32,32,384)
    assert error==0
    np.testing.assert_allclose(grid,0,atol=1e-6)
    np.testing.assert_allclose(pixel,0,atol=1e-6)
