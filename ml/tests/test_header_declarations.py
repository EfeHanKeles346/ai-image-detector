from pixelproof.header_declarations import declaration


def test_explicit_tag_stays_an_unverified_declaration():
    result=declaration({'exif_user_comment':{'text':'Made with OpenAI DALL-E'}})
    assert result['value']=='OpenAI DALL-E' and not result['verified']
    assert result['generator_version'] is None


def test_model_parser_ignores_prompt_mentions_and_rejects_ambiguous_settings():
    assert declaration({'parameters':{'text':'A photo of Model: someone'}}) is None
    assert declaration({'parameters':{'text':'Steps: 30, Model: one, Model: two'}}) is None
    result=declaration({'parameters':{'text':'A scene\nSteps: 30, Seed: 2, Model: declared_name, Size: 512x512'}})
    assert result['value']=='declared_name' and not result['verified']


def test_camera_editor_or_arbitrary_comment_never_becomes_generator_identity():
    for fields in [{'exif_camera_model':{'text':'DALL-E'}}, {'software':{'text':'Matplotlib'}},
                   {'exif_user_comment':{'text':'Model: DALL-E'}}, {'parameters':{'text':None}}]:
        assert declaration(fields) is None
