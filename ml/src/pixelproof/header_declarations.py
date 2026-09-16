"""Interpret selected embedded declarations without certifying image provenance."""
import re


def declaration(fields):
    comment = fields.get('exif_user_comment', {}).get('text')
    if comment == 'Made with OpenAI DALL-E':
        return dict(kind='producer_declaration', value='OpenAI DALL-E',
                    field='exif_user_comment', generator_version=None, verified=False)
    parameters = fields.get('parameters', {}).get('text')
    if isinstance(parameters, str):
        settings = [line for line in parameters.splitlines() if line.startswith('Steps: ')]
        models = [name.strip() for line in settings for name in re.findall(r'(?:^|,\s*)Model:\s*([^,\n]+)', line)]
        if len(models) == 1 and models[0]:
            return dict(kind='model_name_declaration', value=models[0], field='parameters',
                        generator_version=None, verified=False)
    return None
