"""Inspect actual PDF font resources, including fonts inside form XObjects."""
from pypdf import PdfReader


def font_inventory(path):
    fonts = []
    seen = set()

    def visit(resources):
        if resources is None:
            return
        resources = resources.get_object()
        for font_ref in resources.get('/Font', {}).values():
            font = font_ref.get_object()
            leaves = font.get('/DescendantFonts', [font])
            for leaf in leaves:
                leaf = leaf.get_object()
                descriptor = leaf.get('/FontDescriptor')
                descriptor = descriptor.get_object() if descriptor else {}
                fonts.append({
                    'name': str(leaf.get('/BaseFont', font.get('/BaseFont', ''))),
                    'embedded': any(k in descriptor for k in ('/FontFile', '/FontFile2', '/FontFile3')),
                })
        for ref in resources.get('/XObject', {}).values():
            obj = ref.get_object()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            visit(obj.get('/Resources'))

    for page in PdfReader(path).pages:
        visit(page.get('/Resources'))
    return fonts


def uses_embedded_times_new_roman(path):
    fonts = font_inventory(path)
    allowed = {'TimesNewRomanPSMT', 'TimesNewRomanPS-BoldMT',
               'TimesNewRomanPS-ItalicMT', 'TimesNewRomanPS-BoldItalicMT'}
    return bool(fonts) and all(f['embedded'] and f['name'].split('+')[-1].lstrip('/') in allowed for f in fonts)
