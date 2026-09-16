"""Pixel-identical E150 transforms with bounded full-resolution RGB copies."""
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps
from pixelproof.source_pixel_residual import paired_patch_features
from pixelproof.source_pixel_views import CONDITIONS, geometry, cap, assigned_image


def canonical_patch(image):
    if image.mode!='RGB' or min(image.size)<128:
        raise ValueError('Canonical RGB with a complete128 patch required')
    left,top=(image.width-128)//2,(image.height-128)//2
    with image.crop((left,top,left+128,top+128)) as patch:
        return np.asarray(patch,dtype=np.uint8).copy()


def encoded_loaded(image):
    stream=BytesIO()
    image.save(stream,format='JPEG',quality=75,subsampling=2,optimize=False,progressive=False)
    opened=Image.open(BytesIO(stream.getvalue()))
    try:
        opened.load()
        if opened.mode=='RGB':return opened
        converted=opened.convert('RGB');opened.close();return converted
    except BaseException:
        opened.close();raise


def extract(raw,parent):
    values=[]
    with Image.open(BytesIO(raw)) as opened:
        if opened.width*opened.height>100000000:raise ValueError('Source exceeds100MP budget')
        # Operate on our privately opened image; no full-size copy for the common RGB/no-rotation case.
        ImageOps.exif_transpose(opened,in_place=True)
        image=opened if opened.mode=='RGB' else opened.convert('RGB')
        try:
            sizes=geometry(image.size,parent)
            if any(min(size)<128 for size in sizes):raise ValueError('Every condition must support128 without padding or upsampling')
            values.append(paired_patch_features(canonical_patch(image)))
            with assigned_image(image,parent) as view:
                if view.size!=sizes[1]:raise ValueError('Assigned geometry differs')
                values.append(paired_patch_features(canonical_patch(view)))
            with encoded_loaded(image) as view:
                if view.size!=sizes[2]:raise ValueError('Q75 geometry differs')
                values.append(paired_patch_features(canonical_patch(view)))
            with cap(image,1080) as small:
                with encoded_loaded(small) as view:
                    if view.size!=sizes[3]:raise ValueError('Social geometry differs')
                    values.append(paired_patch_features(canonical_patch(view)))
        finally:
            if image is not opened:image.close()
    result=np.stack(values)
    if result.shape!=(4,2,300) or result.dtype!=np.float32 or not np.isfinite(result).all() or \
            (result<0).any() or not np.allclose(result.reshape(4,2,12,25).sum(axis=-1),1,rtol=0,atol=1e-6):
        raise ValueError('Incomplete normalized four-condition histograms')
    return result,np.asarray(sizes,dtype=np.int32)
