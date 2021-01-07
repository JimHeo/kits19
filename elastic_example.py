from starter_code.utils import load_case
import numpy as np
import elasticdeform as els
import nibabel as nb

vol, seg = load_case(0)
spacing = vol.affine
vol_data = vol.get_data()
seg_data = seg.get_data()
seg_data = seg_data.astype(np.int64)

print(seg_data.shape)

'''
# apply deformation with a random 3 x 3 grid
[vol_deformed, seg_deformed] = els.deform_random_grid([vol_data, seg_data], order=[3, 0])

# vol_nii = nib.Nifti1Image(vol_data, affine=spacing)
# nib.save(vol_nii, "vol.nii.gz")
# seg_nii = nib.Nifti1Image(seg_data, affine=spacing)
# nib.save(seg_nii, "seg.nii.gz")
vol_deformed_nii = nb.Nifti1Image(vol_deformed, affine=spacing)
nb.save(vol_deformed_nii, "vol_deformed.nii.gz")
seg_deformed_nii = nb.Nifti1Image(seg_deformed, affine=spacing)
nb.save(seg_deformed_nii, "seg_deformed.nii.gz")

# imageio.imsave('test_X.png', X)
# imageio.imsave('test_X_deformed.png', X_deformed)
'''