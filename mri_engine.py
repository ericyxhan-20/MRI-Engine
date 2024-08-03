import nibabel as nib

class MRIEngine:
    def __init__(self):
        self.flair_path = '/Users/erichan/Documents/Projects/MRI-Engine/BraTS20_Training_006_flair copy.nii'
    def read_flair(self):
        # Upload image file to Colab's local runtime files first
        # Load the NIfTI file using nibabel
        TRUE_LABEL = 1
        img = nib.load(self.flair_path) 
        type(img) # nibabel.nifti1.Nifti1Image

        # Read the NIfTI image data as a numpy array
        img_npy = img.get_fdata()
        type(img_npy) # numpy.memmap
        img_npy.shape # (240, 240, 155)