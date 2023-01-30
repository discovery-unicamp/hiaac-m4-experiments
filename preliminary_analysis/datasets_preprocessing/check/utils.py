import numpy as np

def compare_metadata(dataset_normal, dataset_resampled, columns):

    for column in columns:
        if not np.all(dataset_normal[column] == dataset_resampled[column]):
            return False
    return True


def generate_plot(dataset_normal, dataset_resampled, sample_no):
    pass
