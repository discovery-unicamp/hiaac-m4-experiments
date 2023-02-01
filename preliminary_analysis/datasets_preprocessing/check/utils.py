import numpy as np

def compare_metadata(dataset_normal, dataset_resampled, columns, index_a_fraction=1, index_b_fraction=1):

    for column in columns:
        if column == 'index':
            if not np.all(dataset_normal[column]/index_a_fraction == dataset_resampled[column]/index_b_fraction):
                return False

        elif not np.all(dataset_normal[column] == dataset_resampled[column]):
            return False
    return True


def generate_plot(dataset_normal, dataset_resampled, sample_no):
    pass
