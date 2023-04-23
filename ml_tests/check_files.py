import os
import warnings

import numpy as np
from tqdm import tqdm
import nibabel as nib


def check_file_listing(list_of_files: list,
                       import_library: str = 'nibabel',
                       debug_verbose: bool = False):

    info_of_files = []
    read_problem_of_files = []

    print('Inspecting NIfTI files on disk')
    for (i, filepath) in enumerate(list_of_files):

        if import_library == 'nibabel':
            info_out, read_problem = nibabel_file_checker(filepath=filepath, debug_verbose=debug_verbose)
            info_of_files.append(info_out)
            read_problem_of_files.append(read_problem)
        else:
            raise NotImplementedError('Unknown file import library = "{}"'.format(import_library))

    size_stats = nibabel_summary_of_all_files(info_of_files, read_problem_of_files)

    return size_stats, info_of_files, read_problem_of_files


def nibabel_summary_of_all_files(info_of_files: list, read_problem_of_files: list):

    def get_size_stats_per_coord(list_of_sizes: list) -> dict:
        return {'mean': np.mean(list_of_sizes),
                'median': np.median(list_of_sizes),
                'min': np.min(list_of_sizes),
                'max': np.max(list_of_sizes),
                'n': len(list_of_sizes)
                }


    def get_spatial_size_range(info_of_files, key_size: str = 'volume_shape'):
        x, y, z = [], [], []
        for info_of_file in info_of_files:
            x.append(info_of_file['volume_shape'][0])
            y.append(info_of_file['volume_shape'][1])
            z.append(info_of_file['volume_shape'][2])
        return {'x': get_size_stats_per_coord(x),
                'y': get_size_stats_per_coord(y),
                'z': get_size_stats_per_coord(z)}

    def print_size_stats(size_stats) -> None:
        print('Stats of the spatial dimensions of the dataset:')
        for key_in in size_stats.keys():
            print_string = ' {}: '.format(key_in)
            for stat_key in size_stats[key_in].keys():
                value_in = size_stats[key_in][stat_key]
                if isinstance(value_in, float):
                    print_string += '{}={:.2f} '.format(stat_key, value_in)
                else:
                    print_string += '{}={} '.format(stat_key, value_in)
            print(print_string)
            if key_in == 'z':
                print(' Note! When you are applying crops, your minimum z size is = {} voxels, '
                      'and thus your crops must be smaller than this in z-dimension'.format(size_stats[key_in]['min']))

    size_stats = get_spatial_size_range(info_of_files)
    print_size_stats(size_stats)

    return size_stats


def nibabel_file_checker(filepath: str, debug_verbose: bool = False):
    """
    See e.g. https://nipy.org/nibabel/nifti_images.html
    :param filepath: full path on the disk
    :param verbose: if you want to display stuff like image res, etc.
    :return:
    """
    fname = os.path.split(filepath)[1]
    try:
        n1_img = nib.load(filepath)
        n1_header = n1_img.header
        dict_out = {'n1_header': n1_header,
                    'volume_shape': n1_img.shape}
        if debug_verbose:
            print(' dims = {} ({})'.format(dict_out['volume_shape'], fname))
        read_problem = np.nan

    except Exception as e:
        warnings.warn('Problem opening the file "{}", e = {}'.format(fname, e))
        dict_out = None
        read_problem = fname

    return dict_out, read_problem
