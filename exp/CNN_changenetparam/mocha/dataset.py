"""
Dataset preparation for caffe experiments
"""

import collections
import os.path
import random

import maflib.core
import maflib.util
import piccolo

@maflib.util.rule
def crop_samples(task):
    """Crops regions from dataset extracted from Hawk.

    This rule takes following parameters,

    - ``attribute``: attribute name.
    - ``label_map``: :py:class:`maflib.core.Parameter` dict mapping label name
        to label index (0-origin).
    - ``size_config``: cropping configuration as a
        :py:class:`maflib.core.Parameter` consisting of following entries.

            base_aratio
            base_width
            width
            height
            offset_x
            offset_y

    - ``padding``: width of additional bands (left, top, right, bottom) to be
        additionaly extracted with the region specified by ``size_config``.
    - ``min_height``: threshold of height to only extract large enough regions.
    - ``flip``: True if the flipped images should be generated

    and one input node indicating source dataset and two output nodes indicating
    caffe-formatted labeled image path list and directory containing cropped
    images.

    usage:

    .. code-block:: python

       exp(source='/data/tabe/hawk+withtags.objdet/list',
           target='train-dev.caffe.txt images',
           parameters=[{

    """
    attribute = task.parameter['attribute']
    label_map = task.parameter['label_map']
    size_config = task.parameter['size_config']

    size_adjuster = piccolo.SizeAdjuster(**size_config)
    cropper = piccolo.Cropper(
        size_config['width'], size_config['height'], task.parameter['padding'])

    imagelist = piccolo.dataset.load_imagelist(task.inputs[0].abspath())
    annotations = piccolo.dataset.load_annotations(
        task.inputs[0].abspath(), size_adjuster)
    for annotation in annotations:
        piccolo.dataset.ignore_small_boxes(annotation, task.parameter['min_height'])

    output_dir = task.outputs[1].abspath()
    samples_info = piccolo.dataset.crop_samples(output_dir,
                                                annotations,
                                                imagelist,
                                                cropper,
                                                flip = task.parameter['flip'])
    fp = open(task.outputs[0].abspath(), 'w')
    for sample in samples_info:
        if not sample['attributes'].has_key(attribute):
            continue
        if not label_map.has_key(sample['attributes'][attribute]):
            continue
        value = label_map[sample['attributes'][attribute]]
        fp.write("%s %d\n" % (sample['image'], value))


def read_labeled_data(lines_like):
    """Reads lines and converts them to label-to-paths dict.

    :param lines_like: Dataset to be read.
    :type lines_like: List of str or file object, whose lines can be extracted
        by simple for-statement.
    :return: Dict that maps label name to a list of image paths.
    :rtype: dict

    """
    label_to_data = collections.defaultdict(list)
    for line in lines_like:
        cols = line.strip().split()
        if len(cols) == 0:
            continue  # ignore empty line
        if len(cols) == 1:
            cols.append('')  # empty string as a label
        label_to_data[cols[1]].append(cols[0])
    return label_to_data


def make_directory_wise_list(paths):
    """Divides a list of paths into lists each of which corresponds to one
    directory.

    :param list paths: A list of path strings.
    :return: Mapping from directory names to corresponding lists of paths.
    :rtype: dict

    """
    d = {}
    for path in paths:
        dirname = os.path.dirname(path)
        if dirname not in d:
            d[dirname] = []
        d[dirname].append(path)
    return d


def split_directories(d, ratio):
    """Splits a dataset as a set of directories into a pair of sets of
    directories by ratio:(1-ratio).

    Note: this function uses random module, so do not forget to call
    `random.seed` to make it reproducible.

    :param dict d: Mapping from directory names to corresponding lists of paths.
    :param float ratio: Ratio of size of the first list to be returned.
    :return: A pair of lists of paths the first of which contains approximately
        `ratio` part of all paths given.
    :rtype: tuple

    """
    total_count = sum(len(d[key]) for key in d)
    d2_size_lower_bound = int(total_count * (1 - ratio))

    keys = d.keys()
    random.shuffle(keys)

    d1 = []
    d2 = []
    for key in keys:
        if len(d2) < d2_size_lower_bound:
            d2 += d[key]
        else:
            d1 += d[key]

    return d1, d2


def write_dataset(dataset, out):
    """Writes a dataset into file.

    :param list dataset: A list of (path, label) pairs.
    :param out: Output file-like object.
    :type out: File-like

    """
    for path, label in dataset:
        out.write('{0} {1}\n'.format(path, label))


def split_dataset_directories(seed, train_ratio):
    """Generates a task to split a dataset into two subsets of size
    (train_ratio : 1-train_ratio) approximately.

    The dataset is split by directory-wise manner, i.e. images directly under
    the same directory are put into the same subset.

    The task takes one input node and two output nodes. The input node is the
    input dataset. The output nodes are train/test sub-dataset.

    :param int seed: Seed number of random number generator.
    :param float train_ratio: Ratio of size of training data.

    """
    def impl(task):
        random.seed(seed)

        with open(task.inputs[0].abspath()) as labels_file:
            label_to_data = read_labeled_data(labels_file)

        if 'unknown' in label_to_data:
            del label_to_data['unknown']

        label_to_dir = make_directory_wise_list(label_to_data)

        train = []
        test = []
        for label, dir_to_paths in label_to_dir.iteritems():
            d1, d2 = split_directories(dir_to_paths, train_ratio)
            train += [(d, label) for d in d1]
            test += [(d, label) for d in d2]

        with open(task.outputs[0].abspath(), 'w') as out_train:
            write_dataset(train, out_train)
        with open(task.outputs[1].abspath(), 'w') as out_test:
            write_dataset(test, out_test)

    return maflib.core.Rule(fun=impl, dependson=[split_dataset_directories, seed, train_ratio])


def split_dataset(seed, train_ratio):
    """Generates a task to split a dataset into two subsets of size
    (train_ratio : 1-train_ratio) approximately.

    This task does not care about directories where each data live. When the
    images in the same directory are correlated, it is recommended to use
    :py:func:`split_dataset_directories` instead of split_dataset.

    The task takes one input node and two output nodes. The input node is the
    input dataset. The output nodes are train/test sub-dataset.

    :param int seed: Seed number of random number generator.
    :param float train_ratio: Ratio of size of training data.

    """
    def impl(task):
        random.seed(seed)

        with open(task.inputs[0].abspath()) as labels_file:
            label_to_data = read_labeled_data(labels_file)

        if 'unknown' in label_to_data:
            del label_to_data['unknown']

        train = []
        test = []
        for label, paths in label_to_data.iteritems():
            random.shuffle(paths)
            train_size = int(len(paths) * train_ratio)
            train += [(d, label) for d in paths[:train_size]]
            test += [(d, label) for d in paths[train_size:]]

        with open(task.outputs[0].abspath(), 'w') as out_train:
            write_dataset(train, out_train)
        with open(task.outputs[1].abspath(), 'w') as out_test:
            write_dataset(test, out_test)

    return maflib.core.Rule(
        fun=impl, dependson=[split_dataset, seed, train_ratio])


def balance_labels_by_supersampling(task):
    """A task to balance labels in dataset by supersampling.

    It takes one input node and one output node. The input node is the original
    dataset. The output node is super-sampled dataset.
    """

    with open(task.inputs[0].abspath()) as ds_file:
        label_to_data = read_labeled_data(ds_file)

    max_count = max(len(label_to_data[k]) for k in label_to_data)

    with open(task.outputs[0].abspath(), 'w') as out:
        for label, paths in label_to_data.iteritems():
            count = 0
            random.shuffle(paths)
            while count < max_count:
                for path in paths[:min(max_count - count, len(paths))]:
                    out.write('{0} {1}\n'.format(path, label))
                count += len(paths)
