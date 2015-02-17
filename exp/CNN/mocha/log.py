"""
Utility functions and tasks to read logs of caffe.
"""

import collections
import json
import re

import matplotlib
matplotlib.use('Agg', warn=False)
import matplotlib.pyplot

import maflib.core
import maflib.plot

def read_caffe_log(log_file, score_keys):
    """Extracts time series of scores from caffe log file.

    :param log_file: File-like object containing the log of caffe training
        binary.
    :type log_file: File-like
    :param score_keys: Key strings for score values.
    :type score_keys: Iterable
    :return: A list of dicts each of which describes the result of each
        iteration visible in the log.
    :rtype: list

    """
    iter_to_values = collections.defaultdict(dict)
    test_iter = None
    for line in log_file:
        m1 = re.search('Iteration (\\d+), Testing net', line)
        if m1:
            test_iter = int(m1.group(1))
            continue

        m2 = re.search('Test score #(\\d): (.+)$', line)
        if m2:
            score_idx = int(m2.group(1))
            value = float(m2.group(2))
            if len(score_keys) > score_idx:
                iter_to_values[test_iter][score_keys[score_idx]] = value
            continue

        m3 = re.search('Iteration (\\d+), loss = (.+)$', line)
        if m3:
            loss_iter = int(m3.group(1))
            iter_to_values[loss_iter]['loss'] = float(m3.group(2))

    result = []
    for iteration in sorted(iter_to_values.keys()):
        value = iter_to_values[iteration]
        value['iteration'] = iteration
        result.append(value)

    return result


def add_criteria(d):
    """Adds some scoring criteria to result dict."""

    if 'accuracy' in d:
        d['error'] = 1 - d['accuracy']
    if 'tp' in d and 'fn' in d:
        d['miss-rate'] = d['fn'] / (d['tp'] + d['fn'])
    if 'tn' in d and 'fp' in value:
        d['fp-rate'] = d['fp'] / (d['tn'] + t['fp'])


def plot(result, to, key, ylim=None):
    """Plots the result extracted from log.

    :param list result: A list of iteration results.
    :param str to: Output image path.
    :param str key: Key of y-axis.
    :param int ylim: ylim of matplotlib plot.

    """
    figure = matplotlib.pyplot.figure()
    axes = figure.add_subplot(111)
    axes.set_xlabel('iteration')
    axes.set_ylabel(key)
    if ylim:
        axes.set_ylim(ylim)
    iters = [r['iteration'] for r in result if key in r]
    vals = [r[key] for r in result if key in r]
    axes.plot(iters, vals)
    figure.savefig(to)


def extract_results_from_log(keys):
    """Generates a task to Extract the sequence of scores/losses from log file
    of caffe training binary.

    Each of keys represents a name of corresponding score entry, i.e. i-th key
    is the name of i-th score. It can be smaller than the number of score
    entries for each iteration; in that case, the remaining scores are ignored.

    :param list keys: A list of key strings.
    :return: A task.

    """
    def impl(task):
        with open(task.inputs[0].abspath()) as log:
            result = read_caffe_log(log, keys)
        for r in result:
            add_criteria(r)
        task.outputs[0].write(json.dumps(result, indent=4))

    return maflib.core.Rule(
        fun=impl, dependson=[extract_results_from_log, keys, read_caffe_log])


def plot_result(y_range, value='error', key='arch_type', legend_loc='upper right'):
    """Plots value for each iteration.

    :param y_range: Range of y axis or None (where the range is automatically
        determined by matplotlib).
    :type y_range: tuple or None
    :param str value: Key for y axis.
    :param str key: Key for indicating different lines. Its values are used in
        the legend.
    :param str legend_loc: Location of legend (used in matplotlib).

    """

    @maflib.plot.plot_by
    def impl(figure, plot_data, parameter):
        axes = figure.add_subplot(111)
        axes.set_xlabel('iteration')
        axes.set_ylabel(value)
        if y_range:
            axes.set_ylim(y_range)
        key_to_xys = plot_data.get_data_2d('iteration', value, key=key)
        for k in sorted(key_to_xys):
            x, y = key_to_xys[k]
            axes.plot(x, y, label='{0}={1}'.format(key, k))
        axes.legend(loc=legend_loc)

    return maflib.core.Rule(
        fun=impl,
        dependson=[plot_result, y_range, value, key, legend_loc])
