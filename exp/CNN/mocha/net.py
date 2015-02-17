"""
Utility functions and tasks to manipulate caffe prototxt settings.
"""

import string

import caffe.proto.caffe_pb2
import google.protobuf.text_format

import maflib.core
import maflib.util

@maflib.util.rule
def configure_data_layer(task):
    """Task to add some configuration to the first input data layer of caffe
    network.

    This task takes following inputs:
        0. Config prototxt file (source, batchsize, cropsize and mirror can be
           omitted)
        1. Source dataset
        2. Mean file (optional)

    and following optional parameters:
        'batchsize': size of mini-batch
        'cropsize': size of cropped image
        'mirror': use mirror or not

    """
    net_str = task.inputs[0].read()
    net = caffe.proto.caffe_pb2.NetParameter()
    google.protobuf.text_format.Merge(net_str, net)

    print net.layers
    first_layer = net.layers[0]
    data_layer = first_layer.data_param

    data_layer.source = task.inputs[1].abspath()

    if len(task.inputs) >= 3:
        data_layer.mean_file = task.inputs[2].abspath()

    # TODO(beam2d): Fix parameter names to be same as caffe's parameter names.
    if 'batchsize' in task.parameter:
        data_layer.batch_size = task.parameter['batchsize']
    if 'cropsize' in task.parameter:
        data_layer.crop_size = task.parameter['cropsize']
    if 'mirror' in task.parameter:
        data_layer.mirror = task.parameter['mirror']
    if 'shuffle_images' in task.parameter:
        data_layer.shuffle = task.parameter['shuffle_images']

    result = google.protobuf.text_format.MessageToString(net)
    task.outputs[0].write(result)


@maflib.util.rule
def subst_from_template(task):
    """Rule to substitute the parameter to prototxt template.

    It takes one input and one output nodes. The input node is prototxt
    template. The output node is substituted prototxt file.

    """
    template = string.Template(task.inputs[0].read())
    task.outputs[0].write(template.safe_substitute(task.parameter))


@maflib.util.rule
def subst_and_repeat_from_template(task):
    """Rule to substitute the parameter and given dictionaries to prototxt
    template and repeat it.

    It takes one input and one output nodes. The input node is prototxt
    template. For each entry of the parameter ``ds``, parameter and the entry
    are simultaneously substituted to the prototxt template and the result is
    appended to the output prototxt.

    """
    template = string.Template(task.inputs[0].read())
    with open(task.outputs[0].abspath(), 'w') as out:
        for d in task.parameter['ds']:
            updated = dict(task.parameter)  # copy
            updated.update(d)  # higher priority to d
            out.write(template.safe_substitute(updated))


@maflib.util.rule
def create_solver(task):
    """Rule to create a solver prototxt of caffe.

    The rule takes following input nodes:

        0. Training network prototxt
        1. Testing network prototxt

    and arbitrary parameters which are used to fill the solver parameter.

    """
    solver = caffe.proto.caffe_pb2.SolverParameter()
    solver.train_net = task.inputs[0].abspath()
    solver.test_net = task.inputs[1].abspath()
    for key in task.parameter:
        if hasattr(solver, key):
            if key == 'lr_change_at':
                for p in task.parameter[key]:
                    getattr(solver, key).append(p)
            else:
                setattr(solver, key, task.parameter[key])
    task.outputs[0].write(
        google.protobuf.text_format.MessageToString(solver))


@maflib.util.rule
def edit_solver(task):
    """Rule to edit a solver prototxt.

    The rule takes one input node (original prototxt) and one output node (
    modified prototxt). Parameters are substituted to corresponding entries of
    the solver protobuf message if possible.

    """
    solver_str = task.inputs[0].read()
    solver = caffe.proto.caffe_pb2.SolverParameter()
    google.protobuf.text_format.Merge(solver_str, solver)

    for key in task.parameter:
        if hasattr(solver, key):
            setattr(solver, key, task.parameter[key])
    task.outputs[0].write(
        google.protobuf.text_format.MessageToString(solver))

