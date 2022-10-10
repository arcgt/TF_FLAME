'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.
For comments or questions, please email us at flame@tue.mpg.de
'''


import os
import six
import argparse
import numpy as np
import tensorflow as tf
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer
from utils.landmarks import load_binary_pickle, load_embedding, tf_get_model_lmks, create_lmk_spheres

from tf_smpl.batch_smpl import SMPL
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt

def zero_pose(target_3d_mesh_fname, model_fname, weights, show_fitting=True):
    '''
    Fit FLAME to 3D mesh in correspondence to the FLAME mesh (i.e. same number of vertices, same mesh topology)
    :param target_3d_mesh_fname:    target 3D mesh filename
    :param model_fname:             saved FLAME model
    :param weights:             weights of the individual objective functions
    :return: a mesh with the fitting results
    '''

    target_mesh = Mesh(filename=target_3d_mesh_fname)

    tf_trans_fitting = tf.Variable(np.zeros((1,3)), name="trans_fitting", dtype=tf.float64, trainable=True)
    tf_rot_fitting = tf.Variable(np.zeros((1,3)), name="rot_fitting", dtype=tf.float64, trainable=True)
    tf_pose_fitting = tf.Variable(np.zeros((1,12)), name="pose_fitting", dtype=tf.float64, trainable=True)
    tf_shape_fitting = tf.Variable(np.zeros((1,300)), name="shape_fitting", dtype=tf.float64, trainable=True)
    tf_exp_fitting = tf.Variable(np.zeros((1,100)), name="expression_fitting", dtype=tf.float64, trainable=True)
    smpl_fitting = SMPL(model_fname)
    tf_model_fitting = tf.squeeze(smpl_fitting(tf_trans_fitting,
                               tf.concat((tf_shape_fitting, tf_exp_fitting), axis=-1),
                               tf.concat((tf_rot_fitting, tf_pose_fitting), axis=-1)))

    tf_trans = tf.Variable(np.zeros((1,3)), name="trans", dtype=tf.float64, trainable=True)
    tf_rot = tf.Variable(np.zeros((1,3)), name="rot", dtype=tf.float64, trainable=True)
    tf_pose = tf.Variable(np.zeros((1,12)), name="pose", dtype=tf.float64, trainable=True)
    tf_shape = tf.Variable(np.zeros((1,300)), name="shape", dtype=tf.float64, trainable=True)
    tf_exp = tf.Variable(np.zeros((1,100)), name="expression", dtype=tf.float64, trainable=True)
    smpl = SMPL(model_fname)
    tf_model = tf.squeeze(smpl(tf_trans,
                               tf.concat((tf_shape, tf_exp), axis=-1),
                               tf.concat((tf_rot, tf_pose), axis=-1)))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        mesh_dist = tf.reduce_sum(tf.square(tf.subtract(tf_model_fitting, target_mesh.v)))
        neck_pose_reg = tf.reduce_sum(tf.square(tf_pose_fitting[:,:3]))
        jaw_pose_reg = tf.reduce_sum(tf.square(tf_pose_fitting[:,3:6]))
        eyeballs_pose_reg = tf.reduce_sum(tf.square(tf_pose_fitting[:,6:]))
        shape_reg = tf.reduce_sum(tf.square(tf_shape_fitting))
        exp_reg = tf.reduce_sum(tf.square(tf_exp_fitting))

        # Optimize global transformation first
        vars = [tf_trans_fitting, tf_rot_fitting]
        loss = mesh_dist
        optimizer = scipy_pt(loss=loss, var_list=vars, method='BFGS', options={'disp': 1})
        print('Optimize rigid transformation')
        optimizer.minimize(session)

        # Optimize for the model parameters
        vars = [tf_trans_fitting, tf_rot_fitting, tf_pose_fitting, tf_shape_fitting, tf_exp_fitting]
        loss = weights['data'] * mesh_dist + weights['shape'] * shape_reg + weights['expr'] * exp_reg + \
               weights['neck_pose'] * neck_pose_reg + weights['jaw_pose'] * jaw_pose_reg + weights['eyeballs_pose'] * eyeballs_pose_reg

        optimizer = scipy_pt(loss=loss, var_list=vars, method='BFGS', options={'disp': 1})
        print('Optimize model parameters')
        optimizer.minimize(session)

        print('Fitting done')

        assign_shape = tf.assign(tf_shape, tf_shape_fitting)
        session.run([assign_shape])

        if show_fitting:
            # Visualize fitting
            mv = MeshViewer()
            fitting_mesh = Mesh(session.run(tf_model), smpl.f)
            fitting_mesh.set_vertex_colors('light sky blue')

            mv.set_static_meshes([target_mesh, fitting_mesh])
            six.moves.input('Press key to continue')

        return Mesh(session.run(tf_model), smpl.f)


def run_corresponding_mesh_fitting():
    # Path of the FLAME model
    model_fname = './models/generic_model.pkl'
    # model_fname = '/models/female_model.pkl'
    # model_fname = '/models/male_model.pkl'

    # target 3D mesh in dense vertex-correspondence to the model
    target_mesh_path = './data/arya_coarse.ply'

    # Output filename
    out_mesh_fname = './results/arya_zero_pose.ply'

    weights = {}
    # Weight of the data term
    weights['data'] = 1000.0
    # Weight of the shape regularizer (the lower, the less shape is constrained)
    weights['shape'] = 1e-4
    # Weight of the expression regularizer (the lower, the less expression is constrained)
    weights['expr']  = 1e-4
    # Weight of the neck pose (i.e. neck rotationh around the neck) regularizer (the lower, the less neck pose is constrained)
    weights['neck_pose'] = 1e-4
    # Weight of the jaw pose (i.e. jaw rotation for opening the mouth) regularizer (the lower, the less jaw pose is constrained)
    weights['jaw_pose'] = 1e-4
    # Weight of the eyeball pose (i.e. eyeball rotations) regularizer (the lower, the less eyeballs pose is constrained)
    weights['eyeballs_pose'] = 1e-4
    # Show landmark fitting (default: red = target landmarks, blue = fitting landmarks)
    show_fitting = True

    result_mesh = zero_pose(target_mesh_path, model_fname, weights, show_fitting=show_fitting)

    if not os.path.exists(os.path.dirname(out_mesh_fname)):
        os.makedirs(os.path.dirname(out_mesh_fname))

    result_mesh.write_ply(out_mesh_fname)


if __name__ == '__main__':
    run_corresponding_mesh_fitting()
