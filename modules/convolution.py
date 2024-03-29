'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c)  2015, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import tensorflow as tf
from module import Module
import variables
import pdb
import activations

from math import ceil

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops


class Convolution(Module):
    '''
    Convolutional Layer
    '''

    def __init__(self, output_depth, batch_size=None, input_dim = None, input_depth=None, kernel_size=5, stride_size=2, act = 'linear', keep_prob=1.0, pad = 'SAME',name="conv2d"):
        self.name = name
        #self.input_tensor = input_tensor
        Module.__init__(self)
        
        
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.input_depth = input_depth
        

        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.act = act
        self.keep_prob = keep_prob
        self.pad = pad
        

    def check_input_shape(self):
        inp_shape = self.input_tensor.get_shape().as_list()
        try:
            if len(inp_shape)!=4:
                mod_shape = [self.batch_size, self.input_dim,self.input_dim,self.input_depth]
                self.input_tensor = tf.reshape(self.input_tensor, mod_shape)
        except:
            raise ValueError('Expected dimension of input tensor: 4')

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        #pdb.set_trace()
        self.check_input_shape()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()
        
        # init weights
        self.weights_shape = [self.kernel_size, self.kernel_size, in_depth, self.output_depth]
        self.strides = [1,self.stride_size, self.stride_size,1]
        with tf.variable_scope(self.name):
            self.weights = variables.weights(self.weights_shape)
            self.biases = variables.biases(self.output_depth)
        
        with tf.name_scope(self.name):
            conv = tf.nn.conv2d(self.input_tensor, self.weights, strides = self.strides, padding=self.pad)
            conv = tf.reshape(tf.nn.bias_add(conv, self.biases), conv.get_shape().as_list())
            conv = tf.layers.batch_normalization(conv)
            
            if isinstance(self.act, str): 
                self.activations = activations.apply(conv, self.act)
            elif hasattr(self.act, '__call__'):
                self.activations = self.act(conv)
                
            if self.keep_prob<1.0:
                self.activations = tf.nn.dropout(self.activations, keep_prob=self.keep_prob)
            
            tf.summary.histogram('activations', self.activations)
            tf.summary.histogram('weights', self.weights)
            tf.summary.histogram('biases', self.biases)

        return self.activations

    def padding(self, i,j, pad_in_h, pad_in_w, hstride, wstride, hf, wf):
        pad_bottom = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
        pad_top = i*hstride
        pad_right = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
        pad_left = j*wstride
        return [[pad_top, pad_bottom], [pad_left, pad_right]]
    
    def _simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        import time; start_time = time.time()
        
        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape)!=4:
            self.R = tf.reshape(self.R, activations_shape)

        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        op1 = tf.extract_image_patches(self.input_tensor, ksizes=[1, hf,wf, 1], strides=[1, hstride, wstride, 1], rates=[1, 1, 1, 1], padding=self.pad)
        p_bs, p_h, p_w, p_c = op1.get_shape().as_list()
        image_patches = tf.reshape(op1, [p_bs,p_h,p_w, hf, wf, in_depth])
        #pdb.set_trace()
        Z = tf.expand_dims(self.weights, 0) * tf.expand_dims( image_patches, -1)
        Zs = tf.reduce_sum(Z, [3,4,5], keep_dims=True)  #+ tf.expand_dims(self.biases, 0)
        stabilizer = 1e-12*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
        Zs += stabilizer
        result =   tf.reduce_sum((Z/Zs) * tf.reshape(self.R, [in_N,Hout,Wout,1,1,1,NF]), 6)
        Rx = self.patches_to_images(tf.reshape(result, [p_bs, p_h, p_w, p_c]), in_N, in_h, in_w, in_depth, Hout, Wout, hf,wf, hstride,wstride )
        
        total_time = time.time() - start_time
        print(total_time)
        return Rx

    def _epsilon_lrp(self,R, epsilon):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        import time; start_time = time.time()
        
        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape)!=4:
            self.R = tf.reshape(self.R, activations_shape)

        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        op1 = tf.extract_image_patches(self.input_tensor, ksizes=[1, hf,wf, 1], strides=[1, hstride, wstride, 1], rates=[1, 1, 1, 1], padding='VALID')
        p_bs, p_h, p_w, p_c = op1.get_shape().as_list()
        image_patches = tf.reshape(op1, [p_bs,p_h,p_w, hf, wf, in_depth])
        
        Z = tf.expand_dims(self.weights, 0) * tf.expand_dims( image_patches, -1)
        Zs = tf.reduce_sum(Z, [3,4,5], keep_dims=True)  #+ tf.expand_dims(self.biases, 0)
        stabilizer = epsilon*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
        Zs += stabilizer
        result =   tf.reduce_sum((Z/Zs) * tf.reshape(self.R, [in_N,Hout,Wout,1,1,1,NF]), 6)
        Rx = self.patches_to_images(tf.reshape(result, [p_bs, p_h, p_w, p_c]), in_N, in_h, in_w, in_depth, Hout, Wout, hf,wf, hstride,wstride )
        
        total_time = time.time() - start_time
        print(total_time)
        return Rx

    def _ww_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        import time; start_time = time.time()
        
        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape)!=4:
            self.R = tf.reshape(self.R, activations_shape)

        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        op1 = tf.extract_image_patches(self.input_tensor, ksizes=[1, hf,wf, 1], strides=[1, hstride, wstride, 1], rates=[1, 1, 1, 1], padding='VALID')
        p_bs, p_h, p_w, p_c = op1.get_shape().as_list()
        #image_patches = tf.reshape(op1, [p_bs,p_h,p_w, hf, wf, in_depth])
        image_patches = tf.ones([p_bs,p_h,p_w, hf, wf, in_depth])
        
        
        Z = tf.square(tf.expand_dims(self.weights, 0)) * tf.expand_dims( image_patches, -1)
        Zs = tf.reduce_sum(Z, [3,4,5], keep_dims=True)  #+ tf.expand_dims(self.biases, 0)
        stabilizer = 1e-12*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
        Zs += stabilizer
        result =   tf.reduce_sum((Z/Zs) * tf.reshape(self.R, [in_N,Hout,Wout,1,1,1,NF]), 6)
        Rx = self.patches_to_images(tf.reshape(result, [p_bs, p_h, p_w, p_c]), in_N, in_h, in_w, in_depth, Hout, Wout, hf,wf, hstride,wstride )
        
        total_time = time.time() - start_time
        print(total_time)
        return Rx

    def _flat_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        import time; start_time = time.time()
        
        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape)!=4:
            self.R = tf.reshape(self.R, activations_shape)

        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        op1 = tf.extract_image_patches(self.input_tensor, ksizes=[1, hf,wf, 1], strides=[1, hstride, wstride, 1], rates=[1, 1, 1, 1], padding='VALID')
        p_bs, p_h, p_w, p_c = op1.get_shape().as_list()
        #image_patches = tf.reshape(op1, [p_bs,p_h,p_w, hf, wf, in_depth])
        image_patches = tf.ones([p_bs,p_h,p_w, hf, wf, in_depth])
        
        
        Z = tf.expand_dims( image_patches, -1)
        Zs = tf.reduce_sum(Z, [3,4,5], keep_dims=True)  #+ tf.expand_dims(self.biases, 0)
        stabilizer = 1e-12*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
        Zs += stabilizer
        result =   tf.reduce_sum((Z/Zs) * tf.reshape(self.R, [in_N,Hout,Wout,1,1,1,NF]), 6)
        Rx = self.patches_to_images(tf.reshape(result, [p_bs, p_h, p_w, p_c]), in_N, in_h, in_w, in_depth, Hout, Wout, hf,wf, hstride,wstride )
        
        total_time = time.time() - start_time
        print(total_time)
        return Rx
    
    # OLD METHODS BELOW
    def __simple_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        import time; start_time = time.time()
        
        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape)!=4:
            self.R = tf.reshape(self.R, activations_shape)

        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        #out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()


        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w

            # pr = (out_h -1) * hstride + hf - in_h
            # pc =  (out_w -1) * wstride + wf - in_w
            p_top = pr/2
            p_bottom = pr-(pr/2)
            p_left = pc/2
            p_right = pc-(pc/2)
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[p_top,p_bottom],[p_left, p_right],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            self.pad_input_tensor = self.input_tensor
            
        pad_in_N, pad_in_h, pad_in_w, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        
        pdb.set_trace()
        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0),0)
        for i in xrange(Hout):
            for j in xrange(Wout):
                input_slice = self.pad_input_tensor[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ]
                term2 =  tf.expand_dims(input_slice, -1)
                #pdb.set_trace()
                Z = term1 * term2
                t1 = tf.reduce_sum(Z, [1,2,3], keep_dims=True)
                #Zs = t1 + t2
                Zs = t1
                stabilizer = 1e-8*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
                Zs += stabilizer
                result = tf.reduce_sum((Z/Zs) * tf.expand_dims(self.R[:,i:i+1,j:j+1,:], 3), 4)
                
                #pdb.set_trace()
                #pad each result to the dimension of the out
                pad_bottom = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
                pad_top = i*hstride
                pad_right = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
                pad_left = j*wstride
                result = tf.pad(result, [[0,0],[pad_top, pad_bottom],[pad_left, pad_right],[0,0]], "CONSTANT")
                # print(i,j)
                # print(i*hstride, i*hstride+hf , j*wstride, j*wstride+wf)
                # print(pad_top, pad_bottom,pad_left, pad_right)
                Rx+= result
        #pdb.set_trace()
        total_time = time.time() - start_time
        print(total_time)
        if self.pad=='SAME':
            return Rx[:, (pc/2):in_w+(pc/2), (pr/2):in_h+(pr/2), :]
        elif self.pad =='VALID':
            return Rx
        
    def __flat_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        
        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape)!=4:
            self.R = tf.reshape(self.R, activations_shape)

        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        #out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()


        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w

            # pr = (out_h -1) * hstride + hf - in_h
            # pc =  (out_w -1) * wstride + wf - in_w
            p_top = pr/2
            p_bottom = pr-(pr/2)
            p_left = pc/2
            p_right = pc-(pc/2)
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[p_top,p_bottom],[p_left, p_right],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            self.pad_input_tensor = self.input_tensor
            
        pad_in_N, pad_in_h, pad_in_w, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        
        #pdb.set_trace()
        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0),0)
        for i in xrange(Hout):
            for j in xrange(Wout):
                Z = tf.ones([N, hf,wf,df,NF], dtype=tf.float32)
                Zs = tf.reduce_sum(Z, [1,2,3], keep_dims=True)
                
                result = tf.reduce_sum((Z/Zs) * tf.expand_dims(self.R[:,i:i+1,j:j+1,:], 3), 4)
                
                #pdb.set_trace()
                #pad each result to the dimension of the out
                pad_bottom = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
                pad_top = i*hstride
                pad_right = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
                pad_left = j*wstride
                result = tf.pad(result, [[0,0],[pad_top, pad_bottom],[pad_left, pad_right],[0,0]], "CONSTANT")
                # print(i,j)
                # print(i*hstride, i*hstride+hf , j*wstride, j*wstride+wf)
                # print(pad_top, pad_bottom,pad_left, pad_right)
                Rx+= result
        #pdb.set_trace()
        if self.pad=='SAME':
            return Rx[:, (pc/2):in_w+(pc/2), (pr/2):in_h+(pr/2), :]
        elif self.pad =='VALID':
            return Rx
        
    def __ww_lrp(self,R):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
        
        self.R = R
        R_shape = self.R.get_shape().as_list()
        activations_shape = self.activations.get_shape().as_list()
        if len(R_shape)!=4:
            self.R = tf.reshape(self.R, activations_shape)

        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        #out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()


        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w

            # pr = (out_h -1) * hstride + hf - in_h
            # pc =  (out_w -1) * wstride + wf - in_w
            p_top = pr/2
            p_bottom = pr-(pr/2)
            p_left = pc/2
            p_right = pc-(pc/2)
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[p_top,p_bottom],[p_left, p_right],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            self.pad_input_tensor = self.input_tensor
            
        pad_in_N, pad_in_h, pad_in_w, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        
        #pdb.set_trace()
        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0),0)
        for i in xrange(Hout):
            for j in xrange(Wout):
                Z = tf.square(tf.expand_dims(self.weights, 0))  
                Zs = tf.reduce_sum(Z, [1,2,3], keep_dims=True) 

                result = tf.reduce_sum((Z/Zs) * tf.expand_dims(self.R[:,i:i+1,j:j+1,:], 3), 4)
                
                #pdb.set_trace()
                #pad each result to the dimension of the out
                pad_bottom = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
                pad_top = i*hstride
                pad_right = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
                pad_left = j*wstride
                result = tf.pad(result, [[0,0],[pad_top, pad_bottom],[pad_left, pad_right],[0,0]], "CONSTANT")
                Rx+= result
        #pdb.set_trace()
        if self.pad=='SAME':
            return Rx[:, (pc/2):in_w+(pc/2), (pr/2):in_h+(pr/2), :]
        elif self.pad =='VALID':
            return Rx

    def __epsilon_lrp(self,R, epsilon):
        '''
        LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
        '''
        #pdb.set_trace()
        self.R = R
        R_shape = self.R.get_shape().as_list()
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1]+activations_shape[1:])
        
        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            self.pad_input_tensor = self.input_tensor

        pad_in_N, pad_in_h, pad_in_w, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        
        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0),0)
        for i in xrange(Hout):
            for j in xrange(Wout):
                input_slice = self.pad_input_tensor[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ]
                term2 =  tf.expand_dims(input_slice, -1)
                Z = term1 * term2
                t1 = tf.reduce_sum(Z, [1,2,3], keep_dims=True)
                Zs = t1 + t2
                stabilizer = epsilon*(tf.where(tf.greater_equal(Zs,0), tf.ones_like(Zs, dtype=tf.float32), tf.ones_like(Zs, dtype=tf.float32)*-1))
                Zs += stabilizer
                result = tf.reduce_sum((Z/Zs) * tf.expand_dims(self.R[:,i:i+1,j:j+1,:], 3), 4)
                
                #pad each result to the dimension of the out
                pad_right = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
                pad_left = i*hstride
                pad_bottom = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
                pad_up = j*wstride
                result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")
                Rx+= result
                
        if self.pad=='SAME':
            return Rx[:, (pc/2):in_w+(pc/2), (pr/2):in_h+(pr/2), :]
        elif self.pad =='VALID':
            return Rx
        

    def _alphabeta_lrp(self,R, alpha):
        '''
        LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
        '''
        beta = 1 - alpha
        self.R = R
        R_shape = self.R.get_shape().as_list()
        
        if len(R_shape)!=4:
            activations_shape = self.activations.get_shape().as_list()
            self.R = tf.reshape(self.R, [-1]+activations_shape[1:])

        N,Hout,Wout,NF = self.R.get_shape().as_list()
        hf,wf,df,NF = self.weights_shape
        _, hstride, wstride, _ = self.strides

        out_N, out_h, out_w, out_depth = self.activations.get_shape().as_list()
        in_N, in_h, in_w, in_depth = self.input_tensor.get_shape().as_list()

        if self.pad == 'SAME':
            pr = (Hout -1) * hstride + hf - in_h
            pc =  (Wout -1) * wstride + wf - in_w
            self.pad_input_tensor = tf.pad(self.input_tensor, [[0,0],[pr/2, (pr-(pr/2))],[pc/2,(pc - (pc/2))],[0,0]], "CONSTANT")
        elif self.pad == 'VALID':
            self.pad_input_tensor = self.input_tensor

        pad_in_N, pad_in_h, pad_in_w, pad_in_depth = self.pad_input_tensor.get_shape().as_list()
        Rx = tf.zeros_like(self.pad_input_tensor, dtype = tf.float32)
        
        term1 = tf.expand_dims(self.weights, 0)
        t2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.biases, 0), 0), 0)
        for i in xrange(Hout):
            for j in xrange(Wout):
                input_slice = self.pad_input_tensor[:, i*hstride:i*hstride+hf , j*wstride:j*wstride+wf , : ]
                term2 =  tf.expand_dims(input_slice, -1)
                #pdb.set_trace()
                Z = term1 * term2

                if not alpha == 0:
                    Zp = tf.where(tf.greater(Z,0),Z, tf.zeros_like(Z))
                    t2 = tf.expand_dims(tf.expand_dims(tf.where(tf.greater(self.biases,0),self.biases, tf.zeros_like(self.biases)), 0 ), 0)
                    t1 = tf.expand_dims( tf.reduce_sum(Zp, 1), 1)
                    Zsp = t1 + t2
                    Ralpha = alpha * tf.reduce_sum((Zp / Zsp) * tf.expand_dims(self.R[:,i:i+1,j:j+1,:], 3),4)
                else:
                    Ralpha = 0

                if not beta == 0:
                    Zn = tf.where(tf.less(Z,0),Z, tf.zeros_like(Z))
                    t2 = tf.expand_dims(tf.expand_dims(tf.where(tf.less(self.biases,0),self.biases, tf.zeros_like(self.biases)), 0 ), 0)
                    t1 = tf.expand_dims( tf.reduce_sum(Zn, 1), 1)
                    Zsp = t1 + t2
                    Rbeta = beta * tf.reduce_sum((Zn / Zsp) * tf.expand_dims(self.R[:,i:i+1,j:j+1,:], 3),4)
                else:
                    Rbeta = 0

                result = Ralpha + Rbeta
                
                #pad each result to the dimension of the out
                pad_right = pad_in_h - (i*hstride+hf) if( pad_in_h - (i*hstride+hf))>0 else 0
                pad_left = i*hstride
                pad_bottom = pad_in_w - (j*wstride+wf) if ( pad_in_w - (j*wstride+wf) > 0) else 0
                pad_up = j*wstride
                result = tf.pad(result, [[0,0],[pad_left, pad_right],[pad_up, pad_bottom],[0,0]], "CONSTANT")
                #print pad_bottom, pad_left, pad_right, pad_up
                #pdb.set_trace()
                Rx+= result

        if self.pad=='SAME':
            return Rx[:, (pc/2):in_w+(pc/2), (pr/2):in_h+(pr/2), :]
        elif self.pad =='VALID':
            return Rx
        
    #def patches_to_images(self, grad, in_N, in_h, in_w, in_depth, out_h, out_w, hf,wf, hstride,wstride ):
    def patches_to_images(self, grad, batch_size, rows_in, cols_in, channels, rows_out, cols_out, ksize_r, ksize_c, stride_h, stride_r ):
        rate_r = 1
        rate_c = 1
        padding = self.pad
        
        
        ksize_r_eff = ksize_r + (ksize_r - 1) * (rate_r - 1)
        ksize_c_eff = ksize_c + (ksize_c - 1) * (rate_c - 1)

        if padding == 'SAME':
            rows_out = int(ceil(rows_in / stride_r))
            cols_out = int(ceil(cols_in / stride_h))
            pad_rows = ((rows_out - 1) * stride_r + ksize_r_eff - rows_in) // 2
            pad_cols = ((cols_out - 1) * stride_h + ksize_c_eff - cols_in) // 2

        elif padding == 'VALID':
            rows_out = int(ceil((rows_in - ksize_r_eff + 1) / stride_r))
            cols_out = int(ceil((cols_in - ksize_c_eff + 1) / stride_h))
            pad_rows = (rows_out - 1) * stride_r + ksize_r_eff - rows_in
            pad_cols = (cols_out - 1) * stride_h + ksize_c_eff - cols_in

        pad_rows, pad_cols = max(0, pad_rows), max(0, pad_cols)

        grad_expanded = array_ops.transpose(
            array_ops.reshape(grad, (batch_size, rows_out,
                                     cols_out, ksize_r, ksize_c, channels)),
            (1, 2, 3, 4, 0, 5)
        )
        grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))

        row_steps = range(0, rows_out * stride_r, stride_r)
        col_steps = range(0, cols_out * stride_h, stride_h)

        idx = []
        for i in range(rows_out):
            for j in range(cols_out):
                r_low, c_low = row_steps[i] - pad_rows, col_steps[j] - pad_cols
                r_high, c_high = r_low + ksize_r_eff, c_low + ksize_c_eff

                idx.extend([(r * (cols_in) + c,
                   i * (cols_out * ksize_r * ksize_c) +
                   j * (ksize_r * ksize_c) +
                   ri * (ksize_c) + ci)
                  for (ri, r) in enumerate(range(r_low, r_high, rate_r))
                  for (ci, c) in enumerate(range(c_low, c_high, rate_c))
                  if 0 <= r and r < rows_in and 0 <= c and c < cols_in
                ])

        sp_shape = (rows_in * cols_in,
              rows_out * cols_out * ksize_r * ksize_c)

        sp_mat = sparse_tensor.SparseTensor(
            array_ops.constant(idx, dtype=ops.dtypes.int64),
            array_ops.ones((len(idx),), dtype=ops.dtypes.float32),
            sp_shape
        )

        jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)

        grad_out = array_ops.reshape(
            jac, (rows_in, cols_in, batch_size, channels)
        )
        grad_out = array_ops.transpose(grad_out, (2, 0, 1, 3))
        
        return grad_out
