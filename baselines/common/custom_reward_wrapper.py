import gym
import numpy as np
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd
from baselines.common.trex_utils import preprocess
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

#Ibarz network
class AtariNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)


    def forward(self, traj):
        '''calculate cumulative return of trajectory'''
        x = traj.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = torch.sigmoid(self.fc2(x)) #clip reward?
        #r = self.fc2(x) #clip reward?
        return r
# class AtariNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.fc1 = nn.Linear(64 * 7 * 7, 512)
#         self.output = nn.Linear(512, 1)
#
#     def forward(self, traj):
#         '''calculate cumulative return of trajectory'''
#         x = traj.permute(0,3,1,2) #get into NCHW format
#         #compute forward pass of reward network
#         conv1_output = F.relu(self.conv1(x))
#         conv2_output = F.relu(self.conv2(conv1_output))
#         conv3_output = F.relu(self.conv3(conv2_output))
#         fc1_output = F.relu(self.fc1(conv3_output.view(conv3_output.size(0),-1)))
#         r = self.output(fc1_output)
#         return r


# self attention
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    #print('Attention')
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def positionalEncoding2D(input_tensor):
    #print(input_tensor.shape)
    batch_size = input_tensor.size()[0]
    # Attach 2D position layers to input tensor 
    kernel_w = input_tensor.size()[2]
    kernel_h = input_tensor.size()[3]       
    position_x = torch.arange(0., kernel_w).unsqueeze(0).cuda()
    position_y = torch.arange(0., kernel_h).unsqueeze(0).cuda()
    pe_x = torch.t(position_x.repeat(kernel_h,1).view(kernel_h,kernel_w)).unsqueeze(0)
    pe_y = position_y.repeat(1,kernel_w).view(kernel_w,kernel_h).unsqueeze(0)
    #print(pe_x.shape,pe_y.shape)
    att = torch.cat([pe_x, pe_y],0).unsqueeze(0)
    #print(att.shape)
    att = att.repeat(batch_size,1,1,1)
    #print(att.shape)
    
    out_tensor = torch.cat([input_tensor, att],1)
    #print( out_tensor.shape)
    return out_tensor

def flattenTensor(input_tensor):
    t_size = input_tensor.shape
    flat_input = torch.t(input_tensor.view(t_size[0], t_size[1]*t_size[2]))
    return flat_input


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))   



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h #h = 8; d_model=18
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        #print('MHA')
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # print(query.shape, key.shape, value.shape)
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class FullyConnected(nn.Module):
    def __init__(self, nc=128, ndf=64, num_actions=1):
        super(FullyConnected, self).__init__()
        self.ndf = ndf
        self.max_pool = nn.MaxPool1d(ndf, return_indices=False)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Linear(49, 64),
            # F.leaky_relu(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)
        )


    def forward(self, input_state):
        #print('--> Action Prediction')
        #print('input:',input_state.shape)
        #import pdb;pdb.set_trace()
        #features = self.max_pool(input_state).view(input_state.shape[0],-1)
        features = self.max_pool(input_state).view(-1, 49)
        #print('features:',features.shape)
        output = self.main(features)
        #print('output:',output.shape)
        return output #.view(-1, 1).squeeze(1)

class ImageEncoder(nn.Module):
    "Process input RGB image (128x128)"
    def __init__(self, size, self_attn, feed_forward, dropout, nc=3, ndf=256, hn=30):
        super(ImageEncoder, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size 
        
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(4, 16, 7, stride=3),
            # F.leaky_relu(),
            nn.LeakyReLU(0.2, inplace=True),                        
            nn.Conv2d(16, 16, 5, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            # F.leaky_relu(),       
            nn.Conv2d(16, 16, 3, stride=1),
            # nn.LeakyReLU(0.2, inplace=True),  
            # F.leaky_relu(),  
            #nn.Conv2d(ndf , ndf, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),         
            # nn.Conv2d(ndf, hn, 4, stride=2, padding=1, bias=False),
            nn.Conv2d(16, 16, 3, stride=1),
            # nn.BatchNorm2d(hn),
            nn.LeakyReLU(0.2, inplace=True)
            # F.leaky_relu()
        )
   

    def forward(self, x):
        #import pdb;pdb.set_trace()
        images = self.main(x)
        # print('images:',images.shape)
        pencoded_is = positionalEncoding2D(images) 
        # print('pencoded_is:',pencoded_is.shape)
        flat_i = torch.flatten(pencoded_is,start_dim=2).permute(0,2,1)
        # print('flat_i:',flat_i.shape)
        x = self.sublayer[0](flat_i, lambda flat_i: self.self_attn(flat_i, flat_i, flat_i))
        output =  self.sublayer[1](x.squeeze(-1), self.feed_forward)
        return output


class AtariNetSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        d_model=32 #? or 49?
        d_ff=256
        num_heads = 9#8
        embedding_channels = 18#32
        dropout=0.1
        nc=128
        ndf=64
        hn=30
        num_actions=1

        c = copy.deepcopy
        self.attn = MultiHeadedAttention(num_heads, embedding_channels)
        self.ff = PositionwiseFeedForward(embedding_channels, d_ff, dropout)

        self.model = nn.Sequential(ImageEncoder(embedding_channels, c(self.attn), c(self.ff), dropout, hn=hn), FullyConnected(nc=nc, ndf=ndf, num_actions=num_actions)).to(device)
        # self.model.apply(weights_init)
        #print(model)
    

    def forward(self, traj):
        '''calculate cumulative return of trajectory'''

        x = traj.permute(0,3,1,2) #get into NCHW format
        x = self.model(x)
        r = torch.sigmoid(x)
        return r




class VecRLplusIRLAtariReward(VecEnvWrapper):
    def __init__(self, venv, reward_net_path, combo_param):
        VecEnvWrapper.__init__(self, venv)
        self.reward_net = AtariNetSelfAttention()
        self.reward_net.load_state_dict(torch.load(reward_net_path))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)

        self.lamda = combo_param #how much weight to give to IRL verus RL combo_param \in [0,1] with 0 being RL and 1 being IRL
        self.rew_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.cliprew = 10.

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
##Testing network to see why always giving zero rewards....
        #import pickle
        #filename = 'rand_obs.pkl'
        #infile = open(filename,'rb')
        #rand_obs = pickle.load(infile)
        #infile.close()
        #traj = [obs / 255.0] #normalize!
        #import matplotlib.pyplot as plt
        #plt.figure(1)
        #plt.imshow(obs[0,:,:,0])
        #plt.figure(2)
        #plt.imshow(rand_obs[0,:,:,0])
        #plt.show()
        #print(obs.shape)
        with torch.no_grad():
            rews_network = self.reward_net.cum_return(torch.from_numpy(np.array(obs)).float().to(self.device)).cpu().numpy().transpose()[0]
            #rews2= self.reward_net.cum_return(torch.from_numpy(np.array([rand_obs])).float().to(self.device)).cpu().numpy().transpose()[0]
        #self.rew_rms.update(rews_network)
        #r_hat = rews_network
        #r_hat = np.clip((r_hat - self.rew_rms.mean) / np.sqrt(self.rew_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        #print(rews1)
        #   print(rews2)

        #print(obs.shape)
        # obs shape: [num_env,84,84,4] in case of atari games

        #combine IRL and RL rewards using lambda parameter like Yuke Zhu's paper "Reinforcement and Imitation Learningfor Diverse Visuomotor Skills"
        reward_combo = self.lamda * rews_network + (1-self.lamda) * rews

        return obs, reward_combo , news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs


class VecPyTorchAtariReward(VecEnvWrapper):
    def __init__(self, venv, reward_net_path, env_name, self_attn):
        VecEnvWrapper.__init__(self, venv)
        print('self attention: '+str(self_attn))
        if self_attn:
            self.reward_net = AtariNetSelfAttention()
        else:
            self.reward_net = AtariNet()
        self.reward_net.load_state_dict(torch.load(reward_net_path))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)

        self.rew_rms = RunningMeanStd(shape=())
        self.epsilon = 1e-8
        self.cliprew = 10.
        self.env_name = env_name

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        # obs shape: [num_env,84,84,4] in case of atari games
        #plt.subplot(1,2,1)
        #plt.imshow(obs[0][:,:,0])
        #crop off top of image
        #n = 10
        #no_score_obs = copy.deepcopy(obs)
        #obs[:,:n,:,:] = 0

        #Need to normalize for my reward function
        #normed_obs = obs / 255.0
        #mask and normalize for input to network
        normed_obs = preprocess(obs, self.env_name)
        #plt.subplot(1,2,2)
        #plt.imshow(normed_obs[0][:,:,0])
        #plt.show()
        #print(traj[0][0][40:60,:,:])

        with torch.no_grad():
            rews_network = self.reward_net.forward(torch.from_numpy(np.array(normed_obs)).float().to(self.device)).cpu().numpy().squeeze()

        return obs, rews_network, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs



class VecLiveLongReward(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        rews = np.ones_like(rews)

        #print(obs.shape)
        # obs shape: [num_env,84,84,4] in case of atari games

        return obs, rews, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs


import tensorflow as tf
class VecTFRandomReward(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)

        self.graph = tf.Graph()

        config = tf.ConfigProto(
            device_count = {'GPU': 0}) # Run on CPU
        #config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                self.obs = tf.placeholder(tf.float32,[None,84,84,4])

                self.rewards = tf.reduce_mean(
                    tf.random_normal(tf.shape(self.obs)),axis=[1,2,3])


    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        rews = self.sess.run(self.rewards,feed_dict={self.obs:obs})

        return obs, rews, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        ##############
        # If the reward is based on LSTM or something, then please reset internal state here.
        ##############

        return obs

class VecTFPreferenceReward(VecEnvWrapper):
    def __init__(self, venv, num_models, model_dir):
        VecEnvWrapper.__init__(self, venv)

        self.graph = tf.Graph()

        config = tf.ConfigProto(
            device_count = {'GPU': 0}) # Run on CPU
        #config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                import os, sys
                dir_path = os.path.dirname(os.path.realpath(__file__))
                sys.path.append(os.path.join(dir_path,'..','..','..','..'))
                from preference_learning import Model

                print(os.path.realpath(model_dir))

                self.models = []
                for i in range(num_models):
                    with tf.variable_scope('model_%d'%i):
                        model = Model(self.venv.observation_space.shape[0])
                        model.saver.restore(self.sess,model_dir+'/model_%d.ckpt'%(i))
                    self.models.append(model)

        """
        try:
            self.save = venv.save
            self.load = venv.load
        except AttributeError:
            pass
        """

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()

        with self.graph.as_default():
            with self.sess.as_default():
                r_hat = np.zeros_like(rews)
                for model in self.models:
                    r_hat += model.get_reward(obs)

        rews = r_hat / len(self.models)

        return obs, rews, news, infos

    def reset(self, **kwargs):
        obs = self.venv.reset()

        return obs

if __name__ == "__main__":
    pass
