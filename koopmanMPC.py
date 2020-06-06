import numpy as np
import torch
import torch.nn as nn
import gym
from torch.utils.data import Dataset, DataLoader
import control
import os
from ReplayBuffer import ReplayBuffer
import time
import argparse

from mpc import mpc
from mpc.mpc import QuadCost, LinDx
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default='Pendulum-v0')
parser.add_argument("--max_iter", default=200)
parser.add_argument("--hidden_dim", default=3, type=int)
parser.add_argument("--mode", default="train")
args = parser.parse_args()

def construct_MPC(A, B, ref, dt):
    n_batch, n_state, n_ctrl, T = 1, args.hidden_dim, 1, 5
    n_sc = n_state + n_ctrl

    goal_weights = torch.ones(args.hidden_dim)

    ctrl_penalty = 0.1 * torch.ones(n_ctrl)
    q = torch.cat((goal_weights, ctrl_penalty))
    px = -torch.sqrt(goal_weights) * ref
    p = torch.cat((px[0], torch.zeros(n_ctrl)))
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(T, n_batch, 1, 1)
    p = p.unsqueeze(0).repeat(T, n_batch, 1)

    F = torch.FloatTensor(np.concatenate([np.eye(args.hidden_dim) + dt*A, dt * B], axis = 1))
    F = F.unsqueeze(0).unsqueeze(0).repeat(T, n_batch, 1, 1)
    f = torch.zeros([5, 1, 3])

    u_lower = -torch.ones(T, n_batch, n_ctrl) *2
    u_upper = torch.ones(T, n_batch, n_ctrl) * 2

    cost = QuadCost(Q, p)
    dynamic = LinDx(F)

    mpc_model = mpc.MPC(
        n_state = n_state,
        n_ctrl = n_ctrl,
        n_batch = n_batch,
        backprop = False,
        T=T,
        u_lower = u_lower, 
        u_upper = u_upper,
        lqr_iter = 10,
        verbose = 0,
        exit_unconverged=False,
        eps=1e-2,)
    return mpc_model, cost, dynamic


class DeepKoopman():
    def __init__(self, env_name = "Pendulum-v0", hidden_dim = 2):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.env.max_torque = 5.0
        self.env.action_space.high = np.array([5.0])
        self.env.action_space.low = np.array([-5.0])
        self.state_dim = self.env.observation_space.shape[0]
        self.hidden_dim = hidden_dim
        self.action_dim = self.env.action_space.shape[0]
        N= 32
        self.encoder = nn.Sequential(nn.Linear(self.state_dim, N),
                                      nn.PReLU(),
                                      nn.Linear(N, N),
                                      nn.PReLU(),
                                      nn.Linear(N, hidden_dim),
                                      nn.Tanh())
        
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, N),
                                      nn.PReLU(),
                                      nn.Linear(N, N),
                                      nn.PReLU(),
                                      nn.Linear(N, self.state_dim),
                                      nn.Tanh())
        
        self.propagate = nn.Linear(hidden_dim+self.action_dim, hidden_dim, bias = False)
        
        self.lambda1 = 1.0
        self.lambda2 = 0.3
        
        self.replay_buffer = ReplayBuffer(100000)
    
    def get_system(self):
        weight = self.propagate.weight.data.numpy()
        A = weight[:, :self.hidden_dim]
        B = weight[:, self.hidden_dim:]
        return A, B
    
    def forward(self, xt, ut):
        gt = self.encoder(xt)
        xt_ = self.decoder(gt)
        gtdot = self.propagate(torch.cat((gt, ut), axis = -1))
        gt1 = gt + self.env.env.dt*gtdot
        xt1_ = self.decoder(gt1)
        return gt, gt1, xt_, xt1_
    
    def save(self):
        if not os.path.exists("weights/"):
            os.mkdir("weights/")
        file_name = "weights/" + "MPC_" + self.env_name + ".pt"
        torch.save({"encoder" : self.encoder.state_dict(),
                    "decoder" : self.decoder.state_dict(),
                    "propagate" : self.propagate.state_dict()}, file_name)
        print("save model to " + file_name)
    
    def load(self):
        try:
            if not os.path.exists("weights/"):
                os.mkdir("weights/")
            file_name = "weights/" + "MPC_" + self.env_name + ".pt"
            checkpoint = torch.load(file_name)
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])
            self.propagate.load_state_dict(checkpoint["propagate"])
            print("load model from " + file_name)
        except:
            print("fail to load model!")

    def transform_state(self, x):
        return np.array([x[0], x[1], x[2]/8.0])

    def policy_rollout(self):
        A, B = self.get_system()
        Q = np.eye(self.hidden_dim)
        R = np.array([[0.01]])

        K, _, _ = control.lqr(A, B, Q, R)

        ref = torch.FloatTensor([[1.0, 0., 0.]])
        ref = model.encoder(ref).detach().numpy()

        obs_old = self.transform_state(self.env.reset())
        #obs_old[2] = obs_old[2] / 8.0
        for _ in range(100):
            if np.random.random() > 0.5:
                state = torch.FloatTensor(obs_old.reshape((1, -1)))
                y = model.encoder(state).detach().numpy()
                action = -np.dot(K, (y-ref).T)
                action = np.clip(np.array([action.item()]), self.env.action_space.low, self.env.action_space.high)
            else:
                action = self.env.action_space.sample() * 0.4
            #self.env.render()
            obs, reward, done, info = self.env.step(action)
            #obs[2] = obs[2] / 8.0
            obs = self.transform_state(obs)
            self.replay_buffer.push((obs_old, action / 2.0, obs))
            obs_old = obs

    # def policy_rollout(self):
    #     A, B = self.get_system()
    #     ref = torch.FloatTensor([[1.0, 0., 0.]])
    #     ref = model.encoder(ref).detach()

    #     obs_old = self.transform_state(self.env.reset())
    #     #obs_old[2] = obs_old[2] / 8.0
    #     for k in range(100):
    #         if np.random.random() > 0.05:
    #             state = torch.FloatTensor(obs_old.reshape((1, -1)))
    #             y = model.encoder(state).detach()
    #             mpc_model, cost, dynamic = construct_MPC(A, B, ref, self.env.env.dt)
    #             x_lqr, u_lqr, objs_lqr = mpc_model(y, cost, dynamic)
    #             action = u_lqr[0, 0].detach().numpy()
    #             # import IPython
    #             # IPython.embed()
    #             #action = np.clip(np.array([action.item()]), self.env.action_space.low, self.env.action_space.high)
    #         else:
    #             action = self.env.action_space.sample()
    #         #self.env.render()
    #         obs, reward, done, info = self.env.step(action)
    #         #obs[2] = obs[2] / 8.0
    #         obs = self.transform_state(obs)
    #         self.replay_buffer.push((obs_old, action, obs))
    #         obs_old = obs

    def random_rollout(self):
        obs_old = self.transform_state(self.env.reset())
        #obs_old[2] = obs_old[2] / 8.
        for _ in range(100):
            action = self.env.action_space.sample() * 0.4
            obs, reward, done, info = self.env.step(action)
            obs = self.transform_state(obs)
            #obs[2] = obs[2] / 8.0
            self.replay_buffer.push((obs_old, action, obs))
            obs_old = obs
    
    def train(self, max_iter, lr =0.001):
        mseloss = nn.MSELoss()
        l1loss = nn.L1Loss()
        
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = lr)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr = lr)
        propagate_optimizer = torch.optim.Adam(self.propagate.parameters(), lr = lr)

        for i in range(20):
            self.random_rollout()

        for it in range(max_iter):
            loss_hist = []
            for _ in range(100):
                xt, ut, xt1 = self.replay_buffer.sample(64)

                xt = torch.FloatTensor(xt)
                ut = torch.FloatTensor(ut)
                xt1 = torch.FloatTensor(xt1)

                gt, gt1, xt_, xt1_ = self.forward(xt, ut)

                ae_loss = mseloss(xt_, xt)
                pred_loss = mseloss(xt1_, xt1)
                metric_loss = l1loss(torch.norm(gt1-gt, dim=1), torch.norm(xt1-xt, dim=1))
                
                ref = torch.FloatTensor([[1.0, 0., 0.], [-1., 0., 0.]])
                ref = model.encoder(ref).detach()
                ref = torch.cat((ref, torch.zeros((ref.shape[0], 1))), axis = 1)
                fix_loss = mseloss(self.propagate(ref), torch.zeros((ref.shape[0], args.hidden_dim)))

                #reg_loss = torch.norm(self.propagate.weight.data[:, self.hidden_dim:])
                total_loss = ae_loss + self.lambda1*pred_loss + self.lambda2*metric_loss + fix_loss *1.0
                
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                propagate_optimizer.zero_grad()
                
                total_loss.backward()
                
                encoder_optimizer.step()
                decoder_optimizer.step()
                propagate_optimizer.step()
                loss_hist.append(total_loss.detach().numpy())
            print("epoch: %d, loss: %2.5f" % (it, np.mean(loss_hist)))
            if it % 3 == 0:
                for i in range(5):
                    self.policy_rollout()
                for i in range(5):
                    self.random_rollout()
            
class PendulumDynamics(torch.nn.Module):
    def __init__(self, A, B):
        super(PendulumDynamics, self).__init__()
        self.A = torch.FloatTensor(A)
        self.B = torch.FloatTensor(B)

    def forward(self, state, action):
        dx = torch.mm(self.A, state.T) + torch.mm(self.B, action)
        state = state + 0.04 * dx.T
        return state

if __name__ == "__main__":
    model = DeepKoopman(args.env_name, args.hidden_dim)

    if args.mode == "train":
        model.train(args.max_iter, 0.001)
        model.save()
    else:
        model.load()
        A, B = model.get_system()
        Q = np.eye(args.hidden_dim)
        R = np.array([[1]])
        K, _, _ = control.lqr(A, B, Q, R)
        print(K)
        env = gym.make(args.env_name)
        env.env.max_torque = 5.0
        env.action_space.high = np.array([5.0])
        env.action_space.low = np.array([-5.0])

        import IPython
        IPython.embed()
        
        n_batch, n_state, n_ctrl, T = 1, args.hidden_dim, 1, 10
        n_sc = n_state + n_ctrl

        goal_weights = torch.ones(args.hidden_dim)

        ref = torch.FloatTensor([[1.0, 0.0, 0.0]])
        ref = model.encoder(ref).detach()

        ctrl_penalty = 0.01 * torch.ones(n_ctrl)
        q = torch.cat((goal_weights, ctrl_penalty))
        px = -torch.sqrt(goal_weights) * ref
        p = torch.cat((px, torch.zeros((n_batch, n_ctrl))), dim = 1)
        Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(T, n_batch, 1, 1)
        p = p.unsqueeze(0).repeat(T, 1, 1)

        
        F = torch.FloatTensor(np.concatenate([np.eye(args.hidden_dim) + env.env.dt*A, env.env.dt * B], axis = 1))
        F = F.unsqueeze(0).unsqueeze(0).repeat(T, n_batch, 1, 1)
        f = torch.zeros([5, 1, 3])

        u_lower = -torch.ones(T, n_batch, n_ctrl) *2
        u_upper = torch.ones(T, n_batch, n_ctrl) * 2

        cost = QuadCost(Q, p)
        dynamic = LinDx(F)

        u_init = None
        for k in range(5):
            state = env.reset()
            state = model.transform_state(state)
            for i in range(100):
                env.render()
                state = torch.FloatTensor(state.copy().reshape((1, -1)))
                y = model.encoder(state).detach()
                
                act = -np.dot(K, (y-ref).T)
                if i % 5 == 0:
                    mpc_model = mpc.MPC(
                        n_state = n_state,
                        n_ctrl = n_ctrl,
                        n_batch = n_batch,
                        backprop = False,
                        T=T,
                        u_lower = u_lower, 
                        u_upper = u_upper,
                        lqr_iter = 10,
                        verbose = 0,
                        #u_init=None,
                        #grad_method=mpc.GradMethods.AUTO_DIFF,
                        exit_unconverged=False,
                        eps=1e-2,)
                    x_lqr, u_lqr, objs_lqr = mpc_model(y, QuadCost(Q, p), LinDx(F))
                    action = u_lqr.detach().numpy()
                #u_init = torch.cat((u_lqr[1:], torch.zeros(1, n_batch, n_ctrl)), dim=0)

                # print(action)
                
                
                state, reward, done, info = env.step(action[i%5])
                

                #print(state)
                state = model.transform_state(state)

        env.close()

