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

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default='InvertedPendulum-v2')
parser.add_argument("--max_iter", default=200)
parser.add_argument("--hidden_dim", default=3, type=int)
parser.add_argument("--mode", default="train")
args = parser.parse_args()


class DeepKoopman():
    def __init__(self, env_name = "Pendulum-v0", hidden_dim = 2):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]+1
        self.hidden_dim = hidden_dim
        self.action_dim = self.env.action_space.shape[0]
        self.encoder = nn.Sequential(nn.Linear(self.state_dim, 16),
                                      nn.PReLU(),
                                      nn.Linear(16, 16),
                                      nn.PReLU(),
                                      nn.Linear(16, hidden_dim))
        
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, 16),
                                      nn.PReLU(),
                                      nn.Linear(16, 16),
                                      nn.PReLU(),
                                      nn.Linear(16, self.state_dim))
        
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
        file_name = "weights/" + self.env_name + ".pt"
        torch.save({"encoder" : self.encoder.state_dict(),
                    "decoder" : self.decoder.state_dict(),
                    "propagate" : self.propagate.state_dict()}, file_name)
        print("save model to " + file_name)
    
    def load(self):
        try:
            if not os.path.exists("weights/"):
                os.mkdir("weights/")
            file_name = "weights/" + self.env_name + ".pt"
            checkpoint = torch.load(file_name)
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])
            self.propagate.load_state_dict(checkpoint["propagate"])
            print("load model from " + file_name)
        except:
            print("fail to load model!")

    def transform_state(self, x):
        return np.array([x[1], np.sin(x[1]), np.cos(x[1]), x[2], x[3]])

    def policy_rollout(self):
        A, B = self.get_system()
        Q = np.eye(self.hidden_dim)
        R = np.array([[0.01]])

        K, _, _ = control.lqr(A, B, Q, R)

        ref = torch.FloatTensor([[0.0, 0.0, 1.0, 0., 0.]])
        ref = model.encoder(ref).detach().numpy()

        obs_old = self.transform_state(self.env.reset())
        #obs_old[2] = obs_old[2] / 8.0
        for _ in range(200):
            if np.random.random() > 0.05:
                state = torch.FloatTensor(obs_old.reshape((1, -1)))
                y = model.encoder(state).detach().numpy()
                action = -np.dot(K, (y-ref).T)
                action = np.clip(np.array([action.item()]), -1., 1.)
            else:
                action = self.env.action_space.sample()
            #self.env.render()
            obs, reward, done, info = self.env.step(action)
            #obs[2] = obs[2] / 8.0
            obs = self.transform_state(obs)
            self.replay_buffer.push((obs_old, action, obs))
            obs_old = obs

    def random_rollout(self):
        obs_old = self.transform_state(self.env.reset())
        #obs_old[2] = obs_old[2] / 8.
        for _ in range(200):
            action = self.env.action_space.sample()
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
                #reg_loss = torch.norm(self.propagate.weight.data[:, self.hidden_dim:])
                total_loss = ae_loss + self.lambda1*pred_loss + self.lambda2*metric_loss
                
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                propagate_optimizer.zero_grad()
                
                total_loss.backward()
                
                encoder_optimizer.step()
                decoder_optimizer.step()
                propagate_optimizer.step()
                loss_hist.append(total_loss.detach().numpy())
            print("epoch: %d, loss: %2.5f" % (it, np.mean(loss_hist)))
            for i in range(5):
                self.policy_rollout()
            for i in range(5):
                self.random_rollout()

if __name__ == "__main__":
    model = DeepKoopman(args.env_name, args.hidden_dim)

    if args.mode == "train":
        model.train(args.max_iter, 0.001)
        model.save()
    else:
        model.load()
        A, B = model.get_system()
        Q = np.eye(args.hidden_dim)
        R = np.array([[0.08]])
        K, _, _ = control.lqr(A, B, Q, R)
        print(A)
        print(B)
        print(K)

        env = gym.make(args.env_name)

        ref = torch.FloatTensor([[0.0, 0.0, 1.0, 0., 0.]])
        ref = model.encoder(ref).detach().numpy()
        offset = [0.1, 0.2, 0.3, 0.4, 0.5]
        for k in range(5):
            state = env.reset()
            state[1] = offset[k]
            env.env.set_state(state[:2], state[:2])
            state = model.transform_state(state)
            for i in range(200):
                env.render()
                state = torch.FloatTensor(state.reshape((1, -1)))
                #state[0, 2] = state[0, 2] / 8.0
                y = model.encoder(state).detach().numpy()
                action = -np.dot(K, (y-ref).T)
                state, reward, done, info = env.step(action)
                #print(state)
                state = model.transform_state(state)
        env.close()

