# This code is adapted from Karras et al. (EDM, https://github.com/NVlabs/edm/blob/main/training/loss.py).
# The original code is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models" and
"Label-Noise Robust Diffusion Models"."""

import torch
import numpy as np
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Classifier Loss function corresponding to the variance preserving (VP) formulation

@persistence.persistent_class
class VPClassifierLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        logit = net(y + n, sigma, labels, augment_labels=augment_labels)
        log_prob_class = torch.log(0.4 / 9 + (1 - 4 / 9) * torch.nn.Softmax(dim=1)(logit))
        loss = weight * torch.nn.NLLLoss(reduce=False)(log_prob_class, torch.argmax(labels, dim=1))
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Classifier Loss function corresponding to the Improved loss function proposed in
# the paper "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMClassifierLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, noise_type='none', noise_rate=0., num_classes=10, dataset='cifar10'):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.dataset = dataset

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        logit = net(y + n, sigma, None, augment_labels=augment_labels)

        if self.noise_type == 'sym':
            log_prob_class = torch.log(self.noise_rate / (self.num_classes - 1) + (1 - self.num_classes * self.noise_rate / (self.num_classes - 1)) * torch.nn.Softmax(dim=1)(logit))
        elif self.noise_type == 'asym':
            if self.dataset == 'mnist':
                tm = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1. - self.noise_rate, 0, 0, 0, 0, self.noise_rate, 0, 0],
                      [0, 0, 0, 1. - self.noise_rate, 0, 0, 0, 0, self.noise_rate, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1. - self.noise_rate, self.noise_rate, 0, 0, 0],
                      [0, 0, 0, 0, 0, self.noise_rate, 1. - self.noise_rate, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
            elif self.dataset == 'cifar10':
                # 9->1, 2->0, 4->7, 3 <->5
                tm = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [self.noise_rate, 0, 1. - self.noise_rate, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1. - self.noise_rate, 0, self.noise_rate, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1 - self.noise_rate, 0, 0, self.noise_rate, 0, 0],
                      [0, 0, 0, self.noise_rate, 0, 1. - self.noise_rate, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1., 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, self.noise_rate, 0, 0, 0, 0, 0, 0, 0, 1 - self.noise_rate]]
            elif self.dataset == 'cifar100':
                tm = np.eye(self.num_classes)
                noise_label_lst = np.array([51, 32, 11, 42, 30, 20,  7, 14, 13, 10, 16, 35, 17, 48, 18, 19, 28,
                                            37, 24, 21, 25, 31, 39, 33,  6, 84, 45, 29, 61, 44, 55, 38, 67, 49,
                                            63, 46, 50, 68, 15, 40, 86, 69, 43, 88, 78, 77, 98, 52, 58, 60, 65,
                                            53, 56, 57, 62, 72, 59, 83, 90, 96, 71,  9, 70, 64, 66, 74, 75, 73,
                                            76, 81, 82, 23, 95, 91, 80, 34, 12, 79, 93, 99, 36, 85, 92,  0, 94,
                                            89, 87, 22, 97, 41,  8,  1, 54, 27,  5,  4, 47,  3,  2, 26])
                for i in range(self.num_classes):
                    tm[i, i] = 1 - self.noise_rate
                    tm[i, noise_label_lst[i]] = self.noise_rate
            else:
                raise NotImplementedError
            tm = torch.FloatTensor(tm).to(logit.device)
            log_prob_class = torch.log(torch.matmul(tm.transpose(1, 0).unsqueeze(0), torch.nn.Softmax(dim=1)(logit).unsqueeze(-1)).squeeze(-1))
        else:
            raise NotImplementedError

        loss = torch.nn.NLLLoss(reduce=False)(log_prob_class, torch.argmax(labels, dim=1))
        return loss

#----------------------------------------------------------------------------
# Classifier Loss function with VolMinNet, corresponding to the Improved loss function proposed in
# the paper "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMClassifierWithVMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, noise_type='none', noise_rate=0., num_classes=10, dataset='cifar10'):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.dataset = dataset

    def __call__(self, net, images, labels=None, augment_pipe=None):
        divide_factor = 32
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        logit, transition_matrix = net(y + n, sigma, None, augment_labels=augment_labels)
        probs = torch.softmax(logit, dim=1)
        log_prob_class = torch.log(torch.mm(probs, transition_matrix) + 1e-12)
        loss = torch.nn.NLLLoss(reduce=False)(log_prob_class, torch.argmax(labels, dim=1))

        # Volume Minimization Regularization
        loss_vm = torch.logdet(transition_matrix)

        loss = loss + 0.0001 * loss_vm

        return loss

#----------------------------------------------------------------------------
# TDSM Loss function corresponding to the Improved loss function proposed in
# the paper "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMNoiseLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, noise_type='clean', noise_rate=0., num_classes=10, dataset='cifar10', tau=1e-2, temp=1.):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.dataset = dataset
        self.sm = torch.nn.Softmax(dim=1)
        self.tau = tau
        self.temp = temp

    def __call__(self, net, images, labels=None, augment_pipe=None, cls_net=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma

        with torch.no_grad():
            if self.noise_type == 'learn':
                logit, tm = cls_net(y + n, sigma, None)
                logit = logit / self.temp
                noisy_prob_class = torch.matmul(tm.transpose(1, 0).unsqueeze(0), self.sm(logit).unsqueeze(-1)).squeeze(-1)
            elif self.noise_type == 'sym':
                noisy_prob_class = self.noise_rate / (self.num_classes - 1) + (1 - self.num_classes * self.noise_rate / (self.num_classes - 1)) * self.sm(cls_net(y + n, sigma, None) / self.temp)
            elif self.noise_type == 'asym':
                if self.dataset == 'mnist':
                    tm = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1. - self.noise_rate, 0, 0, 0, 0, self.noise_rate, 0, 0],
                          [0, 0, 0, 1. - self.noise_rate, 0, 0, 0, 0, self.noise_rate, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1. - self.noise_rate, self.noise_rate, 0, 0, 0],
                          [0, 0, 0, 0, 0, self.noise_rate, 1. - self.noise_rate, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
                elif self.dataset == 'cifar10':
                    tm = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [self.noise_rate, 0, 1. - self.noise_rate, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1. - self.noise_rate, 0, self.noise_rate, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1 - self.noise_rate, 0, 0, self.noise_rate, 0, 0],
                          [0, 0, 0, self.noise_rate, 0, 1. - self.noise_rate, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1., 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, self.noise_rate, 0, 0, 0, 0, 0, 0, 0, 1 - self.noise_rate]]
                elif self.dataset == 'cifar100':
                    tm = np.eye(self.num_classes)
                    noise_label_lst = np.array([51, 32, 11, 42, 30, 20,  7, 14, 13, 10, 16, 35, 17, 48, 18, 19, 28,
                                                37, 24, 21, 25, 31, 39, 33,  6, 84, 45, 29, 61, 44, 55, 38, 67, 49,
                                                63, 46, 50, 68, 15, 40, 86, 69, 43, 88, 78, 77, 98, 52, 58, 60, 65,
                                                53, 56, 57, 62, 72, 59, 83, 90, 96, 71,  9, 70, 64, 66, 74, 75, 73,
                                                76, 81, 82, 23, 95, 91, 80, 34, 12, 79, 93, 99, 36, 85, 92,  0, 94,
                                                89, 87, 22, 97, 41,  8,  1, 54, 27,  5,  4, 47,  3,  2, 26])
                    for i in range(self.num_classes):
                        tm[i, i] = 1 - self.noise_rate
                        tm[i, noise_label_lst[i]] = self.noise_rate
                else:
                    raise NotImplementedError
                tm = torch.FloatTensor(tm).to(images.device)
                noisy_prob_class = torch.matmul(tm.transpose(1, 0).unsqueeze(0), self.sm(cls_net(y + n, sigma, None) / self.temp).unsqueeze(-1)).squeeze(-1)
            else:
                raise NotImplementedError

        if self.noise_type == 'learn':
            prior = tm / (tm.sum(dim=0, keepdims=True))
            prior_t = (1 / self.num_classes) * (tm.sum(dim=0, keepdims=True))
        elif self.noise_type == 'sym':
            prior = (1 - self.noise_rate) * torch.eye(self.num_classes) + self.noise_rate / (self.num_classes - 1) * (1 - torch.eye(self.num_classes))
            prior = torch.FloatTensor(prior).to(images.device)
            prior_t = (1 / self.num_classes) * (prior.sum(dim=0, keepdims=True))
        elif self.noise_type == 'asym':
            prior = tm / (tm.sum(dim=0, keepdims=True))
            prior_t = (1 / self.num_classes) * (tm.sum(dim=0, keepdims=True))
        else:
            raise NotImplementedError

        prior = prior.transpose(1, 0)
        prior_inverse = torch.inverse(prior)
        class_labels_idx = torch.argmax(labels, dim=1)

        p_y_bar_x_yt = torch.matmul(noisy_prob_class / prior_t, prior_inverse.transpose(1, 0))
        prior_batch = prior.unsqueeze(0).repeat(len(noisy_prob_class), 1, 1)
        p_y_cond_x = prior_batch / (noisy_prob_class / prior_t).unsqueeze(2)
        w_x = p_y_cond_x * p_y_bar_x_yt.unsqueeze(1)
        w_x_yt = w_x[torch.arange(len(noisy_prob_class)), class_labels_idx]

        w_x_yt = torch.clamp(w_x_yt, min=0, max=1)
        w_x_yt = w_x_yt / torch.sum(w_x_yt, dim=1, keepdim=True)

        w_x_yt = torch.nan_to_num(w_x_yt, 0, 0, 0)

        D_yn_train = net(y + n, sigma, labels, augment_labels=augment_labels)
        D_yn_train = D_yn_train * w_x_yt[torch.arange(labels.shape[0]), class_labels_idx][:, None, None, None]
        with torch.no_grad():
            for c in range(labels.shape[1]):
                labels_temp = torch.nn.functional.one_hot(c * torch.ones(labels.shape[0], dtype=int).to(images.device), labels.shape[1])
                candidate = (class_labels_idx != c) & (w_x_yt[:, c] > self.tau)
                if candidate.sum() != 0:
                    D_yn = net((y + n)[candidate], sigma[candidate], labels_temp[candidate], augment_labels=augment_labels[candidate])
                    D_yn_train[candidate] = D_yn_train[candidate] + D_yn * w_x_yt[candidate, c][:, None, None, None]
        loss = weight * ((D_yn_train - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# TDSM Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPNoiseLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5, noise_type='clean', noise_rate=0., num_classes=10, dataset='cifar10', tau=1e-2):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.dataset = dataset
        self.sm = torch.nn.Softmax(dim=1)
        self.tau = tau

    def __call__(self, net, images, labels=None, augment_pipe=None, cls_net=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma

        with torch.no_grad():
            if self.noise_type == 'learn':
                logit, tm = cls_net(y + n, sigma, None)
                tm = torch.FloatTensor(tm).to(images.device)
                noisy_prob_class = torch.matmul(tm.transpose(1, 0).unsqueeze(0), self.sm(logit).unsqueeze(-1)).squeeze(-1)
            elif self.noise_type == 'sym':
                noisy_prob_class = self.noise_rate / (self.num_classes - 1) + (1 - self.num_classes * self.noise_rate / (self.num_classes - 1)) * self.sm(cls_net(y + n, sigma, None))
            elif self.noise_type == 'asym':
                if self.dataset == 'mnist':
                    tm = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1. - self.noise_rate, 0, 0, 0, 0, self.noise_rate, 0, 0],
                          [0, 0, 0, 1. - self.noise_rate, 0, 0, 0, 0, self.noise_rate, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1. - self.noise_rate, self.noise_rate, 0, 0, 0],
                          [0, 0, 0, 0, 0, self.noise_rate, 1. - self.noise_rate, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
                elif self.dataset == 'cifar10':
                    tm = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [self.noise_rate, 0, 1. - self.noise_rate, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1. - self.noise_rate, 0, self.noise_rate, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1 - self.noise_rate, 0, 0, self.noise_rate, 0, 0],
                          [0, 0, 0, self.noise_rate, 0, 1. - self.noise_rate, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1., 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, self.noise_rate, 0, 0, 0, 0, 0, 0, 0, 1 - self.noise_rate]]
                elif self.dataset == 'cifar100':
                    tm = np.eye(self.num_classes)
                    noise_label_lst = np.array([51, 32, 11, 42, 30, 20,  7, 14, 13, 10, 16, 35, 17, 48, 18, 19, 28,
                                                37, 24, 21, 25, 31, 39, 33,  6, 84, 45, 29, 61, 44, 55, 38, 67, 49,
                                                63, 46, 50, 68, 15, 40, 86, 69, 43, 88, 78, 77, 98, 52, 58, 60, 65,
                                                53, 56, 57, 62, 72, 59, 83, 90, 96, 71,  9, 70, 64, 66, 74, 75, 73,
                                                76, 81, 82, 23, 95, 91, 80, 34, 12, 79, 93, 99, 36, 85, 92,  0, 94,
                                                89, 87, 22, 97, 41,  8,  1, 54, 27,  5,  4, 47,  3,  2, 26])
                    for i in range(self.num_classes):
                        tm[i, i] = 1 - self.noise_rate
                        tm[i, noise_label_lst[i]] = self.noise_rate
                else:
                    raise NotImplementedError
                tm = torch.FloatTensor(tm).to(images.device)
                noisy_prob_class = torch.matmul(tm.transpose(1, 0).unsqueeze(0), self.sm(cls_net(y + n, sigma, None)).unsqueeze(-1)).squeeze(-1)
            else:
                raise NotImplementedError

        if self.noise_type == 'learn':
            prior = tm / (tm.sum(dim=0, keepdims=True))
            prior_t = (1 / self.num_classes) * (tm.sum(dim=0, keepdims=True))
        elif self.noise_type == 'sym':
            prior = (1 - self.noise_rate) * torch.eye(self.num_classes) + self.noise_rate / (self.num_classes - 1) * (1 - torch.eye(self.num_classes))
            prior = torch.FloatTensor(prior).to(images.device)
            prior_t = (1 / self.num_classes) * (prior.sum(dim=0, keepdims=True))
        elif self.noise_type == 'asym':
            prior = tm / (tm.sum(dim=0, keepdims=True))
            prior_t = (1 / self.num_classes) * (tm.sum(dim=0, keepdims=True))
        else:
            raise NotImplementedError

        prior = prior.transpose(1, 0)
        prior_inverse = torch.inverse(prior)
        class_labels_idx = torch.argmax(labels, dim=1)

        p_y_bar_x_yt = torch.matmul(noisy_prob_class / prior_t, prior_inverse.transpose(1, 0))
        prior_batch = prior.unsqueeze(0).repeat(len(noisy_prob_class), 1, 1)
        p_y_cond_x = prior_batch / ((noisy_prob_class) / prior_t).unsqueeze(2)
        w_x = p_y_cond_x * p_y_bar_x_yt.unsqueeze(1)
        w_x_yt = w_x[torch.arange(len(noisy_prob_class)), class_labels_idx]

        w_x_yt = torch.clamp(w_x_yt, min=0, max=1)
        w_x_yt = w_x_yt / torch.sum(w_x_yt, dim=1, keepdim=True)

        w_x_yt = torch.nan_to_num(w_x_yt, 0, 0, 0)

        D_yn_train = net(y + n, sigma, labels, augment_labels=augment_labels)
        D_yn_train = D_yn_train * w_x_yt[torch.arange(labels.shape[0]), class_labels_idx][:, None, None, None]
        with torch.no_grad():
            for c in range(labels.shape[1]):
                labels_temp = torch.nn.functional.one_hot(c * torch.ones(labels.shape[0], dtype=int).to(images.device), labels.shape[1])
                candidate = (class_labels_idx != c) & (w_x_yt[:, c] > self.tau)
                if candidate.sum() != 0:
                    D_yn = net((y + n)[candidate], sigma[candidate], labels_temp[candidate], augment_labels=augment_labels[candidate])
                    D_yn_train[candidate] = D_yn_train[candidate] + D_yn * w_x_yt[candidate, c][:, None, None, None]
        loss = weight * ((D_yn_train - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# TDSM Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VENoiseLoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, noise_type='clean', noise_rate=0., num_classes=10, dataset='cifar10', tau=1e-2):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.dataset = dataset
        self.sm = torch.nn.Softmax(dim=1)
        self.tau = tau

    def __call__(self, net, images, labels=None, augment_pipe=None, cls_net=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma

        with torch.no_grad():
            if self.noise_type == 'learn':
                logit, tm = cls_net(y + n, sigma, None)
                tm = torch.FloatTensor(tm).to(images.device)
                noisy_prob_class = torch.matmul(tm.transpose(1, 0).unsqueeze(0), self.sm(logit).unsqueeze(-1)).squeeze(-1)
            elif self.noise_type == 'sym':
                noisy_prob_class = self.noise_rate / (self.num_classes - 1) + (1 - self.num_classes * self.noise_rate / (self.num_classes - 1)) * self.sm(cls_net(y + n, sigma, None))
            elif self.noise_type == 'asym':
                if self.dataset == 'mnist':
                    tm = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1. - self.noise_rate, 0, 0, 0, 0, self.noise_rate, 0, 0],
                          [0, 0, 0, 1. - self.noise_rate, 0, 0, 0, 0, self.noise_rate, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1. - self.noise_rate, self.noise_rate, 0, 0, 0],
                          [0, 0, 0, 0, 0, self.noise_rate, 1. - self.noise_rate, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
                elif self.dataset == 'cifar10':
                    tm = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [self.noise_rate, 0, 1. - self.noise_rate, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1. - self.noise_rate, 0, self.noise_rate, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1 - self.noise_rate, 0, 0, self.noise_rate, 0, 0],
                          [0, 0, 0, self.noise_rate, 0, 1. - self.noise_rate, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1., 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, self.noise_rate, 0, 0, 0, 0, 0, 0, 0, 1 - self.noise_rate]]
                elif self.dataset == 'cifar100':
                    tm = np.eye(self.num_classes)
                    noise_label_lst = np.array([51, 32, 11, 42, 30, 20,  7, 14, 13, 10, 16, 35, 17, 48, 18, 19, 28,
                                                37, 24, 21, 25, 31, 39, 33,  6, 84, 45, 29, 61, 44, 55, 38, 67, 49,
                                                63, 46, 50, 68, 15, 40, 86, 69, 43, 88, 78, 77, 98, 52, 58, 60, 65,
                                                53, 56, 57, 62, 72, 59, 83, 90, 96, 71,  9, 70, 64, 66, 74, 75, 73,
                                                76, 81, 82, 23, 95, 91, 80, 34, 12, 79, 93, 99, 36, 85, 92,  0, 94,
                                                89, 87, 22, 97, 41,  8,  1, 54, 27,  5,  4, 47,  3,  2, 26])
                    for i in range(self.num_classes):
                        tm[i, i] = 1 - self.noise_rate
                        tm[i, noise_label_lst[i]] = self.noise_rate
                else:
                    raise NotImplementedError
                tm = torch.FloatTensor(tm).to(images.device)
                noisy_prob_class = torch.matmul(tm.transpose(1, 0).unsqueeze(0), self.sm(cls_net(y + n, sigma, None)).unsqueeze(-1)).squeeze(-1)
            else:
                raise NotImplementedError

        if self.noise_type == 'learn':
            prior = tm / (tm.sum(dim=0, keepdims=True))
            prior_t = (1 / self.num_classes) * (tm.sum(dim=0, keepdims=True))
        elif self.noise_type == 'sym':
            prior = (1 - self.noise_rate) * torch.eye(self.num_classes) + self.noise_rate / (self.num_classes - 1) * (1 - torch.eye(self.num_classes))
            prior = torch.FloatTensor(prior).to(images.device)
            prior_t = (1 / self.num_classes) * (prior.sum(dim=0, keepdims=True))
        elif self.noise_type == 'asym':
            prior = tm / (tm.sum(dim=0, keepdims=True))
            prior_t = (1 / self.num_classes) * (tm.sum(dim=0, keepdims=True))
        else:
            raise NotImplementedError

        prior = prior.transpose(1, 0)
        prior_inverse = torch.inverse(prior)
        class_labels_idx = torch.argmax(labels, dim=1)

        p_y_bar_x_yt = torch.matmul(noisy_prob_class / prior_t, prior_inverse.transpose(1, 0))
        prior_batch = prior.unsqueeze(0).repeat(len(noisy_prob_class), 1, 1)
        p_y_cond_x = prior_batch / ((noisy_prob_class) / prior_t).unsqueeze(2)
        w_x = p_y_cond_x * p_y_bar_x_yt.unsqueeze(1)
        w_x_yt = w_x[torch.arange(len(noisy_prob_class)), class_labels_idx]

        w_x_yt = torch.clamp(w_x_yt, min=0, max=1)
        w_x_yt = w_x_yt / torch.sum(w_x_yt, dim=1, keepdim=True)

        w_x_yt = torch.nan_to_num(w_x_yt, 0, 0, 0)

        D_yn_train = net(y + n, sigma, labels, augment_labels=augment_labels)
        D_yn_train = D_yn_train * w_x_yt[torch.arange(labels.shape[0]), class_labels_idx][:, None, None, None]
        with torch.no_grad():
            for c in range(labels.shape[1]):
                labels_temp = torch.nn.functional.one_hot(c * torch.ones(labels.shape[0], dtype=int).to(images.device), labels.shape[1])
                candidate = (class_labels_idx != c) & (w_x_yt[:, c] > self.tau)
                if candidate.sum() != 0:
                    D_yn = net((y + n)[candidate], sigma[candidate], labels_temp[candidate], augment_labels=augment_labels[candidate])
                    D_yn_train[candidate] = D_yn_train[candidate] + D_yn * w_x_yt[candidate, c][:, None, None, None]
        loss = weight * ((D_yn_train - y) ** 2)
        return loss
#----------------------------------------------------------------------------