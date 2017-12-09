import os.path
from collections import deque

import numpy as np
import torch
import torch.nn.functional
from torch.autograd import Variable

from core.platform import PrivateGameState
from learning.vdcnn.VDCNN import VDCNN

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def get_model(recover_path=None):
    classes = sum(PrivateGameState.max_combinations()) + 1
    model = VDCNN(n_classes=classes, input_dim=9, depth=29, shortcut=True)
    if use_cuda:
        model.cuda()
    if recover_path is not None and os.path.exists(recover_path):
        print("loading model", recover_path)
        model.load_state_dict(torch.load(recover_path))
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)

class DeepLearner(object):
    def __init__(self, model):
        self.model = model
        self.model = self.model.train(False)
        self.cache = deque(maxlen=1000)
        self.keys = deque(maxlen=1000)
    def _gen_last_dealt_hand(self, private_state):
        dealt_hand = np.zeros((1, 54))
        if private_state.last_dealt_hand is not None:
            for card in private_state.last_dealt_hand.hand:
                seq = card.seq()
                if seq < 13:
                    color_seq = card.color_seq()
                    dealt_hand[0, seq * 4 + color_seq] = 1
                else:
                    dealt_hand[0, 52 + seq - 13] = 1
        return dealt_hand

    def _gen_hand(self, private_state):
        hand = np.zeros((1, 54))
        for card in private_state.agent_state.cards:
            seq = card.seq()
            if seq < 13:
                color_seq = card.color_seq()
                hand[0, seq * 4 + color_seq] = 1
            else:
                hand[0, 52 + seq - 13] = 1
        return hand

    def _gen_other_cards(self, private_state):
        cards = np.zeros((1, 54))
        for card in private_state.other_cards:
            seq = card.seq()
            if seq < 13:
                color_seq = card.color_seq()
                cards[0, seq * 4 + color_seq] = 1
            else:
                cards[0, 52 + seq - 13] = 1
        return cards

    def _gen_landlord(self, private_state):
        landlord = private_state.x == 0
        land_tensor = np.full((1, 54), landlord)
        return land_tensor

    def _gen_landlord_guard(self, private_state):
        guard = private_state.x == 1
        return np.full((1, 54), guard)

    def _gen_is_last_pass(self, private_state):
        last_pass = private_state.pass_count == 1
        return np.full((1, 54), last_pass)

    def _gen_other_agents_number_cards(self, private_state):
        cards = np.zeros((3, 54))
        for i, card_num in enumerate(private_state.agent_num_cards):
            cards[i] = card_num
        return cards

    def _gen_input(self, private_state):
        last_dealt_hand = self._gen_last_dealt_hand(private_state)
        last_hand = self._gen_hand(private_state)
        other_hand = self._gen_other_cards(private_state)
        landload = self._gen_landlord(private_state)
        landload_guard = self._gen_landlord_guard(private_state)
        last_pass = self._gen_is_last_pass(private_state)
        other_cards = self._gen_other_agents_number_cards(private_state)
        return np.concatenate(
            (last_dealt_hand, last_hand, other_hand, landload, landload_guard, last_pass, other_cards), axis=0)

    # def get_action(self, net_out):
    #     indices = np.array(PrivateGameState.getAllActions())
    #     vals, sel_indices = torch.max(net_out, 0)
    #     return indices[sel_indices[0]]
    #
    # def estimate_actions(self, private_state):
    #     net_in = self._gen_input(private_state)
    #     net_out = self.model(net_in)
    #     card_indices = self.get_action(net_out)
    #     return card_indices
    def estimate_leaf_prior_value(self, private_state, required_mask):
        try:
            idx = self.keys.index(private_state)
        except:
            net_in = self._gen_input(private_state)
            net_in = np.expand_dims(net_in, axis=0)
            net_in = Variable(torch.from_numpy(net_in).float(), volatile=True)
            if use_cuda:
                net_in = net_in.cuda()
            prior, value = self.model(net_in)
            mask = np.array(required_mask, dtype="uint8")
            mask = torch.from_numpy(mask).cuda().nonzero().squeeze()
            sub_prior = prior.index_select(1, mask)
            sub_prior = torch.nn.functional.softmax(sub_prior)
            sub_prior, value = sub_prior.data, value.data
            if use_cuda:
                sub_prior, value = sub_prior.cpu(), value.cpu()
            res = (sub_prior.numpy()[0], value.numpy()[0])
            res_0 = np.zeros(prior.size()[1])
            res_0[required_mask] = res[0]
            res = (res_0, res[1])
            self.keys.append(private_state)
            self.cache.append(res)
            #print(res[0].shape)
            return res
        else:
            return self.cache[idx]
