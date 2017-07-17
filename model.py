from util import read_conll, write_conll
import util
import decoder
from operator import itemgetter
from itertools import chain
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
import torch.optim as optim
import random
torch.manual_seed(1)


class MSTParserLSTM(nn.Module):

    def __init__(self, vocab, pos, rels, w2i, options):

        super(MSTParserLSTM, self).__init__()
        self.vocab = {word: ind for ind, word in enumerate(w2i)}
        self.pos = {word: ind for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.blstmFlag = options.blstmFlag
        self.model_path = options.output + '/model.pt'
        self.wordsCount = vocab
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.layers = options.layers
        self.dropout = options.dropout
        self.hidden_size = options.lstm_dims
        self.hidden_units = options.hidden_units
        # self.rdims = options.rdims
        dims = self.wdims + self.pdims
        self.wlookup = nn.Embedding(len(self.vocab), self.wdims)
        # self.wlookup.weight.requires_grad = False
        self.plookup = nn.Embedding(len(self.pos), self.pdims)
        # self.plookup.weight.requires_grad = False
        # self.load_pretrained_vectors(options)
        # self.rloopup = nn.Embedding(len(rels), self.rdims)

        # bidirectional lstm for feature extraction, can use one lstm too
        self.forward_lstm = nn.LSTMCell(dims, self.hidden_size)
        # self.backward_lstm = nn.LSTMCell(dims, self.hidden_size)

        # MLP for score calculating
        self.hidLayerFOH = nn.Linear(
            self.hidden_size * 2, self.hidden_units, bias=True)
        self.hidLayerFOM = nn.Linear(
            self.hidden_size * 2, self.hidden_units, bias=True)
        self.outLayer = nn.Linear(self.hidden_units, 1, bias=True)
        self.bias = Variable(torch.FloatTensor((self.hidden_units)))
        self.bias.requires_grad = True
        self.renew()

    def renew(self):
        self.f_h, self.f_c = self.init_hidden()
        self.b_h, self.b_c = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, self.hidden_size)),
                autograd.Variable(torch.zeros(1, self.hidden_size)))

    def load_pretrained_vectors(self, options):
        if options.external_embedding is not None:
            pretrained = torch.load(options.external_embedding)
            self.wlookup.weight.data.copy_(pretrained)

    # do the MLP
    def __getExpr(self, sentence, i, j, train):

        if sentence[i].headfov is None:
            sentence[i].headfov = self.hidLayerFOH(torch.cat(
                [sentence[i].lstms[0], sentence[i].lstms[1]], 1))
        if sentence[j].modfov is None:
            sentence[j].modfov = self.hidLayerFOM(torch.cat(
                [sentence[j].lstms[0], sentence[j].lstms[1]], 1))
        hidout = torch.add(sentence[i].headfov, sentence[j].modfov)
        # print(hidout.data.numpy())
        # print(sentence[i].headfov.data)
        output = self.outLayer(hidout)

        # output = F.tanh(output)
        # print(output.data.numpy())
        return output

    def __evaluate(self, sentence, train):
        exprs = [[self.__getExpr(sentence, i, j, train) for j in range(
            len(sentence))] for i in range(len(sentence))]
        tr_exprs = torch.FloatTensor(exprs)

        scores = [[output
                   for output in exprsRow] for exprsRow in exprs]

        return scores, exprs

    def forward(self, sentence, train):
        # for i in range(len(sentence)):
        #     print(sentence[i].lstms)
        exprs = [[self.__getExpr(sentence, i, j, train) for j in range(
            len(sentence))] for i in range(len(sentence))]

        # scores = np.array([[output.data.numpy()
        #                     for output in exprsRow] for exprsRow in exprs])
        scores = [[output.data.numpy()
                   for output in exprsRow] for exprsRow in exprs]
        scores = np.squeeze(scores)
        # print(scores)
        return scores, exprs

    def save(self):
        self.model.save_state_dict(self.model_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.model_path))
    # def __evaluateLabel(self, sentence, i, j):
    #     if sentence[i].rheadfov is None:
    #         sentence[i].rheadfov = self.rhidLayerFOH.expr(
    #         ) * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
    #     if sentence[j].rmodfov is None:
    #         sentence[j].rmodfov = self.rhidLayerFOM.expr(
    #         ) * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

    #     if self.hidden2_units > 0:
    #         output = self.routLayer.expr() * self.activation(self.rhid2Bias.expr() + self.rhid2Layer.expr() *
    #                                                          self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr())) + self.routBias.expr()
    #     else:
    #         output = self.routLayer.expr() * self.activation(sentence[i].rheadfov + sentence[
    #             j].rmodfov + self.rhidBias.expr()) + self.routBias.expr()

    #     return output.value(), output

    # def Save(self, filename):
    #     self.model.save(filename)

    # def Load(self, filename):
    #     self.model.load(filename)


def Train(model, epoch, conll_path):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=0.5)
    errors = 0
    batch = 0
    eloss = 0.0
    mloss = 0.0
    eerrors = 0
    etotal = 0
    start = time.time()
    firstFlag = True
    # np.savetxt('wembedding', model.wlookup.weight.data.numpy())
    # np.savetxt('pembedding', model.plookup.weight.data.numpy())
    # time.sleep(10)
    with open(conll_path, 'r') as conllFP:
        shuffledData = list(read_conll(conllFP))
        random.shuffle(shuffledData)

        errs = []
        lerrs = []
        eeloss = 0.0

        for iSentence, sentence in enumerate(shuffledData):

            if iSentence % 100 == 0 and iSentence != 0:
                print('Processing sentence number:', iSentence, 'Loss:', eloss / etotal,
                      'Errors:', (float(eerrors)) / etotal, 'Time', time.time() - start)
                start = time.time()
                eerrors = 0
                eloss = 0.0
                etotal = 0
                lerrors = 0
                ltotal = 0

            conll_sentence = [entry for entry in sentence if isinstance(
                entry, util.ConllEntry)]

            # model.zero_grad()
            for entry in conll_sentence:
                c = float(model.wordsCount.get(entry.norm, 0))
                dropFlag = (random.random() < (c / (0.25 + c)))
                windex = model.vocab.get(entry.norm, 0)
                pindex = model.pos.get(entry.pos, 0)
                ww = autograd.Variable(
                    torch.LongTensor([windex]))

                wordvec = model.wlookup(ww)
                posvec = model.plookup(autograd.Variable(
                    torch.LongTensor([pindex])))
                # print(wordvec)
                # print(posvec)
                evec = None

                entry.vec = torch.cat((wordvec, posvec), 1)

                # print(windex, pindex)
                # print(entry.norm)
                entry.lstms = [entry.vec, entry.vec]
                entry.headfov = None  # head
                entry.modfov = None  # modifier

                entry.rheadfov = None
                entry.rmodfov = None

            if model.blstmFlag:

                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):

                    model.f_h, model.f_c = model.forward_lstm(
                        entry.vec, (model.f_h, model.f_c))
                    # model.b_h, model.b_c = model.backward_lstm(
                    # rentry.vec, (model.b_h, model.b_c))

                    entry.lstms[1] = model.f_h
                    entry.lstms[0] = model.f_h
                    # rentry.lstms[0] = model.b_h
                    # print(model.f_h, model.b_h)

                model.renew()
            scores, exprs = model.forward(conll_sentence, True)
            gold = [entry.parent_id for entry in conll_sentence]
            heads = decoder.parse_proj(
                scores, gold)

            # if model.labelsFlag:
            #     for modifier, head in enumerate(gold[1:]):
            #         rscores, rexprs = self.__evaluateLabel(
            #             conll_sentence, head, modifier + 1)
            #         goldLabelInd = self.rels[
            #             conll_sentence[modifier + 1].relation]
            #         wrongLabelInd = max(((l, scr) for l, scr in enumerate(
            #             rscores) if l != goldLabelInd), key=itemgetter(1))[0]
            #         if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
            #             lerrs.append(
            #                 rexprs[wrongLabelInd] - rexprs[goldLabelInd])

            e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
            # UAS
            eerrors += e
            if e > 0:
                loss = [(torch.abs(exprs[h][i] - exprs[g][i])) for i, (h, g)
                        in enumerate(zip(heads, gold)) if h != g]  # * (1.0/float(e))
                real_loss = [(scores[h][i] - scores[g][i]) for i, (h, g)
                             in enumerate(zip(heads, gold)) if h != g]
                if random.random() > 0.9999:
                    # pass
                    print(scores)
                    print(heads, gold)
                    time.sleep(1)
                eloss += (e)
                mloss += (e)
                errs.extend(loss)

            etotal += len(conll_sentence)

            if len(errs) > 0:
                eeloss = 0.0
                # s = time.time()
                if len(errs) > 0:
                    # * (1.0/(float(len(errs))))
                    eerrs = sum((errs))

                    eerrs.backward()

                    optimizer.step()
                    errs = []
                    lerrs = []
                # print(time.time() - s)

    # if len(errs) > 0:
    #     eerrs = (torch.sum(errs + lerrs))  # * (1.0/(float(len(errs))))
    #     eerrs.backward()
    #     optimizer.step()

    #     errs = []
    #     lerrs = []
    #     eeloss = 0.0

    print("Loss: ", mloss / iSentence)


def Predict(self, conll_path, options):
    with open(conll_path, 'r') as conllFP:
        for iSentence, sentence in enumerate(read_conll(conllFP)):
            conll_sentence = [entry for entry in sentence if isinstance(
                entry, util.ConllEntry)]

            for entry in conll_sentence:
                wordvec = self.wlookup[
                    entry.norm] if self.wdims > 0 else None
                posvec = self.plookup[
                    entry.pos] if self.pdims > 0 else None

                entry.vec = concatenate(
                    filter(None, [wordvec, posvec, evec]))

                entry.lstms = [entry.vec, entry.vec]
                entry.headfov = None
                entry.modfov = None

                entry.rheadfov = None
                entry.rmodfov = None

            if self.blstmFlag:
                # lstm_forward = self.builders[0].initial_state()
                # lstm_backward = self.builders[1].initial_state()

                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    # lstm_forward = lstm_forward.add_input(entry.vec)
                    # lstm_backward = lstm_backward.add_input(rentry.vec)

                    # entry.lstms[1] = lstm_forward.output()
                    # rentry.lstms[0] = lstm_backward.output()
                    f_output, self.f_hidden = forward_lstm(
                        entry.vec, self.f_hidden)
                    b_output, self.b_hidden = backward_lstm(
                        rentry.vec, self.b_hidden)

                    entry.lstm[1] = f_output
                    rentry.lstm[0] = b_output

            scores, exprs = self.__evaluate(conll_sentence, True)
            heads = decoder.parse_proj(scores)

            for entry, head in zip(conll_sentence, heads):
                entry.pred_parent_id = head
                entry.pred_relation = '_'

            dump = False

            if self.labelsFlag:
                for modifier, head in enumerate(heads[1:]):
                    scores, exprs = self.__evaluateLabel(
                        conll_sentence, head, modifier + 1)
                    conll_sentence[
                        modifier + 1].pred_relation = self.irels[max(enumerate(scores), key=itemgetter(1))[0]]

            if not dump:
                yield sentence
