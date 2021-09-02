from scipy.stats import spearmanr as corr
from .model_basis import _iFAST_model_serial_dependence_
import torch
import torch.nn as nn
import timeit
import numpy as np


class fr_VQA_model(object):
    def __init__(self, model_config):

        self.epochs = model_config.get('epochs', 200)
        self.lr = float(model_config.get('lr', 1e-4))
        self.optimizer = model_config.get('optimizer', 'adam')
        self.train_batch_size = model_config.get('train_batch_size', 2)
        self.test_batch_size = model_config.get('test_batch_size', 2)
        self.model = _iFAST_model_serial_dependence_().cuda()
        self.loss = nn.SmoothL1Loss().cuda()
        self.weight = 0.05
        self.test_freq = model_config.get('test_freq', 5)
        self.iter = 0

    def run(self, train_loader, test_loader, optimizer, lr_sch=None):

        torch.autograd.set_detect_anomaly(True)
        n_epoch = self.epochs

        best_srcc = 0
        best_epoch = 0

        info = []

        torch.backends.cudnn.benchmark = True
        for i in range(n_epoch):

            start_time = timeit.default_timer()
            loss_train, srcc, loss_eval, train_srcc, info0 = self.train(train_loader, optimizer, test_loader)
            # srcc, loss_eval, pred_quality, ground_truth = self.eval(test_loader)

            if lr_sch is not None:  # update lr
                pass

            info.append(info0)
            stop_time = timeit.default_timer()
            minutes, seconds = divmod(stop_time - start_time, 60)
            print('at Epoch %3d, train_srcc: %.4f, train_loss: %.2f, eval_srocc: %.4f, eval_loss: %.2f' %
                  (i, train_srcc, loss_train, srcc, loss_eval) +
                  ' elapsed: {:02.0f}:{:05.2f}'.format(minutes, seconds))

            if srcc > best_srcc:
                best_srcc = srcc
                best_epoch = i

                # self.save_model()

        print('--' * 15)
        print('best epoch: %3d, best srocc: %.4f' % (best_epoch, best_srcc))

        info = np.vstack(info)
        return best_srcc, best_epoch, info
        # return best_srcc, best_epoch, loss_train_, loss_test_, srcc_test_

    def train(self, train_loader, optimizer, test_loader):

        self.model.train()

        loss_total = 0.
        loss_freq = 0.

        max_srcc = 0
        fin_loss = 0
        info = []

        pred_list = []
        mos_list = []
        hidden = ((0.1 * torch.ones(self.train_batch_size, 4).cuda(),
                   0.1 * torch.ones(self.train_batch_size, 4).cuda(),
                   0.1 * torch.ones(self.train_batch_size, 1).cuda()),
                  (0.1 * torch.ones(self.train_batch_size, 4).cuda(),
                   0.1 * torch.ones(self.train_batch_size, 4).cuda(),
                   0.1 * torch.ones(self.train_batch_size, 1).cuda()),
                  0.1 * torch.ones(self.train_batch_size, 4).cuda())

        for (appearance, motion_content, motion_desc, mos) in train_loader:
            trb = appearance.shape[0]
            sh = (-1, 1, motion_desc.shape[4], motion_desc.shape[5])

            appearance = appearance.cuda()
            motion_content = motion_content.cuda()
            motion_desc_simi = motion_desc[:, :, :, 2, :, :].view(sh).cuda()

            optimizer.zero_grad()

            mos_pred = self.model(appearance, motion_content, motion_desc_simi, hidden)
            mos_pred = torch.mean(mos_pred.view((trb, -1)), dim=1)

            del appearance, motion_desc_simi, motion_content

            loss = self.loss(mos_pred, mos.float().cuda()) * 10
            loss_total += float(loss)
            loss_freq += float(loss)

            loss.backward()
            optimizer.step()

            pred_list.append(mos_pred.data.cpu().numpy())
            mos_list.append(mos.numpy())
            self.iter += 1
            if (self.iter % self.test_freq) == 0:
                # To reproduce the results in the paper, you can validate every 5 iterations as given in the config file.
                # For a more common comparison, validate the performance every epoch is suggested. It is easy to change the code
                srcc, loss_eval, pred_quality, ground_truth = self.eval(test_loader)

                # (iter_cnt , loss_train, eval_srcc, loss_test, pred, mos)
                info.append(np.concatenate((np.array([self.iter, loss_freq, srcc, loss_eval]).reshape(1, -1),
                                            pred_quality.reshape(1, -1), ground_truth. reshape(1, -1)), 1))
                loss_freq = 0.
                if srcc > max_srcc:
                    max_srcc = srcc
                    fin_loss = loss_eval
                self.model.train()

        # loss_total /= len(train_loader)
        train_srcc = abs(corr(np.asarray(pred_list).flatten(), np.asarray(mos_list).flatten())[0])
        info = np.vstack(info)
        return loss_total, max_srcc, fin_loss, train_srcc, info

    def eval(self, test_loader):

        self.model.eval()

        mos_list = []
        pred_list = []

        loss = 0.

        with torch.no_grad():
            hidden = ((0.1 * torch.ones(self.test_batch_size, 4).cuda(),
                       0.1 * torch.ones(self.test_batch_size, 4).cuda(),
                       0.1 * torch.ones(self.test_batch_size, 1).cuda()),
                      (0.1 * torch.ones(self.test_batch_size, 4).cuda(),
                       0.1 * torch.ones(self.test_batch_size, 4).cuda(),
                       0.1 * torch.ones(self.test_batch_size, 1).cuda()),
                      0.1 * torch.ones(self.test_batch_size, 4).cuda())
            for (appearance, motion_content, motion_desc, mos) in test_loader:
                trb = appearance.shape[0]
                sh = (-1, 1, motion_desc.shape[4], motion_desc.shape[5])

                appearance = appearance.cuda()
                motion_content = motion_content.cuda()
                motion_desc_simi = motion_desc[:, :, :, 2, :, :].view(sh).cuda()

                mos_pred = self.model(appearance, motion_content, motion_desc_simi, hidden)
                mos_pred = torch.mean(mos_pred.view((trb, -1)), dim=1)

                del appearance, motion_desc_simi, motion_content

                loss_this = self.loss(mos_pred, mos.float().cuda()) * 10
                loss += float(loss_this)

                mos_list.append(mos.numpy())
                pred_list.append(mos_pred.data.cpu().numpy())

            # loss /= len(test_loader)
            pred_quality = np.asarray(pred_list).flatten()
            ground_truth = np.asarray(mos_list).flatten()
            srcc = abs(corr(pred_quality, ground_truth)[0])

        return srcc, loss, pred_quality, ground_truth

    def save_model(self, prefix='best_loss'):
        torch.save(self.model.state_dict(), prefix + '.pkl')

    def restore(self, prefix='best_loss'):
        self.model.load_state_dict(torch.load(prefix + '.pkl'))
