import os
import pickle
import sys
from tqdm import tqdm

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
    average_precision_score, accuracy_score, precision_score,f1_score,recall_score
from sklearn.neighbors import KNeighborsClassifier

from loss import * #NTXentLoss, NTXentLoss_poly

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

def Trainer(model,  temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, configs_target, experiment_log_dir, training_mode, model_F=None, model_F_optimizer=None,
            classifier=None, classifier_optimizer=None, epochs=40, pipes = None):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    """Pretraining"""
    if training_mode == 'pre_train':
        print('Pretraining on source dataset')
        for epoch in range(1, epochs + 1):
            # Train and validate
            """Train. In fine-tuning, this part is also trained???"""
            train_loss, train_acc, train_auc = model_pretrain(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion,
                                                              train_dl, config, device, training_mode, model_F=model_F, model_F_optimizer=model_F_optimizer, pipes=pipes)

            if training_mode != 'self_supervised':  # use scheduler in all other modes.
                scheduler.step(train_loss)
            logger.debug(f'\nPre-training Epoch : {epoch}\n'
                         f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\t | \tTrain AUC : {train_auc:2.4f}\n'
                         )

        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True) # only save in self_supervised mode.
        chkpoint = {'model_state_dict': model.state_dict(),}
        torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
        print('Pretrained model is stored at folder:{}'.format(experiment_log_dir+'saved_models'+'ckp_last.pt'))

    """Fine-tuning and Test"""
    if training_mode != 'pre_train':  # no need to run the evaluation for self-supervised mode.
        """fine-tune"""
        print('Fine-tune  on Fine-tuning set')
        performance_list = []
        for epoch in range(1, epochs + 1):
            valid_loss, valid_acc, valid_auc, valid_prc, emb_finetune, label_finetune, F1 = model_finetune(model, temporal_contr_model, valid_dl, config, configs_target, device, training_mode,
                                                   model_optimizer, model_F=model_F, model_F_optimizer=model_F_optimizer,
                                                        classifier=classifier, classifier_optimizer=classifier_optimizer, pipes=pipes)

            if training_mode != 'pre_train':  # use scheduler in all other modes.
                scheduler.step(valid_loss)
            logger.debug(f'\nEpoch : {epoch}\n'
                         f'finetune Loss  : {valid_loss:.4f}\t | \tfinetune Accuracy : {valid_acc:2.4f}\t | '
                         f'\tfinetune AUC : {valid_auc:2.4f} \t |finetune PRC: {valid_prc:0.4f} ')



            # evaluate on the test set
            """Testing set"""
            logger.debug('\nTest on Target dataset test set')
            try:
                test_loss, test_acc, test_auc, test_prc, emb_test, label_test, performance = model_test(model, temporal_contr_model, test_dl, configs_target, device, training_mode,
                                                                model_F=model_F, model_F_optimizer=model_F_optimizer,
                                                             classifier=classifier, classifier_optimizer=classifier_optimizer, logger = logger, pipes=pipes)
            except Exception as e:
                print(f"error in test: {e}")
            

            performance_list.append(performance)
        performance_array = np.array(performance_list)
        best_performance = performance_array[np.argmax(performance_array[:,0], axis=0)]
        print('Best Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | PRC=%.4f'
              % (best_performance[0], best_performance[1], best_performance[2], best_performance[3], best_performance[4], best_performance[5]))
        
        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True) # only save in self_supervised mode.
        chkpoint = {'model_state_dict': model.state_dict(),}
        torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
        print('Finetuned model is stored at folder:{}'.format(experiment_log_dir+'saved_models'+'ckp_last.pt'))

    logger.debug("\n################## Training is Done! #########################")



def model_pretrain(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config,
                   device, training_mode, model_F=None, model_F_optimizer=None, pipes=None):
    total_loss = []
    total_acc = []
    total_auc = []
    model.train()

    # for batch_idx, (data, labels, aug1, data_f, aug1_f) in enumerate(train_loader):
    # usando tqdm
    for data, labels, aug1, data_f, aug1_f in tqdm(train_loader):
        data, labels = data.float().to(device), labels.long().to(device) # data: [128, 1, 178], labels: [128]
        aug1 = aug1.float().to(device)  # aug1 = aug2 : [128, 1, 178]
        data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)  # aug1 = aug2 : [128, 1, 178]

        # optimizer
        model_optimizer.zero_grad()

        """Produce embeddings"""
        h_t, z_t, h_f, z_f=model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug=model(aug1, aug1_f)


        """Compute Pre-train loss"""
        """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
        nt_xent_criterion = NTXentLoss_poly(device, config.batch_size, config.Context_Cont.temperature,
                                       config.Context_Cont.use_cosine_similarity) # device, 128, 0.2, True

        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)

        l_TF = nt_xent_criterion(z_t, z_f)
        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
        loss_c = (1+ l_TF -l_1) + (1+ l_TF -l_2) + (1+ l_TF -l_3)

        lam = 0.2
        loss = lam *(loss_t + loss_f) + (1- lam)*loss_c

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

    print('preptraining: overall loss:{}, l_t: {}, l_f:{}, l_c:{}'.format(loss,loss_t,loss_f, loss_c))

    total_loss = torch.tensor(total_loss).mean()
    if training_mode == "pre_train":
        total_acc = 0
        total_auc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
        total_auc = torch.tensor(total_auc).mean()
    return total_loss, total_acc, total_auc


def model_finetune(model, temporal_contr_model, val_dl, config, configs_target, device, training_mode, model_optimizer, model_F=None, model_F_optimizer=None,
                   classifier=None, classifier_optimizer=None, pipes=None):
    model.train()
    classifier.train()

    total_loss = []
    total_acc = []
    total_auc = []  # it should be outside of the loop
    total_prc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    print(f"format of finetune dataset: {len(val_dl)} batches of size {val_dl.batch_size}, resulting in {val_dl.batch_size*len(val_dl)} samples")
    for data, labels, aug1, data_f, aug1_f in tqdm(val_dl):
        # print('Fine-tuning: {} of target samples'.format(labels.shape[0]))
        data, labels = data.float().to(device), labels.long().to(device)
        data_f = data_f.float().to(device)
        aug1 = aug1.float().to(device)
        aug1_f = aug1_f.float().to(device)

        # """if random initialization:"""
        model_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        """Produce embeddings"""
        # print(f"entrada do treino finetune: {labels}")
        h_t, z_t, h_f, z_f=model(data, data_f)

        h_t_aug, z_t_aug, h_f_aug, z_f_aug=model(aug1, aug1_f)

        nt_xent_criterion = NTXentLoss_poly(device, config.target_batch_size, config.Context_Cont.temperature,
                                            config.Context_Cont.use_cosine_similarity)
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)

        l_TF = nt_xent_criterion(z_t, z_f)
        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug,
                                                                                                            z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3) #


        """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test"""
        if pipes == "frequency":
            fea_concat = z_f
        elif pipes == "time":
            fea_concat = z_t
        else:
            fea_concat = torch.cat((z_t, z_f), dim=1)
        predictions = classifier(fea_concat) # how to define classifier? MLP? CNN?
        fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
        loss_p = criterion(predictions, labels) # predictor loss, actually, here is training loss

        

        lam = 0.2
        loss =  loss_p + (1-lam)*loss_c + lam*(loss_t + loss_f )
        loss = loss_p

        acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
        onehot_label = F.one_hot(labels, num_classes=configs_target.num_classes_target)
        pred_numpy = predictions.detach().cpu().numpy()

        #verifica qual classe de labels nao possui exemplos
        no_examples_classes = [i for i in range(config.num_classes_target) if i not in labels]
        # remove as colunas de indice em no_examples_classes
        pred_numpy_removed_classes = np.delete(pred_numpy, no_examples_classes, axis=1)
        onehot_label_removed_classes = np.delete(onehot_label.cpu(), no_examples_classes, axis=1)

        try:
            auc_bs = roc_auc_score(onehot_label_removed_classes.detach().cpu().numpy(), pred_numpy_removed_classes, average="macro", multi_class="ovr" )
        except Exception as e:
            print(f"{labels_original=}")
            print(f"{pred_numpy=}")
            raise e
        prc_bs = average_precision_score(onehot_label_removed_classes.detach().cpu().numpy(), pred_numpy_removed_classes)




        total_acc.append(acc_bs)
        total_auc.append(auc_bs)
        total_prc.append(prc_bs)
        total_loss.append(loss.item())
        loss.backward()


        model_optimizer.step()

        classifier_optimizer.step()

        if training_mode != "pre_train":
            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())


    labels_numpy = labels.detach().cpu().numpy()
    pred_numpy = np.argmax(pred_numpy, axis=1)
    precision = precision_score(labels_numpy, pred_numpy, average='macro', zero_division=0)  # labels=np.unique(ypred))
    recall = recall_score(labels_numpy, pred_numpy, average='macro', zero_division=0)  # labels=np.unique(ypred))
    F1 = f1_score(labels_numpy, pred_numpy, average='macro', zero_division=0)  # labels=np.unique(ypred))
    acc = accuracy_score(labels_numpy, pred_numpy,)

    print('finetune: Acuracy = %.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f' % (acc * 100, precision * 100, recall * 100, F1 * 100))

    # """Save embeddings for visualization"""
    # pickle.dump(features1_f, open('embeddings/fea_t_withLc.p', 'wb'))
    # pickle.dump(fea_f, open('embeddings/fea_f_withLc.p', 'wb'))
    # print('embedding saved')

    total_loss = torch.tensor(total_loss).mean()  # average loss
    total_acc = torch.tensor(total_acc).mean()  # average acc
    total_auc = torch.tensor(total_auc).mean()  # average acc
    total_prc = torch.tensor(total_prc).mean()
    return total_loss, total_acc, total_auc, total_prc, fea_concat_flat, trgs, F1

def plot_confusion_matrix(cm, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)


def model_test(model, temporal_contr_model, test_dl, config,  device, training_mode, model_F=None, model_F_optimizer=None,
               classifier=None, classifier_optimizer=None, logger = None, pipes=None):
    model.eval()
    classifier.eval()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []
    total_precision, total_recall, total_f1 = [], [], []

    criterion = nn.CrossEntropyLoss() # This loss is not used in gradient. It means nothing.
    outs = np.array([])
    trgs = np.array([])
    emb_test_all = []

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for data, labels, _,data_f, _ in test_dl:
            # print('TEST: {} of target samples'.format(labels.shape[0]))
            data, labels = data.float().to(device), labels.long().to(device)
            data_f = data_f.float().to(device)

            """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test"""
            h_t, z_t, h_f, z_f = model(data, data_f)

            fea_concat = torch.cat((z_t, z_f), dim=1)
            # print(f"data: {data}", f"data_f: {data_f}")
            if pipes == "frequency":
                fea_concat = z_f
            elif pipes == "time":
                fea_concat = z_t
            else:
                fea_concat = torch.cat((z_t, z_f), dim=1)
            predictions_test = classifier(fea_concat)  # how to define classifier? MLP? CNN?
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
            emb_test_all.append(fea_concat_flat)

            if training_mode != "pre_train":
                loss = criterion(predictions_test, labels)
                acc_bs = labels.eq(predictions_test.detach().argmax(dim=1)).float().mean()
                onehot_label = F.one_hot(labels, num_classes=config.num_classes_target)
                pred_numpy = predictions_test.detach().cpu().numpy()
                labels_numpy = labels.detach().cpu().numpy()
                
                no_examples_classes = [i for i in range(config.num_classes_target) if i not in labels]
                # remove as colunas de indice em no_examples_classes
                pred_numpy_removed_classes = np.delete(pred_numpy, no_examples_classes, axis=1)
                onehot_label_removed_classes = np.delete(onehot_label.cpu(), no_examples_classes, axis=1)
                try:
                    auc_bs = roc_auc_score(onehot_label_removed_classes.detach().cpu().numpy(), pred_numpy_removed_classes,
                                       average="macro", multi_class="ovr")
                except Exception as e:
                    print(f"{onehot_label=}")
                    print(f"{pred_numpy=}")
                    raise e

                prc_bs = average_precision_score(onehot_label_removed_classes.detach().cpu().numpy(), pred_numpy_removed_classes, average="macro")


                pred_numpy = np.argmax(pred_numpy, axis=1)
                # precision = precision_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))
                # recall = recall_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))
                # F1 = f1_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))

                total_acc.append(acc_bs)
                total_auc.append(auc_bs)
                total_prc.append(prc_bs)
                # total_precision.append(precision)
                # total_recall.append(recall)
                # total_f1.append(F1)

                total_loss.append(loss.item())
                pred = predictions_test.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())
            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))
    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]

    # print(f"{labels_numpy_all=} {pred_numpy_all=}")

    # plot confusion matrix using sklearn
    cm = confusion_matrix(labels_numpy_all, pred_numpy_all)
    # plot_confusion_matrix(cm)


    # print('Test classification report', classification_report(labels_numpy_all, pred_numpy_all))
    # print(confusion_matrix(labels_numpy_all, pred_numpy_all))
    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', zero_division=0)
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', zero_division=0)
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', zero_division=0)
    acc = accuracy_score(labels_numpy_all, pred_numpy_all, )

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    total_auc = torch.tensor(total_auc).mean()
    total_prc = torch.tensor(total_prc).mean()

    # precision_mean = torch.tensor(total_precision).mean()
    # recal_mean = torch.tensor(total_recall).mean()
    # f1_mean = torch.tensor(total_f1).mean()
    performance = [acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100]
    if logger:
        logger.debug('Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | PRC=%.4f'
            % (acc*100, precision * 100, recall * 100, F1 * 100, total_auc*100, total_prc*100))
    else:
        print('Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | PRC=%.4f'
            % (acc*100, precision * 100, recall * 100, F1 * 100, total_auc*100, total_prc*100))

    emb_test_all = torch.concat(tuple(emb_test_all))
    return total_loss, total_acc, total_auc, total_prc, emb_test_all, trgs, performance
