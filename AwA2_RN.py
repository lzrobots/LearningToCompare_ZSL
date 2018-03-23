import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import scipy.io as sio
import math
import argparse
import random
import os
from sklearn.metrics import accuracy_score
import pdb

# '/home/lz/Workspace/ZSL/data/Animals_with_Attributes2',

parser = argparse.ArgumentParser(description="Zero Shot Learning")
parser.add_argument("-b","--batch_size",type = int, default = 32)
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 1e-5)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters

BATCH_SIZE = args.batch_size
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

class AttributeNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
    


def main():
    # step 1: init dataset
    print("init dataset")
    
    dataroot = './data'
    dataset = 'AwA2_data'
    image_embedding = 'res101' 
    class_embedding = 'att'

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
    feature = matcontent['features'].T
    label = matcontent['labels'].astype(int).squeeze() - 1
    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
    # numpy array index starts from 0, matlab starts from 1
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
  
    attribute = matcontent['original_att'].T 

    x = feature[trainval_loc] # train_features
    train_label = label[trainval_loc].astype(int)  # train_label
    att = attribute[train_label] # train attributes
    
    x_test = feature[test_unseen_loc]  # test_feature
    test_label = label[test_unseen_loc].astype(int) # test_label
    x_test_seen = feature[test_seen_loc]  #test_seen_feature
    test_label_seen = label[test_seen_loc].astype(int) # test_seen_label
    test_id = np.unique(test_label)   # test_id
    att_pro = attribute[test_id]      # test_attribute
    
    
    # train set
    train_features=torch.from_numpy(x)
    print(train_features.shape)

    train_label=torch.from_numpy(train_label).unsqueeze(1)
    print(train_label.shape)

    # attributes
    all_attributes=np.array(attribute)
    print(all_attributes.shape)
    
    attributes = torch.from_numpy(attribute)
    # test set

    test_features=torch.from_numpy(x_test)
    print(test_features.shape)

    test_label=torch.from_numpy(test_label).unsqueeze(1)
    print(test_label.shape)

    testclasses_id = np.array(test_id)
    print(testclasses_id.shape)

    test_attributes = torch.from_numpy(att_pro).float()
    print(test_attributes.shape)

    
    test_seen_features = torch.from_numpy(x_test_seen)
    print(test_seen_features.shape)
    
    test_seen_label = torch.from_numpy(test_label_seen)
    
    

    train_data = TensorDataset(train_features,train_label)
    

    # init network
    print("init networks")
    attribute_network = AttributeNetwork(85,1024,2048)
    relation_network = RelationNetwork(4096,400)

    attribute_network.cuda(GPU)
    relation_network.cuda(GPU)

    attribute_network_optim = torch.optim.Adam(attribute_network.parameters(),lr=LEARNING_RATE,weight_decay=1e-5)
    attribute_network_scheduler = StepLR(attribute_network_optim,step_size=200000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=200000,gamma=0.5)

    # if os.path.exists("./models/zsl_awa2_attribute_network_v33.pkl"):
    #     attribute_network.load_state_dict(torch.load("./models/zsl_awa2_attribute_network_v33.pkl"))
    #     print("load attribute network success")
    # if os.path.exists("./models/zsl_awa2_relation_network_v33.pkl"):
    #     relation_network.load_state_dict(torch.load("./models/zsl_awa2_relation_network_v33.pkl"))
    #     print("load relation network success")

    print("training...")
    last_accuracy = 0.0

    for episode in range(EPISODE):
        attribute_network_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)

        batch_features,batch_labels = train_loader.__iter__().next()

        sample_labels = []
        for label in batch_labels.numpy():
            if label not in sample_labels:
                sample_labels.append(label)
        # pdb.set_trace()
        
        sample_attributes = torch.Tensor([all_attributes[i] for i in sample_labels]).squeeze(1)
        class_num = sample_attributes.shape[0]

        batch_features = Variable(batch_features).cuda(GPU).float()  # 32*1024
        sample_features = attribute_network(Variable(sample_attributes).cuda(GPU)) #k*312


        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_SIZE,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(class_num,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        
        #print(sample_features_ext)
        #print(batch_features_ext)
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,4096)
        # pdb.set_trace()
        relations = relation_network(relation_pairs).view(-1,class_num)
        #print(relations)

        # re-build batch_labels according to sample_labels
        sample_labels = np.array(sample_labels)
        re_batch_labels = []
        for label in batch_labels.numpy():
            index = np.argwhere(sample_labels==label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)
        # pdb.set_trace()
        

        # loss
        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(BATCH_SIZE, class_num).scatter_(1, re_batch_labels.view(-1,1), 1)).cuda(GPU)
        loss = mse(relations,one_hot_labels)
        # pdb.set_trace()

        # update
        attribute_network.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        attribute_network_optim.step()
        relation_network_optim.step()

        if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.data[0])

        if (episode+1)%2000 == 0:
            # test
            print("Testing...")

            def compute_accuracy(test_features,test_label,test_id,test_attributes):
                
                test_data = TensorDataset(test_features,test_label)
                test_batch = 32
                test_loader = DataLoader(test_data,batch_size=test_batch,shuffle=False)
                total_rewards = 0
                # fetch attributes
                # pdb.set_trace()

                sample_labels = test_id
                sample_attributes = test_attributes
                class_num = sample_attributes.shape[0]
                test_size = test_features.shape[0]
                
                print("class num:",class_num)
                
                for batch_features,batch_labels in test_loader:

                    batch_size = batch_labels.shape[0]

                    batch_features = Variable(batch_features).cuda(GPU).float()  # 32*1024
                    sample_features = attribute_network(Variable(sample_attributes).cuda(GPU).float())

                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1)
                    batch_features_ext = batch_features.unsqueeze(0).repeat(class_num,1,1)
                    batch_features_ext = torch.transpose(batch_features_ext,0,1)

                    relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,4096)
                    relations = relation_network(relation_pairs).view(-1,class_num)

                    # re-build batch_labels according to sample_labels

                    re_batch_labels = []
                    for label in batch_labels.numpy():
                        index = np.argwhere(sample_labels==label)
                        re_batch_labels.append(index[0][0])
                    re_batch_labels = torch.LongTensor(re_batch_labels)
                    # pdb.set_trace()


                    _,predict_labels = torch.max(relations.data,1)
    
                    rewards = [1 if predict_labels[j]==re_batch_labels[j] else 0 for j in range(batch_size)]
                    total_rewards += np.sum(rewards)
                test_accuracy = total_rewards/1.0/test_size

                return test_accuracy
            
            zsl_accuracy = compute_accuracy(test_features,test_label,test_id,test_attributes)
            gzsl_unseen_accuracy = compute_accuracy(test_features,test_label,np.arange(50),attributes)
            gzsl_seen_accuracy = compute_accuracy(test_seen_features,test_seen_label,np.arange(50),attributes)
            
            H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)
            
            print('zsl:', zsl_accuracy)
            print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_seen_accuracy, gzsl_unseen_accuracy, H))
            

            if zsl_accuracy > last_accuracy:

                # save networks
                torch.save(attribute_network.state_dict(),"./models/zsl_awa2_attribute_network_v33.pkl")
                torch.save(relation_network.state_dict(),"./models/zsl_awa2_relation_network_v33.pkl")

                print("save networks for episode:",episode)

                last_accuracy = zsl_accuracy



if __name__ == '__main__':
    main()