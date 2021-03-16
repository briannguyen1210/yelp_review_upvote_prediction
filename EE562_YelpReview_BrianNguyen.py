#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 20:45:38 2020

@author: Brian Nguyen
"""

import json
import pandas as pd
import numpy as np
import string
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

#initialization for later uses
HIDDEN_SIZE = 16
NUM_CLASSES = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30
NUM_FEATURES = 5

#This function is to read the check-in json data to get the businessId and NumCheckIns
def read_checkin_data(filename):
    checkins = []
    
    with open(filename) as file:
        
        for line in file:
            checkin = json.loads(line)
            
            #add the number of checkin_info for the number of check-in
            numCheckins = 0
            for checkin_info in checkin['checkin_info']:
                numCheckins += checkin['checkin_info'][checkin_info]
            
            checkins.append({
                'BusinessId': checkin['business_id'],
                'NumCheckins': numCheckins
            })
            
    return pd.DataFrame(checkins)

#This function is to read the user json data to get the userId and the average votes per review of a user 
def read_user_data(filename):
    users = []
    
    with open(filename) as file:
        
        for line in file:
            user = json.loads(line)
            #sum the upvotes and get the average votes per review of a user
            sumVotes = user['votes']['funny'] + user['votes']['useful'] + user['votes']['cool']
            averageVotePerReview = sumVotes / user['review_count']
            
            users.append({
                'UserId': user['user_id'],
                'UserAvgVotePerReview': averageVotePerReview
            })
            
    return pd.DataFrame(users)

#This function is to read the business json data to get businessID, businessReviewCount, and BusinessOpen
def read_business_data(filename):
    businesses = []
    
    with open(filename) as file:
        
        for line in file:
            business = json.loads(line)
            
            #if a business is open, it's 1, else 0
            isOpen = 0
            if business['open']:
                isOpen = 1
            
            businesses.append({
                'BusinessId': business['business_id'],
                'BusinessReviewCount': business['review_count'],
                'BusinessOpen': isOpen
            })
            
    return pd.DataFrame(businesses)

#This function is to read the review json file to get the userID, businessId, star rating of a review,
#the length of a review, the number of key words per review (determined by the list of key words I created), 
#and the vote class (5 different classes for the number of upvotes)
def read_review_data(filename):
    keyWords = {'good', 'place', 'food', 'time', 'service', 'go', 'dont', 'nice', 'love', 'other', 'little',
                'do', 'much', 'try', 'chicken', 'menu', 'restaurant', 'order', 'know', 'bar', 'didnt', 
                'way', 'staff', 'lunch', 'pizza', 'delicious', 'few', 'cheese', 'fresh', 'salad', 'come',
                'new', 'eat', 'happy', 'sure', 'sauce', 'wait', 'take', 'area', 'experience', 'but', 'next',
                'cant', 'everything', 'bad', 'location', 'meal', 'table', 'small', 'last', 'big', 'hot',
                'awesome', 'favorite', 'home', 'hour', 'sandwich', 'burger', 'price', 'tasty', 'beer', 'sweet',
                'breakfast', 'meat', 'old', 'different', 'recommend', 'excellent', 'flavor', 'clean', 'perfect',
                'ok', 'ill', 'free', 'friend', 'server', 'quality', 'patio', 'customer'}
    reviews = []
    
    with open(filename) as file:
        
        for line in file:
            review = json.loads(line)
            
            #split the review to a list of words and remove punctuation
            words = review['text'].split()
            trans_table = str.maketrans('', '', string.punctuation)
            words = [w.translate(trans_table) for w in words]
            
            numKeyWords = 0
            
            #find number of key words per review
            for word in words:
                if word.lower() in keyWords:
                    numKeyWords += 1
            
            #convert number of votes into classes
            numVotes = review['votes']['funny'] + review['votes']['useful'] + review['votes']['cool']
            
            voteClass = -1
            if numVotes == 0:
                voteClass = 0
            elif numVotes > 0 and numVotes <= 5:
                voteClass = 1
            elif numVotes > 5 and numVotes <= 10:
                voteClass = 2
            elif numVotes > 10 and numVotes <= 15:
                voteClass = 3
            else:
                voteClass = 4
            
            reviews.append({
                'UserId': review['user_id'],
                'BusinessId': review['business_id'],
                'StarRating': review['stars'],
                'ReviewLength': len(review['text']),
                'NumKeyWords': numKeyWords,
                'VoteClass': voteClass
            })
        
    return pd.DataFrame(reviews)

#Neural Network with 2 hidden layers and 1 output layer
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size * 2)
        self.layer3 = nn.Linear(hidden_size * 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

print("Reading Data Files")

#read the json data and convert it to a Pandas DataFrame
review_df = read_review_data('./yelp_training_set/yelp_training_set_review.json')
user_df = read_user_data('./yelp_training_set/yelp_training_set_user.json')
business_df = read_business_data('./yelp_training_set/yelp_training_set_business.json')
checkin_df = read_checkin_data('./yelp_training_set/yelp_training_set_checkin.json')

#merge them all in one dataframe
input_df = review_df.merge(user_df, how='outer')
input_df = input_df.merge(business_df, how='outer')
input_df = input_df.merge(checkin_df, how='outer')
input_df.fillna(0, inplace=True)

print("Data Files Read")

#the list of features for consideration
feature_list = ['StarRating', 'ReviewLength', 'NumKeyWords', 'UserAvgVotePerReview',
                'BusinessReviewCount', 'BusinessOpen', 'NumCheckins']

combinations = itertools.combinations(feature_list, NUM_FEATURES)
feature_combinations = list(combinations)

print(f"Num of feature combinations: {len(feature_combinations)}")

#NBAcc is Naive Bayes Accuracy and NNAcc is Neural Network Accuracy
results = pd.DataFrame(columns=['Features', 'NBAcc', 'NNAcc'])

#loop through each combination
for combination in feature_combinations:
    features = list(combination)

    print(f"Features: {features}")
    
    x = input_df[features]
    y = input_df[['VoteClass']]
    
    INPUT_SIZE = len(x.columns)
    
    #split the data to train and test 80/20
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

    # Normalize features
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    
    # Naive-Bayes
    model_nb = GaussianNB().fit(x_train, y_train.values.ravel())
    gnb_predictions = model_nb.predict(x_test)
    accuracy_nb = model_nb.score(x_test, y_test) * 100.0
    
    print(f"    NAIVE-BAYES | Acc: {accuracy_nb:.5f}")

    # validation data for NN
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)

    x_train_tensor = torch.Tensor(x_train)
    y_train_tensor = torch.LongTensor(y_train.values).reshape(-1)

    x_val_tensor = torch.Tensor(x_val)
    y_val_tensor = torch.LongTensor(y_val.values).reshape(-1)

    x_test_tensor = torch.Tensor(x_test)
    y_test_tensor = torch.LongTensor(y_test.values).reshape(-1)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Neural Network
    neuralNetwork = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(neuralNetwork.parameters(), lr=LEARNING_RATE)

    print("    Training Neural Network")

    # Neural Network Training
    for epoch in range(NUM_EPOCHS):
        
        train_epoch_loss = 0
        neuralNetwork.train()
        
        for i, (data, labels) in enumerate(train_loader):
            
            # forward
            outputs = neuralNetwork(data)
            
            loss = criterion(outputs, labels)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
                
        with torch.no_grad():
            
            val_epoch_loss = 0
            neuralNetwork.eval()
            
            for data, labels in val_loader:
                outputs = neuralNetwork(data)
                
                val_loss = criterion(outputs, labels)
                val_epoch_loss += val_loss.item()

            if (epoch + 1) % 5 == 0:
                print(f'        Epoch: {epoch+1:02}/{NUM_EPOCHS} | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f}')

    # TESTING
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        
        for data, labels in test_loader:
            outputs = neuralNetwork(data)
            
            _, predictions = torch.max(outputs, 1) # returns value, index (index is the label 0-4)
            num_samples += labels.shape[0]
            num_correct += (predictions == labels).sum().item()
            
        accuracy_nn = 100.0 * num_correct / num_samples
        error_nn = 100.0 * (num_samples - num_correct) / num_samples
        print(f'    NEURAL NETWORK | Acc = {accuracy_nn:.5f} | Err = {error_nn:.5f}')

    row = pd.DataFrame([[features, accuracy_nb, accuracy_nn]], columns=['Features', 'NBAcc', 'NNAcc'])
    results = results.append(row)

print(results)
results.to_csv(path_or_buf=f"Results_{NUM_FEATURES}_Features.csv", index=False)