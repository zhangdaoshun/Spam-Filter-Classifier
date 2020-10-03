import argparse
import pandas as pd
import sys
import os
import math


def argumentParsing():
    parser = argparse.ArgumentParser(add_help = False, description = "Usage: python q2_classifier.py -f1 <train_dataset> -f2 <test_dataset> -o <output_file>")
    parser.add_argument("-f1", metavar = "train_dataset")
    parser.add_argument("-f2", metavar = "test_dataset")
    parser.add_argument("-o", metavar = "output_file")
    args = parser.parse_args()
    return args.f1, args.f2, args.o

if __name__ == '__main__':
    train_dataset, test_dataset, output_file = argumentParsing()
    if not train_dataset:
        sys.exit(0)

    if not test_dataset:
        sys.exit(0)

    if not output_file:
        sys.exit(0)

    
    df = pd.DataFrame(data = pd.read_csv(train_dataset, header = None))
    dataframe_tmp = df[0].str.split(' ')
    
    dataframe = pd.DataFrame(index = range(len(dataframe_tmp)), columns = ['id', 'classification', 'text'])

    for i in range(len(dataframe_tmp)):
        dataframe['id'][i] = dataframe_tmp[i][0]
        dataframe['classification'][i] = dataframe_tmp[i][1]
        dataframe['text'][i] = dataframe_tmp[i][2:]

    #dict to store frequencies
    freqTbl = {}
    countHam = 0
    countSpam = 0
    #initializing dictionaries
    for i in range(0,len(dataframe['text'])):
        data = dataframe['text'][i]
        for j in range(0, len(data), 2):
            if freqTbl.get(data[j]) == None:
                freqTbl[data[j]] = (0,0)
            if dataframe['classification'][i] == 'ham':
                freqTbl[data[j]] = (freqTbl[data[j]][0], freqTbl[data[j]][1]+int(data[j+1]))
            else:
                freqTbl[data[j]] = (freqTbl[data[j]][0]+int(data[j+1]), freqTbl[data[j]][1])

    #getting the total count 
    for i in dataframe['classification']:
        if i == 'spam':
            countSpam += 1
        else:
            countHam += 1


    #calculating the Conditional Probability
    total = countHam + countSpam
    prior={}
    conditionalProb = {}
    probSpam = float(countSpam)/total
    probHam = float(countHam)/total
    V = len(dataframe)
    for k,v in freqTbl.iteritems():
        tmp = v[1]/float(total)
        prior[k] = (1-tmp, tmp)
        conditionalProb[k] = ((v[0] + 1)/(float(countSpam) + V*V), (v[1] + 1)/(float(countHam) + V*V))

    #Loading test data    
    
    df = pd.DataFrame(data = pd.read_csv(test_dataset, header = None))
    dataframe_tmp = df[0].str.split(' ')

    

    dataframeTest = pd.DataFrame(index = range(len(dataframe_tmp)), columns = ['id', 'classification', 'text'])

    
    for i in range(len(dataframe_tmp)):
        # dataframe['id'][i] = dataframe[i][0]
        dataframeTest['id'][i] = dataframe_tmp[i][0]
        dataframeTest['classification'][i] = dataframe_tmp[i][1]
        dataframeTest['text'][i] = dataframe_tmp[i][2:]

    label = list()
    correct = 0
    tempS = 0
    tempH = 0

    
    #It iterates over each entry in the test set. For each entry it takes a word, its corresponding
    #count and its previously calculated conditionalProb. Then the multinomial naive bayes formula is appplied to calculate
    #P(spam/ham | word). 
    #As the conditionalProb are very small, python gave error during taking power, so we took the log.
    

    for i in range(len(dataframeTest['text'])):
        tempS = 1
        tempH = 1

        id = dataframeTest['id'][i]
        for j in range(0, len(dataframeTest['text'][i]), 2):
            word = dataframeTest['text'][i][j]
            probS = conditionalProb[word][0]
            probH = conditionalProb[word][1]

            power = dataframeTest['text'][i][j+1]
            
            if probS != 0:
                term1 = math.log(probS)*int(power)
            else:
                term1 = 0
            if probH != 0:
                term2 = math.log(probH)*int(power)
            else:
                term2 = 0

            
            tempS = tempS + term1
            tempH = tempH + term2

        tempS = tempS + math.log(probSpam)
        tempH = tempH + math.log(probHam)

        if tempS > tempH:
            label.append([id, 'spam'])
            if dataframeTest['classification'][i] == 'spam':
                correct += 1
        else:
            label.append([id, 'ham'])
            if dataframeTest['classification'][i] == 'ham':
                correct += 1


    accuracy = float(correct) / len(dataframeTest.index)
    df_out = pd.DataFrame(label)
    df_out.to_csv(output_file, index=False, header = False, sep =' ')
    print accuracy
