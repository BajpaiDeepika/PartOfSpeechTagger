
###################################
# CS B551 Fall 2015, Assignment #5
#
# Deepika Bajpai
# UserId:dbajpai
#
# (Based on skeleton code by D. Crandall)
#
#
####
#******************************************Report************************************************************************************
# Step 1: Learning -
# Part 1
# First we  calculated the initial Probabilities i.e for each 12 part of speech, the probability that the sentence will start with
# them.For this we will calculate the total no of sentences in training data and how many times the first word is one of the
# given part of speech.
# Part 2:
# Find the self.transition probabilities from training data.i.e count for each part of speech tag,how many times,it is followed by other part of speech.
# ex: For all Nouns, see how many times Verb Follows a noun, i.e P(Sn+1=Verb| Sn=Noun).. so for 12 part of speeches, there will be 12*12
# self.transition probabilities.We stored them in a dictionary data structure i.e for each part of speech as key-store 12 transition probabilities as a list
# Part 3:
# Find the emission probabilities-i.e for enoun(word)= Count the number of times the word is noun/Total number of times the word occurs.
# Similarly calculate the emmission probability of all test words ex if there are 20 words in test set, 20*12 emission probabilities needs to be calculated
# We stored them also in a dictionary data structure.
#
# Step 2:Naive inference-
# s*i= arg maxsi P(Si = si|W):
# Select that part of speech for each word , which has the highest emission probability.ex: for a sentence having 20 words, for 
# the first word, see which POS tag has the highest emission probability .
# Step 3: MCMC:
# I randomly selected the first sample as all Nouns for all part of speeches.
# Then it will randomly select any part of speech and sample it .It will calculate the 12 possibilities for the next value as calculating
# the emisission probability of sampled word*all transition probabilities for the sentence.I created a method for creating 100 samples per sentence and 
# passing it to MCMC method from which it will select 5 samples.
# Step 4: Max Parginal:
# I reused the sample generated in MCMC for this step and used the maximum likelihood method for calculating the probability.
#
# Step 4:MAP:
# For Viterbi algorithm for T0 state, I calculated the initialProbability of each part of speech multiplued by its emission probability
# Then for no. of states=length of sentence:I calculated the max probability of transition from all 12 previous states multiplied by the emission probability
# of observed word.
# In the end the highest probability tag is then assigned to the word of sentences
#
# Step 6: Best: My best scoring algorithm is so far the naive approach.
#
# 
# 
#                 (ii) Evaluation of bc.test file:
#For bc.test file , the program scored 2000 sentences with 29442 words.
#                    Words correct:     Sentences correct: 
#    0. Ground truth:      100.00%              100.00%
#           1. Naive:       92.96%               41.95%
#         2. Sampler:       62.84%                0.05%
#    3. Max marginal:       70.97%                0.05%
#             4. MAP:       91.97%               38.30%
#            5. Best:       92.96%               41.95%
#
#                  (iii)Problems/Assumption/Simplifications
#  Zero Probabilities: Wherever the probabilities were becoming zero-I normalized them by adding 0.0000001 to them ex: in Vitebi
#  Speed:The speed was a major issue. So I calculated all the transition,initial and emission probabilities 
#        in the train method itself rather than calculating them in respective algorithm functions.
#  I also I reused the samples generated in MCMC to MAX marginal rather then regenarting them.
#  Choice of data structure: I tried to use dictionary and list data structure wherever possible.
#
#
#***********************************************************************************************************************************************************************************

import random
import math
from collections import defaultdict

class Solver:
    def __init__(self):
        self.word_scorecard = {}
        self.sentence_scorecard = {}
        self.word_count = 0
        self.sentence_count = 0
        self.Emission={}
        self.TagsListOrder=['adj','noun','adv','adp','num','pron','verb','.','x','conj','prt','det']
        self.trans={}
        self.InitProb={}
        self.SampleEmisssion={}
        self.SampleEmisssion=defaultdict(lambda: [0,0,0,0,0,0,0,0,0,0,0,0],self.SampleEmisssion)

    #order is [adj,noun,adv,adp,num,pron,verb,dot,x,conj,prt,det]
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        #prob(noun|word)
        # log( P(det, noun, verb | the dog runs) )=log(P(Q0=noun)+log P(noun|det)+logP(verb|noun)+logP(the|det)+logP(dog|noun)+logP(runs|verb)
        #=log(initial prob of observed tag)+log(all transition Prob)+log(all emission Prob of all words)
        posttransprob=.00000001
        InitialProbability=.00000001
        EmissionProb=.00000001
        #for i in range:
        if (label[0]=="noun"):
            InitialProbability=self.InitProb['noun']
        elif (label[0]=="verb"):
            InitialProbability=self.InitProb['verb']
        elif (label[0]=="adj"):
            InitialProbability=self.InitProb['adj']
        elif (label[0]=="adp"):
            InitialProbability=self.InitProb['adp']
        elif (label[0]=="prt"):
            InitialProbability=self.InitProb['prt']
        elif (label[0]=="det"):
            InitialProbability=self.InitProb['det']
        elif (label[0]=="."):
            InitialProbability=self.InitProb['.']
        elif (label[0]=="x"):
            InitialProbability=self.InitProb['x']
        elif (label[0]=="num"):
            InitialProbability=self.InitProb['num']
        elif (label[0]=="adv"):
            InitialProbability=self.InitProb['adv']
        elif (label[0]=="conj"):
            InitialProbability=self.InitProb['conj']
        elif (label[0]=="pron"):
            InitialProbability=self.InitProb['pron']
                #log of transition Probabilities
        for i in range(0,len(sentence)-1):     #[adj,noun,adv,adp,num,pron,verb,dot,x,conj,prt,det]
            if (label[i+1]=="adj"):
                posttransprob=(self.trans[label[i]][0]+.00000001)*posttransprob
            elif(label[i+1]=="noun"):
                posttransprob=(self.trans[label[i]][1]+.00000001)*posttransprob
            elif (label[i+1]=="adv"):
                posttransprob=(self.trans[label[i]][2]+.00000001)*posttransprob
            elif (label[i+1]=="adp"):
                posttransprob=(self.trans[label[i]][3]+.00000001)*posttransprob
            elif (label[i+1]=="num"):
                posttransprob=(self.trans[label[i]][4]+.00000001)*posttransprob
            elif (label[i+1]=="pron"):
                posttransprob=(self.trans[label[i]][5]+.00000001)*posttransprob
            elif (label[i+1]=="verb"):
                posttransprob=(self.trans[label[i]][6]+.00000001)*posttransprob
            elif (label[i+1]=="dot"):
                posttransprob=(self.trans[label[i]][7]+.00000001)*posttransprob
            elif (label[i+1]=="x"):
                posttransprob=(self.trans[label[i]][8]+.00000001)*posttransprob
            elif (label[i+1]=="conj"):
                posttransprob=(self.trans[label[i]][9]+.00000001)*posttransprob
            elif (label[i+1]=="prt"):
                posttransprob=(self.trans[label[i]][10]+.00000001)*posttransprob
            elif (label[i+1]=="det"):
                posttransprob=(self.trans[label[i]][11]+.00000001)*posttransprob
                
        #log of emission probabilities:
        for j in range(0,len(sentence)):
            if (sentence[j]=="adj"):
                EmissionProb=EmissionProb*(self.Emission[sentence[j]][0]+.00000001)
            elif (sentence[j]=="noun"):
                EmissionProb=EmissionProb*(self.Emission[sentence[j]][1]+.00000001)
            elif (sentence[j]=="adv"):
                EmissionProb=EmissionProb*(self.Emission[sentence[j]][2]+.00000001)
            elif (sentence[j]=="adp"):
                EmissionProb=EmissionProb*(self.Emission[sentence[j]][3]+.00000001)
            elif (sentence[j]=="num"):
                EmissionProb=EmissionProb*(self.Emission[sentence[j]][4]+.00000001)
            elif (sentence[j]=="pron"):
                EmissionProb=EmissionProb*(self.Emission[sentence[j]][5]+.00000001)
            elif (sentence[j]=="verb"):
                EmissionProb=EmissionProb*(self.Emission[sentence[j]][6]+.00000001)
            elif (sentence[j]=="."):
                EmissionProb=EmissionProb*(self.Emission[sentence[j]][7]+.00000001)
            elif (sentence[j]=="x"):
                EmissionProb=EmissionProb*(self.Emission[sentence[j]][8]+.00000001)
            elif (sentence[j]=="conj"):
                EmissionProb=EmissionProb*(self.Emission[sentence[j]][9]+.00000001)
            elif (sentence[j]=="prt"):
                EmissionProb=EmissionProb*(self.Emission[sentence[j]][10]+.00000001)
            elif (sentence[j]=="det"):
                EmissionProb=EmissionProb*(self.Emission[sentence[j]][11]+.00001000)
        

        
#             
        #print InitialProb, posttransprob,LogEmissionProb
        post=math.log(InitialProbability+0.00000001)+math.log(posttransprob)+math.log(EmissionProb)
        return post
        #return 0
    # Do the training!
    #
    def train(self, data):
        
#cnoun is number of times noun is the first POS
        self.cnoun=0
        self.cadj=0
        self.cadv=0
        self.cadp=0
        self.cconj=0
        self.cdet=0
        self.cnum=0
        self.cpron=0
        self.cprt=0
        self.cverb=0
        self.cx=0
        self.cdot=0
        for i in xrange(0,len(data)):
            if (data[i][1][0]=="adj"):
                self.cadj=self.cadj+1
            elif(data[i][1][0]=="noun"):
                self.cnoun=self.cnoun+1
            elif(data[i][1][0]=="adv"):
                self.cadv=self.cadv+1
            elif(data[i][1][0]=="adp"):
                self.cadp=self.cadp+1
            elif(data[i][1][0]=="conj"):
                self.cconj=self.cconj+1
            elif(data[i][1][0]=="det"):
                self.cdet=self.cdet+1
            elif(data[i][1][0]=="prt"):
                self.cprt=self.cprt+1
            elif(data[i][1][0]=="verb"):
                self.cverb=self.cverb+1
            elif(data[i][1][0]=="x"):
                self.cx=self.cx+1
            elif(data[i][1][0]=="." ):
                self.cdot=self.cdot+1
            elif(data[i][1][0]=="pron"):
                self.cpron=self.cpron+1
            elif(data[i][1][0]=="num"):
                self.cnum=self.cnum+1
        #print cnoun
        #Initial Probabilities of all POS        
        self.InitProb['noun']=float(self.cnoun)/len(data)
        self.InitProb['adj']=float(self.cadj)/len(data)
        self.InitProb['adv']=float(self.cadv)/len(data)
        self.InitProb['adp']=float(self.cadp)/len(data)
        self.InitProb['conj']=float(self.cconj)/len(data)
        self.InitProb['det']=float(self.cdet)/len(data)
        self.InitProb['num']=float(self.cnum)/len(data)
        self.InitProb['pron']=float(self.cpron)/len(data)
        self.InitProb['prt']=float(self.cprt)/len(data)
        self.InitProb['verb']=float(self.cverb)/len(data)
        self.InitProb['x']=float(self.cx)/len(data)
        self.InitProb['.']=float(self.cdot)/len(data)
        #print self.Wnoun


        # self.transition Probabilities-count for each part of speech tag,how many times,it is followed by other part of speech
#         1.Calculate how many times each POS occurs in training file
        self.Tadj=0
        self.Tnoun=0
        self.Tadv=0
        self.Tadp=0
        self.Tdet=0
        self.Tconj=0
        self.Tprt=0
        self.Tverb=0
        self.Tx=0
        self.Tdot=0
        self.Tpron=0
        self.Tnum=0
        
        for i in xrange(0,len(data)):
            for j in xrange(0,len(data[i][1])):
                if (data[i][1][j]=="adj"):
                    self.Tadj=self.Tadj+1
                elif (data[i][1][j]=="noun"):
                    self.Tnoun=self.Tnoun+1
                elif (data[i][1][j]=="adv"):
                    self.Tadv=self.Tadv+1
                elif (data[i][1][j]=="adp"):
                    self.Tadp=self.Tadp+1
                elif (data[i][1][j]=="conj"):
                    self.Tconj=self.Tconj+1
                elif (data[i][1][j]=="det"):
                    self.Tdet=self.Tdet+1
                elif (data[i][1][j]=="prt"):
                    self.Tprt=self.Tprt+1
                elif (data[i][1][j]=="verb"):
                    self.Tverb=self.Tverb+1
                elif (data[i][1][j]=="x"):
                    self.Tx=self.Tx+1
                elif (data[i][1][j]=="."):
                    self.Tdot=self.Tdot+1
                elif (data[i][1][j]=="pron"):
                    self.Tpron=self.Tpron+1
                elif (data[i][1][j]=="num"):
                    self.Tnum=self.Tnum+1

#         print Tadj,Tnoun,Tadv,Tadp,Tnum,Tpron,Tverb,Tdot,Tx,Tconj,Tprt,Tdet-total no of occurances
#          2.Calculate how many times each POS is followed by every other 11 POS
        #print data[0][1][3]
        #order is [adj,noun,adv,adp,num,pron,verb,dot,x,conj,prt,det]
        self.trans["adj"]=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.trans["noun"]=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.trans["adv"]=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.trans["adp"]=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.trans["det"]=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.trans["conj"]=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.trans["prt"]=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.trans["verb"]=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.trans["x"]=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.trans["."]=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.trans["pron"]=[0,0,0,0,0,0,0,0,0,0,0,0]
        self.trans["num"]=[0,0,0,0,0,0,0,0,0,0,0,0]
        for k in xrange(0,len(data)):
            for l in xrange(0,len(data[k][1])-1):
                if (data[k][1][l]=="adj"):
                    if (data[k][1][l+1]=="adj"):
                        self.trans["adj"][0]=self.trans["adj"][0]+1
                    elif (data[k][1][l+1]=="noun"):
                        self.trans["adj"][1]=self.trans["adj"][1]+1
                    elif (data[k][1][l+1]=="adv"):
                        self.trans["adj"][2]=self.trans["adj"][2]+1
                    elif (data[k][1][l+1]=="adp"):
                        self.trans["adj"][3]=self.trans["adj"][3]+1
                    elif (data[k][1][l+1]=="num"):
                        self.trans["adj"][4]=self.trans["adj"][4]+1
                    elif (data[k][1][l+1]=="pron"):
                        self.trans["adj"][5]=self.trans["adj"][5]+1
                    elif (data[k][1][l+1]=="verb"):
                        self.trans["adj"][6]=self.trans["adj"][6]+1
                    elif (data[k][1][l+1]=="."):
                        self.trans["adj"][7]=self.trans["adj"][7]+1
                    elif (data[k][1][l+1]=="x"):
                        self.trans["adj"][8]=self.trans["adj"][8]+1
                    elif (data[k][1][l+1]=="conj"):
                        self.trans["adj"][9]=self.trans["adj"][9]+1
                    elif (data[k][1][l+1]=="prt"):
                        self.trans["adj"][10]=self.trans["adj"][10]+1
                    elif (data[k][1][l+1]=="det"):
                        self.trans["adj"][11]=self.trans["adj"][11]+1
                if (data[k][1][l]=="noun"):
                    if (data[k][1][l+1]=="adj"):
                        self.trans["noun"][0]=self.trans["noun"][0]+1
                    elif (data[k][1][l+1]=="noun"):
                        self.trans["noun"][1]=self.trans["noun"][1]+1
                    elif (data[k][1][l+1]=="adv"):
                        self.trans["noun"][2]=self.trans["noun"][2]+1
                    elif (data[k][1][l+1]=="adp"):
                        self.trans["noun"][3]=self.trans["noun"][3]+1
                    elif (data[k][1][l+1]=="num"):
                        self.trans["noun"][4]=self.trans["noun"][4]+1
                    elif (data[k][1][l+1]=="pron"):
                        self.trans["noun"][5]=self.trans["noun"][5]+1
                    elif (data[k][1][l+1]=="verb"):
                        self.trans["noun"][6]=self.trans["noun"][6]+1
                    elif (data[k][1][l+1]=="."):
                        self.trans["noun"][7]=self.trans["noun"][7]+1
                    elif (data[k][1][l+1]=="x"):
                        self.trans["noun"][8]=self.trans["noun"][8]+1
                    elif (data[k][1][l+1]=="conj"):
                        self.trans["noun"][9]=self.trans["noun"][9]+1
                    elif (data[k][1][l+1]=="prt"):
                        self.trans["noun"][10]=self.trans["noun"][10]+1
                    elif (data[k][1][l+1]=="det"):
                        self.trans["noun"][11]=self.trans["noun"][11]+1
                #3.adv
                if (data[k][1][l]=="adv"):
                    if (data[k][1][l+1]=="adj"):
                        self.trans["adv"][0]=self.trans["adv"][0]+1
                    elif (data[k][1][l+1]=="noun"):
                        self.trans["adv"][1]=self.trans["adv"][1]+1
                    elif (data[k][1][l+1]=="adv"):
                        self.trans["adv"][2]=self.trans["adv"][2]+1
                    elif (data[k][1][l+1]=="adp"):
                        self.trans["adv"][3]=self.trans["adv"][3]+1
                    elif (data[k][1][l+1]=="num"):
                        self.trans["adv"][4]=self.trans["adv"][4]+1
                    elif (data[k][1][l+1]=="pron"):
                        self.trans["adv"][5]=self.trans["adv"][5]+1
                    elif (data[k][1][l+1]=="verb"):
                        self.trans["adv"][6]=self.trans["adv"][6]+1
                    elif (data[k][1][l+1]=="."):
                        self.trans["adv"][7]=self.trans["adv"][7]+1
                    elif (data[k][1][l+1]=="x"):
                        self.trans["adv"][8]=self.trans["adv"][8]+1
                    elif (data[k][1][l+1]=="conj"):
                        self.trans["adv"][9]=self.trans["adv"][9]+1
                    elif (data[k][1][l+1]=="prt"):
                        self.trans["adv"][10]=self.trans["adv"][10]+1
                    elif (data[k][1][l+1]=="det"):
                        self.trans["adv"][11]=self.trans["adv"][11]+1
                #4.adp
                if (data[k][1][l]=="adp"):
                    if (data[k][1][l+1]=="adj"):
                        self.trans["adp"][0]=self.trans["adp"][0]+1
                    elif (data[k][1][l+1]=="noun"):
                        self.trans["adp"][1]=self.trans["adp"][1]+1
                    elif (data[k][1][l+1]=="adv"):
                        self.trans["adp"][2]=self.trans["adp"][2]+1
                    elif (data[k][1][l+1]=="adp"):
                        self.trans["adp"][3]=self.trans["adp"][3]+1
                    elif (data[k][1][l+1]=="num"):
                        self.trans["adp"][4]=self.trans["adp"][4]+1
                    elif (data[k][1][l+1]=="pron"):
                        self.trans["adp"][5]=self.trans["adp"][5]+1
                    elif (data[k][1][l+1]=="verb"):
                        self.trans["adp"][6]=self.trans["adp"][6]+1
                    elif (data[k][1][l+1]=="."):
                        self.trans["adp"][7]=self.trans["adp"][7]+1
                    elif (data[k][1][l+1]=="x"):
                        self.trans["adp"][8]=self.trans["adp"][8]+1
                    elif (data[k][1][l+1]=="conj"):
                        self.trans["adp"][9]=self.trans["adp"][9]+1
                    elif (data[k][1][l+1]=="prt"):
                        self.trans["adp"][10]=self.trans["adp"][10]+1
                    elif (data[k][1][l+1]=="det"):
                        self.trans["adp"][11]=self.trans["adp"][11]+1
                #5.det
                if (data[k][1][l]=="det"):
                    if (data[k][1][l+1]=="adj"):
                        self.trans["det"][0]=self.trans["det"][0]+1
                    elif (data[k][1][l+1]=="noun"):
                        self.trans["det"][1]=self.trans["det"][1]+1
                    elif (data[k][1][l+1]=="adv"):
                        self.trans["det"][2]=self.trans["det"][2]+1
                    elif (data[k][1][l+1]=="adp"):
                        self.trans["det"][3]=self.trans["det"][3]+1
                    elif (data[k][1][l+1]=="num"):
                        self.trans["det"][4]=self.trans["det"][4]+1
                    elif (data[k][1][l+1]=="pron"):
                        self.trans["det"][5]=self.trans["det"][5]+1
                    elif (data[k][1][l+1]=="verb"):
                        self.trans["det"][6]=self.trans["det"][6]+1
                    elif (data[k][1][l+1]=="."):
                        self.trans["det"][7]=self.trans["det"][7]+1
                    elif (data[k][1][l+1]=="x"):
                        self.trans["det"][8]=self.trans["det"][8]+1
                    elif (data[k][1][l+1]=="conj"):
                        self.trans["det"][9]=self.trans["det"][9]+1
                    elif (data[k][1][l+1]=="prt"):
                        self.trans["det"][10]=self.trans["det"][10]+1
                    elif (data[k][1][l+1]=="det"):
                        self.trans["det"][11]=self.trans["det"][11]+1
                #6.conj
                if (data[k][1][l]=="conj"):
                    if (data[k][1][l+1]=="adj"):
                        self.trans["conj"][0]=self.trans["conj"][0]+1
                    elif (data[k][1][l+1]=="noun"):
                        self.trans["conj"][1]=self.trans["conj"][1]+1
                    elif (data[k][1][l+1]=="adv"):
                        self.trans["conj"][2]=self.trans["conj"][2]+1
                    elif (data[k][1][l+1]=="adp"):
                        self.trans["conj"][3]=self.trans["conj"][3]+1
                    elif (data[k][1][l+1]=="num"):
                        self.trans["conj"][4]=self.trans["conj"][4]+1
                    elif (data[k][1][l+1]=="pron"):
                        self.trans["conj"][5]=self.trans["conj"][5]+1
                    elif (data[k][1][l+1]=="verb"):
                        self.trans["conj"][6]=self.trans["conj"][6]+1
                    elif (data[k][1][l+1]=="."):
                        self.trans["conj"][7]=self.trans["conj"][7]+1
                    elif (data[k][1][l+1]=="x"):
                        self.trans["conj"][8]=self.trans["conj"][8]+1
                    elif (data[k][1][l+1]=="conj"):
                        self.trans["conj"][9]=self.trans["conj"][9]+1
                    elif (data[k][1][l+1]=="prt"):
                        self.trans["conj"][10]=self.trans["conj"][10]+1
                    elif (data[k][1][l+1]=="det"):
                        self.trans["conj"][11]=self.trans["conj"][11]+1
                #7.Prt
                if (data[k][1][l]=="prt"):
                    if (data[k][1][l+1]=="adj"):
                        self.trans["prt"][0]=self.trans["prt"][0]+1
                    elif (data[k][1][l+1]=="noun"):
                        self.trans["prt"][1]=self.trans["prt"][1]+1
                    elif (data[k][1][l+1]=="adv"):
                        self.trans["prt"][2]=self.trans["prt"][2]+1
                    elif (data[k][1][l+1]=="adp"):
                        self.trans["prt"][3]=self.trans["prt"][3]+1
                    elif (data[k][1][l+1]=="num"):
                        self.trans["prt"][4]=self.trans["prt"][4]+1
                    elif (data[k][1][l+1]=="pron"):
                        self.trans["prt"][5]=self.trans["prt"][5]+1
                    elif (data[k][1][l+1]=="verb"):
                        self.trans["prt"][6]=self.trans["prt"][6]+1
                    elif (data[k][1][l+1]=="."):
                        self.trans["prt"][7]=self.trans["prt"][7]+1
                    elif (data[k][1][l+1]=="x"):
                        self.trans["prt"][8]=self.trans["prt"][8]+1
                    elif (data[k][1][l+1]=="conj"):
                        self.trans["prt"][9]=self.trans["prt"][9]+1
                    elif (data[k][1][l+1]=="prt"):
                        self.trans["prt"][10]=self.trans["prt"][10]+1
                    elif (data[k][1][l+1]=="det"):
                        self.trans["prt"][11]=self.trans["prt"][11]+1
                #8.Verb
                if (data[k][1][l]=="verb"):
                    if (data[k][1][l+1]=="adj"):
                        self.trans["verb"][0]=self.trans["verb"][0]+1
                    elif (data[k][1][l+1]=="noun"):
                        self.trans["verb"][1]=self.trans["verb"][1]+1
                    elif (data[k][1][l+1]=="adv"):
                        self.trans["verb"][2]=self.trans["verb"][2]+1
                    elif (data[k][1][l+1]=="adp"):
                        self.trans["verb"][3]=self.trans["verb"][3]+1
                    elif (data[k][1][l+1]=="num"):
                        self.trans["verb"][4]=self.trans["verb"][4]+1
                    elif (data[k][1][l+1]=="pron"):
                        self.trans["verb"][5]=self.trans["verb"][5]+1
                    elif (data[k][1][l+1]=="verb"):
                        self.trans["verb"][6]=self.trans["verb"][6]+1
                    elif (data[k][1][l+1]=="."):
                        self.trans["verb"][7]=self.trans["verb"][7]+1
                    elif (data[k][1][l+1]=="x"):
                        self.trans["verb"][8]=self.trans["verb"][8]+1
                    elif (data[k][1][l+1]=="conj"):
                        self.trans["verb"][9]=self.trans["verb"][9]+1
                    elif (data[k][1][l+1]=="prt"):
                        self.trans["verb"][10]=self.trans["verb"][10]+1
                    elif (data[k][1][l+1]=="det"):
                        self.trans["verb"][11]=self.trans["verb"][11]+1
                #9.X
                if (data[k][1][l]=="x"):
                    if (data[k][1][l+1]=="adj"):
                        self.trans["x"][0]=self.trans["x"][0]+1
                    elif (data[k][1][l+1]=="noun"):
                        self.trans["x"][1]=self.trans["x"][1]+1
                    elif (data[k][1][l+1]=="adv"):
                        self.trans["x"][2]=self.trans["x"][2]+1
                    elif (data[k][1][l+1]=="adp"):
                        self.trans["x"][3]=self.trans["x"][3]+1
                    elif (data[k][1][l+1]=="num"):
                        self.trans["x"][4]=self.trans["x"][4]+1
                    elif (data[k][1][l+1]=="pron"):
                        self.trans["x"][5]=self.trans["x"][5]+1
                    elif (data[k][1][l+1]=="verb"):
                        self.trans["x"][6]=self.trans["x"][6]+1
                    elif (data[k][1][l+1]=="."):
                        self.trans["x"][7]=self.trans["x"][7]+1
                    elif (data[k][1][l+1]=="x"):
                        self.trans["x"][8]=self.trans["x"][8]+1
                    elif (data[k][1][l+1]=="conj"):
                        self.trans["x"][9]=self.trans["x"][9]+1
                    elif (data[k][1][l+1]=="prt"):
                        self.trans["x"][10]=self.trans["x"][10]+1
                    elif (data[k][1][l+1]=="det"):
                        self.trans["x"][11]=self.trans["x"][11]+1
                #10.Dot
                if (data[k][1][l]=="dot"):
                    if (data[k][1][l+1]=="adj"):
                        self.trans["."][0]=self.trans["."][0]+1
                    elif (data[k][1][l+1]=="noun"):
                        self.trans["."][1]=self.trans["."][1]+1
                    elif (data[k][1][l+1]=="adv"):
                        self.trans["."][2]=self.trans["."][2]+1
                    elif (data[k][1][l+1]=="adp"):
                        self.trans["."][3]=self.trans["."][3]+1
                    elif (data[k][1][l+1]=="num"):
                        self.trans["."][4]=self.trans["."][4]+1
                    elif (data[k][1][l+1]=="pron"):
                        self.trans["."][5]=self.trans["."][5]+1
                    elif (data[k][1][l+1]=="verb"):
                        self.trans["."][6]=self.trans["."][6]+1
                    elif (data[k][1][l+1]=="."):
                        self.trans["."][7]=self.trans["."][7]+1
                    elif (data[k][1][l+1]=="x"):
                        self.trans["."][8]=self.trans["."][8]+1
                    elif (data[k][1][l+1]=="conj"):
                        self.trans["."][9]=self.trans["."][9]+1
                    elif (data[k][1][l+1]=="prt"):
                        self.trans["."][10]=self.trans["."][10]+1
                    elif (data[k][1][l+1]=="det"):
                        self.trans["."][11]=self.trans["."][11]+1
                #11.pron
                if (data[k][1][l]=="pron"):
                    if (data[k][1][l+1]=="adj"):
                        self.trans["pron"][0]=self.trans["pron"][0]+1
                    elif (data[k][1][l+1]=="noun"):
                        self.trans["pron"][1]=self.trans["pron"][1]+1
                    elif (data[k][1][l+1]=="adv"):
                        self.trans["pron"][2]=self.trans["pron"][2]+1
                    elif (data[k][1][l+1]=="adp"):
                        self.trans["pron"][3]=self.trans["pron"][3]+1
                    elif (data[k][1][l+1]=="num"):
                        self.trans["pron"][4]=self.trans["pron"][4]+1
                    elif (data[k][1][l+1]=="pron"):
                        self.trans["pron"][5]=self.trans["pron"][5]+1
                    elif (data[k][1][l+1]=="verb"):
                        self.trans["pron"][6]=self.trans["pron"][6]+1
                    elif (data[k][1][l+1]=="."):
                        self.trans["pron"][7]=self.trans["pron"][7]+1
                    elif (data[k][1][l+1]=="x"):
                        self.trans["pron"][8]=self.trans["pron"][8]+1
                    elif (data[k][1][l+1]=="conj"):
                        self.trans["pron"][9]=self.trans["pron"][9]+1
                    elif (data[k][1][l+1]=="prt"):
                        self.trans["pron"][10]=self.trans["pron"][10]+1
                    elif (data[k][1][l+1]=="det"):
                        self.trans["pron"][11]=self.trans["pron"][11]+1
                #12.num
                if (data[k][1][l]=="num"):
                    if (data[k][1][l+1]=="adj"):
                        self.trans["num"][0]=self.trans["num"][0]+1
                    elif (data[k][1][l+1]=="noun"):
                        self.trans["num"][1]=self.trans["num"][1]+1
                    elif (data[k][1][l+1]=="adv"):
                        self.trans["num"][2]=self.trans["num"][2]+1
                    elif (data[k][1][l+1]=="adp"):
                        self.trans["num"][3]=self.trans["num"][3]+1
                    elif (data[k][1][l+1]=="num"):
                        self.trans["num"][4]=self.trans["num"][4]+1
                    elif (data[k][1][l+1]=="pron"):
                        self.trans["num"][5]=self.trans["num"][5]+1
                    elif (data[k][1][l+1]=="verb"):
                        self.trans["num"][6]=self.trans["num"][6]+1
                    elif (data[k][1][l+1]=="."):
                        self.trans["num"][7]=self.trans["num"][7]+1
                    elif (data[k][1][l+1]=="x"):
                        self.trans["num"][8]=self.trans["num"][8]+1
                    elif (data[k][1][l+1]=="conj"):
                        self.trans["num"][9]=self.trans["num"][9]+1
                    elif (data[k][1][l+1]=="prt"):
                        self.trans["num"][10]=self.trans["num"][10]+1
                    elif (data[k][1][l+1]=="det"):
                        self.trans["num"][11]=self.trans["num"][11]+1

        #calculating actual self.transition probabilities:
        for m in range(0,12):
            self.trans["adj"][m]=float(self.trans["adj"][m])/float(self.Tadj)
            self.trans["noun"][m]=float(self.trans["noun"][m])/float(self.Tnoun)
            self.trans["adv"][m]=float(self.trans["adv"][m])/float(self.Tadv)
            self.trans["adp"][m]=float(self.trans["adp"][m])/float(self.Tadp)
            self.trans["det"][m]=float(self.trans["det"][m])/float(self.Tdet)
            self.trans["conj"][m]=float(self.trans["conj"][m])/float(self.Tconj)
            self.trans["prt"][m]=float(self.trans["prt"][m])/float(self.Tprt)
            self.trans["verb"][m]=float(self.trans["verb"][m])/float(self.Tverb)
            self.trans["."][m]=float(self.trans["."][m])/float(self.Tdot)
            self.trans["x"][m]=float(self.trans["x"][m])/float(self.Tx)
            self.trans["pron"][m]=float(self.trans["pron"][m])/float(self.Tpron)
            self.trans["num"][m]=float(self.trans["num"][m])/float(self.Tnum)
        #print self.trans["noun"][6]
        #print self.transAdj,self.transNoun,self.transAdv,self.transAdp,self.transDet,self.transConj,self.transPrt,self.transVerb,self.transDot,self.transX,self.transPron,self.transNum
        #calculate Emission probabilities of each word in the test file
        #print data[0][0][0],data[0][0][1]   #:the fulton    
        self.Emission={}
        for m in xrange(0,len(data)):
            for n in xrange(0,len(data[m][0])):
                self.Emission[data[m][0][n]]=[0,0,0,0,0,0,0,0,0,0,0,0]
#Emission[data[item][0][words]]=[]  #value is a list with 12 values adj,noun,adv,adp,num,pron,verb,dot,x,conj,prt,det
        for x in xrange(0,len(data)):
            for i in xrange(0,len(data[x][1])):
                if (data[x][1][i]=="adj"):
                    self.Emission[data[x][0][i]][0]=self.Emission[data[x][0][i]][0]+1
                elif (data[x][1][i]=="noun"):
                    self.Emission[data[x][0][i]][1]=self.Emission[data[x][0][i]][1]+1 
                elif (data[x][1][i]=="adv"):
                    self.Emission[data[x][0][i]][2]=self.Emission[data[x][0][i]][2]+1 
                elif (data[x][1][i]=="adp"):
                    self.Emission[data[x][0][i]][3]=self.Emission[data[x][0][i]][3]+1 
                elif (data[x][1][i]=="num"):
                    self.Emission[data[x][0][i]][4]=self.Emission[data[x][0][i]][4]+1 
                elif (data[x][1][i]=="pron"):
                    self.Emission[data[x][0][i]][5]=self.Emission[data[x][0][i]][5]+1 
                elif (data[x][1][i]=="verb"):
                    self.Emission[data[x][0][i]][6]=self.Emission[data[x][0][i]][6]+1 
                elif (data[x][1][i]=="."):
                    self.Emission[data[x][0][i]][7]=self.Emission[data[x][0][i]][7]+1 
                elif (data[x][1][i]=="x"):
                    self.Emission[data[x][0][i]][8]=self.Emission[data[x][0][i]][8]+1 
                elif (data[x][1][i]=="conj"):
                    self.Emission[data[x][0][i]][9]=self.Emission[data[x][0][i]][9]+1 
                elif (data[x][1][i]=="prt"):
                    self.Emission[data[x][0][i]][10]=self.Emission[data[x][0][i]][10]+1 
                elif (data[x][1][i]=="det"):
                    self.Emission[data[x][0][i]][11]=self.Emission[data[x][0][i]][11]+1 
           # calculating actual emission prob by dividing with counts         

        for item in self.Emission:
            Total=sum(self.Emission[item])
            if Total==0:
                Total=1
            for i in range(0,12):
                self.Emission[item][i]=float(self.Emission[item][i])/float(Total)
#         print self.Emission["nick's"]
                        
        pass

    # Functions for each algorithm.
    #
    def naive(self, sentence):
        self.naivePOS=[]
        for word in sentence:
            #print word
            if word in self.Emission:
                r = self.Emission[word].index(max(self.Emission[word]))  #value is a list with 12 values adj,noun,adv,adp,num,pron,verb,dot,x,conj,prt,det
            else:
                r=1
            if(r==0):
                self.naivePOS.append('adj')
            elif (r==1):
                self.naivePOS.append('noun')
            elif (r==2):
                self.naivePOS.append('adv')
            elif (r==3):
                self.naivePOS.append('adp')
            elif (r==4):
                self.naivePOS.append('num')
            elif (r==5):
                self.naivePOS.append('pron')
            elif (r==6):
                self.naivePOS.append('verb')
            elif (r==7):
                self.naivePOS.append('.')
            elif (r==8):
                self.naivePOS.append('x')
            elif (r==9):
                self.naivePOS.append('conj')
            elif (r==10):
                self.naivePOS.append('prt')
            elif (r==11):
                self.naivePOS.append('det')
#             
            #print r
        #print [ [ naivePOS, [] ]
#             for v in self.Emission[word].items():
#                 print k, max(v)
        #print naivePOS
        #return [ [ [ "noun" ] * len(sentence)], [] ]    
        return [[self.naivePOS], [] ]
    
    def createSample(self,sentence):
        dict= {}
        Sample=[]
        Sampleinit=[ "noun" ] * len(sentence)
        dict = defaultdict(lambda: [1,1,1,1,1,1,1,1,1,1,1,1],dict)
        for i in range(0,50):
            if i!=0:
                if len(sentence)>4:
                    Sampleinit[random.choice([0,1,2,3,4])]=random.choice(self.TagsListOrder)
            for j in range(0,len(sentence)):
                for k in range(0,12):
                    if j not in [0,len(sentence)-1]:
                        if sentence[j] in self.Emission:
                            dict[sentence[j]][k] = float(self.trans[Sampleinit[j-1]][k])*float(self.trans[self.TagsListOrder[k]][self.TagsListOrder.index(Sampleinit[j+1])])*float(self.Emission[sentence[j]][k])
                Sampleinit[j] = self.TagsListOrder[dict[sentence[j]].index(max(dict[sentence[j]]))]
                if Sampleinit not in Sample:
                    Sample.append(Sampleinit[:])
        return Sample
    
    
    def mcmc(self, sentence, sample_count):
        returnlist=[]
        Sample=self.createSample(sentence)
        for i in range(0, len(sentence)):
            for j in range(0,len(Sample)):
                if Sample[j][i]=="adj":
                    self.SampleEmisssion[sentence[i]][0]=self.SampleEmisssion[sentence[i]][0]+1
                elif Sample[j][i]=="noun":
                    self.SampleEmisssion[sentence[i]][1]=self.SampleEmisssion[sentence[i]][1]+1
                elif Sample[j][i] =="adv":
                    self.SampleEmisssion[sentence[i]][2]=self.SampleEmisssion[sentence[i]][2]+1
                elif Sample[j][i] =="adp":
                    self.SampleEmisssion[sentence[i]][3]=self.SampleEmisssion[sentence[i]][3]+1
                elif Sample[j][i]=="num":
                    self.SampleEmisssion[sentence[i]][4]=self.SampleEmisssion[sentence[i]][4]+1
                elif Sample[j][i]=="pron":
                    self.SampleEmisssion[sentence[i]][5]=self.SampleEmisssion[sentence[i]][5]+1
                elif Sample[j][i]=="verb":
                    self.SampleEmisssion[sentence[i]][6]=self.SampleEmisssion[sentence[i]][6]+1
                elif Sample[j][i] == ".":
                    self.SampleEmisssion[sentence[i]][7]=self.SampleEmisssion[sentence[i]][7]+1
                elif Sample[j][i]=="x":
                    self.SampleEmisssion[sentence[i]][8]=self.SampleEmisssion[sentence[i]][8]+1
                elif Sample[j][i]=="conj":
                    self.SampleEmisssion[sentence[i]][9]=self.SampleEmisssion[sentence[i]][9]+1
                elif Sample[j][i]=="prt":
                    self.SampleEmisssion[sentence[i]][10]=self.SampleEmisssion[sentence[i]][10]+1
                elif Sample[j][i]=="det":
                    self.SampleEmisssion[sentence[i]][11]=self.SampleEmisssion[sentence[i]][11]+1

        for item in self.SampleEmisssion:
            x=sum(self.SampleEmisssion[item])
            if x==0:
                x=1
            for i in range(0,12):
                self.SampleEmisssion[item][i]=float(self.SampleEmisssion[item][i])/float(x)
        listfinal=[]
        if len(sentence)>1:
            if len(Sample)>50:
                for i in range(20,25):
                    returnlist.append(Sample[i])
                return [ returnlist, [] ]
            elif len(Sample)>5:
                i=len(Sample)-1
                j=len(Sample)-6
                while (i>j):
                    listfinal.append(Sample[i])
                    i=i-1
                return [ listfinal, [] ]
            else:
                return [ [ [ "noun" ] * len(sentence) ] * sample_count, [] ]

        else:
            return [ [ [ "noun" ] * len(sentence) ] * sample_count, [] ]

    def best(self, sentence):
        
        return [[self.naivePOS], [] ]


    def max_marginal(self, sentence):
        list1=[]
        list2=[]
        for i in range(0, len(sentence)):
            r=self.SampleEmisssion[sentence[i]].index(max(self.SampleEmisssion[sentence[i]]))
            list2.append(max(self.SampleEmisssion[sentence[i]]))
            if(r==0):
                list1.append('adj')
            elif (r==1):
                list1.append('noun')
            elif (r==2):
                list1.append('adv')
            elif (r==3):
                list1.append('adp')
            elif (r==4):
                list1.append('num')
            elif (r==5):
                list1.append('pron')
            elif (r==6):
                list1.append('verb')
            elif (r==7):
                list1.append('.')
            elif (r==8):
                list1.append('x')
            elif (r==9):
                list1.append('conj')
            elif (r==10):
                list1.append('prt')
            elif (r==11):
                list1.append('det')

        return [ [list1], [list2] ]

    def calculateViterbiProb(self,sentence):
        MAPList={}
        MAPList = defaultdict(lambda: [0,0,0,0,0,0,0,0,0,0,0,0],MAPList)
        EmissionLst=[]

        for i in range(0,len(sentence)):
            if sentence[i] not in self.Emission:
                self.Emission[sentence[i]]=[0,0,0,0,0,0,0,0,0,0,0,0]

            for j in range(0,12):

                if i==0:
                    MAPList[sentence[i]][j] = float(self.Emission[sentence[i]][j]+0.00000001)*float(self.InitProb[self.TagsListOrder[j]]+0.00000001)
                else:
                    for k in range(0,12):
                        EmissionLst.append(MAPList[sentence[i-1]][k]*(self.trans[self.TagsListOrder[j]][k]+0.00000001))
                    MAPList[sentence[i]][j]= float(self.Emission[sentence[i]][j]+0.00000001)*max(EmissionLst)
        return MAPList
    def viterbi(self, sentence):
        
        MAPList= self.calculateViterbiProb(sentence)            #listEmissionProbabilities=[0,0,0,0,0,0,0,0,0,0,0,0]
        sequence=[]
        for i in range(0, len(sentence)):
            id= MAPList[sentence[i]].index(max(MAPList[sentence[i]]))
            sequence.append(self.TagsListOrder[id])


        return [[sequence], []]
    

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #    Most algorithms only return a single labeling per sentence, except for the
    #    mcmc sampler which is supposed to return 5.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for max_marginal() and is the marginal probabilities for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Naive":
            return self.naive(sentence)
        elif algo == "Sampler":
            return self.mcmc(sentence, 5)
        elif algo == "Max marginal":
            return self.max_marginal(sentence)
        elif algo == "MAP":
            return self.viterbi(sentence)
        elif algo == "Best":
            return self.best(sentence)
        else:
            print "Unknown algo!"

    
