# Coop Interative Game

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.special as sps
import random

numplayers = int(input("Enter number of players \n"))
print("\n")

def actionlist():
    print("a = Take a coin \nb = Take two coins (Foreign Aid) \nc = Claim three coins as the Duke \nd = Take two coins as the Captain \ne = Get new cards with the Ambassador \nf = Pay three coins to assassinate \ng = Pay seven coins to coop")
    
def cardlist():
    firstcard = random.randint(1,5)
    secondcard = random.randint(1,5)
    
    if firstcard == 1:
        firstcard = "duke"
    if firstcard == 2:
        firstcard = "ambassador"
    if firstcard == 3:
        firstcard = "captain"
    if firstcard == 4:
        firstcard = "assassin"
    if firstcard == 5:
        firstcard = "contessa"
        
    if secondcard == 1:
        secondcard = "duke"
    if secondcard == 2:
        secondcard = "ambassador"
    if secondcard == 3:
        secondcard = "captain"
    if secondcard == 4:
        secondcard = "assassin"
    if secondcard == 5:
        secondcard = "contessa"
        
    cardvector = [];
    cardvector.append(secondcard)
    cardvector.append(firstcard)
    
    return cardvector

class player1:
    def __init__(self, name, numcoins, numlives, card1, card2):
        plcardvector = [];
        plcardvector = cardlist();
        self.name = name
        self.numcoins = numcoins
        self.numlives = numlives
        self.card1 = plcardvector[0]
        self.card2 = plcardvector[1]
  
p1 = player1("Player 1", 2, 2, "string1", "string2")

class player2:
    def __init__(self, name, numcoins, numlives, card1, card2):
        plcardvector = [];
        plcardvector = cardlist();
        self.name = name
        self.numcoins = numcoins
        self.numlives = numlives
        self.card1 = plcardvector[0]
        self.card2 = plcardvector[1]
        
p2 = player2("Player 1", 2, 2, "string1", "string2")

class player3:
    def __init__(self, name, numcoins, numlives, card1, card2):
        plcardvector = [];
        plcardvector = cardlist();
        self.name = name
        self.numcoins = numcoins
        self.numlives = numlives
        self.card1 = plcardvector[0]
        self.card2 = plcardvector[1]
  
p3 = player3("Player 3", 2, 2, "string1", "string2")

class player4:
    def __init__(self, name, numcoins, numlives, card1, card2):
        plcardvector = [];
        plcardvector = cardlist();
        self.name = name
        self.numcoins = numcoins
        self.numlives = numlives
        self.card1 = plcardvector[0]
        self.card2 = plcardvector[1]
        
p4 = player4("Player 4", 2, 2, "string1", "string2")

class player5:
    def __init__(self, name, numcoins, numlives, card1, card2):
        plcardvector = [];
        plcardvector = cardlist();
        self.name = name
        self.numcoins = numcoins
        self.numlives = numlives
        self.card1 = plcardvector[0]
        self.card2 = plcardvector[1]
  
p5 = player5("Player 1", 2, 2, "string1", "string2")

class player6:
    def __init__(self, name, numcoins, numlives, card1, card2):
        plcardvector = [];
        plcardvector = cardlist();
        self.name = name
        self.numcoins = numcoins
        self.numlives = numlives
        self.card1 = plcardvector[0]
        self.card2 = plcardvector[1]
        
p6 = player6("Player 1", 2, 2, "string1", "string2")

def captain():
    capchoice = int(input("Choose the player you wish to steal from. Player 1 = 1, Player 2 = 2, etc \n"))
    if (capchoice == 1):
        p1.numcoins = p1.numcoins - 2
    if (capchoice == 2):
        p2.numcoins = p2.numcoins - 2
    if (capchoice == 3):
        p3.numcoins = p3.numcoins - 2
    if (capchoice == 4):
        p4.numcoins = p4.numcoins - 2
    if (capchoice == 5):
        p5.numcoins = p5.numcoins - 2
    if (capchoice == 6):
        p6.numcoins = p6.numcoins - 2
    
def assassion():
    aschoice = int(input("Choose the player you wish to assassinate. Player 1 = 1, Player 2 = 2, etc \n"))
    if (aschoice == 1):
        p1.numlives = p1.numlives - 1
    if (aschoice == 2):
        p2.numlives = p2.numlives - 1
    if (aschoice == 3):
        p3.numlives = p3.numlives - 1
    if (aschoice == 4):
        p4.numlives = p4.numlives - 1
    if (aschoice == 5):
        p5.numlives = p5.numlives - 1
    if (aschoice == 6):
        p6.numlives = p6.numlives - 1
        
def coup():
    couchoice = int(input("Choose the player you wish to coup. Player 1 = 1, Player 2 = 2, etc \n"))
    if (couchoice == 1):
        p1.numlives = p1.numlives - 1
    if (couchoice == 2):
        p2.numlives = p2.numlives - 1
    if (couchoice == 3):
        p3.numlives = p3.numlives - 1
    if (couchoice == 4):
        p4.numlives = p4.numlives - 1
    if (couchoice == 5):
        p5.numlives = p5.numlives - 1
    if (couchoice == 6):
        p6.numlives = p6.numlives - 1

def turn():
    playgame = True
    while(playgame == True):
        for i in range(1,numplayers + 1):
            if i == 1:
                unplayer = p1
            if i == 2:
                unplayer = p2
            if i == 3:
                unplayer = p3
            if i == 4:
                unplayer = p4
            if i == 5:
                unplayer = p5
            if i == 6:
                unplayer = p6
            a=0; b=0; c=0; d=0; e=0; f=0; g=0
            print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            print("Begin your turn Player {:1}\n".format(i))
            cardchoice = input("Do you need to see your cards? Type yes or no \n")
            if (cardchoice == "yes"):
                print("\n")
                print(unplayer.card1)
                print(unplayer.card2)
            coinschoice = input("Do you need to see your coins? Type yes or no \n")
            if (cardchoice == "yes"):
                print("\n")
                print(unplayer.numcoins)
                print("\n\n")
            actionlist()
            turnchoice = input("Choose an action from the list above\n")
            if (turnchoice == "a"):
                unplayer.numcoins = unplayer.numcoins + 1
            if (turnchoice == "b"):
                unplayer.numcoins = unplayer.numcoins + 2
            if (turnchoice == "c"):
                unplayer.numcoins = unplayer.numcoins + 3
            if (turnchoice == "d"):
                captain()
                unplayer.numcoins = unplayer.numcoins + 2
            if (turnchoice == "e"):
                ambvector = cardlist()
                print("\n")
                print(ambvector)
                ambchoice1 = input("Do you want card number 1? Type yes or no \n")
                if (ambchoice1 == "yes"):
                    print("\n")
                    print("These are your current cards")
                    print(unplayer.card1)
                    print(unplayer.card2)
                    ambchoice12 = int(input("Which card do you want to switch card number one with? Choose 1 or 2 \n"))
                    if (ambchoice12 == 1):
                        unplayer.card1 = ambvector[0]
                    if (ambchoice12 == 2):
                        unplayer.card2 = ambvector[1]
                ambchoice2 = input("Do you want card number 2? Type yes or no \n")
                if (ambchoice2 == "yes"):
                    print("\n")
                    print("These are your current cards")
                    print(unplayer.card1)
                    print(unplayer.card2)
                    ambchoice22 = int(input("Which card do you want to switch card number 2 with? Choose 1 or 2 \n"))
                    if (ambchoice22 == 1):
                        unplayer.card1 = ambvector[0]
                    if (ambchoice22 == 2):
                        unplayer.card2 = ambvector[1]
            if (turnchoice == "f"):
                assassion()
                unplayer.numcoins = unplayer.numcoins - 3
            if (turnchoice == "g"):
                coup()
                unplayer.numcoins = unplayer.numcoins - 7
        endchoice = input("Is the game over. Type yes or no \n")
        if (endchoice == "yes"):
            playgame = False
            
        
turn()



