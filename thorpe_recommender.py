##################### Recommender for Thorpe's strategy ######################

'''In table: 0=stand, 1=hit, 2=double

Lower part of table 1 - only if we have at least 1 ace (a==1)'''
rec_ace=[[1,1,1,2,2,1,1,1,1,1], \
         [1,1,1,2,2,1,1,1,1,1], \
         [1,1,2,2,2,1,1,1,1,1], \
         [1,1,2,2,2,1,1,1,1,1], \
         [1,2,2,2,2,1,1,1,1,1], \
         [0,2,2,2,2,0,0,1,1,1], \
         [0,0,0,0,0,0,0,0,0,0]]

'''First 3 rows are from table 2, the rest is the upper part of table 1'''
rec=[[1,2,2,2,2,1,1,1,1,1], \
     [2,2,2,2,2,2,2,2,1,1], \
     [2,2,2,2,2,2,2,2,2,1], \
     [1,1,0,0,0,1,1,1,1,1], \
     [0,0,0,0,0,1,1,1,1,1], \
     [0,0,0,0,0,1,1,1,1,1], \
     [0,0,0,0,0,1,1,1,1,1], \
     [0,0,0,0,0,1,1,1,1,1]] \

'''If a==1, then we have at least 1 ace otherwise not, 
p = sum of player's hand
d = sum of dealer's hand'''

class recommender():
    def __init__(self, a, p, d): 
        self.a = a
        self.p = p
        self.d = d
    
    def thorpe(self):    
        if self.a == 1 and self.p <= 19:
            return rec_ace[self.p-13][self.d-2] #lowest value of player's hand = ace + 2 = 13
        elif self.a == 1 and self.p > 19:
            return 0  
        elif self.a != 1 and self.p <= 16 and self.p >= 9:
            return rec[self.p-9][self.d-2] #lowest value of player's hand = 9
        elif self.a != 1 and self.p < 9:
            return 1
        else:
            return 0

recommendation = recommender(a=1,p=21,d=2) 
recommendation.thorpe()
