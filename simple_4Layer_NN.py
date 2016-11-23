import numpy as np
import time
import sys


# sigmoid function
def sigmoid(x, deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))


def Neural_Network(X, y):
    
    iter = 60000
    progress = iter/10
    
    # seed random numbers by generating same number at beginning
    np.random.seed(1)
   
    # init weights randomly with mean 0
    syn0 = 2*np.random.random((2,4)) - 1
    syn1 = 2*np.random.random((4,4)) - 1
    syn2 = 2*np.random.random((4,1)) - 1
    
    for i in xrange(60000):
        
    	# forward propagation through layers 0, 1, and 2
    	l0 = X
    	l1 = sigmoid(np.dot(l0,syn0))
    	l2 = sigmoid(np.dot(l1,syn1))
        l3 = sigmoid(np.dot(l2,syn2))
    
    	# finding error
    	l3_error = y - l3
    	# print Error
    	if (i% progress) == 0:
	   	    #print "Error:" + str(np.mean(np.abs(l3_error)))
	   	    print '\b. ' ,
	   	    sys.stdout.flush()
            
        # multipy error by the slop of sigmoid at values in l3
        l3_delta = l3_error * sigmoid(l3, True)
    
    	# How much did each l2 contirbue to l3? adjust accordingly
        l2_error = l3_delta.dot(syn2.T)
        l2_delta = l2_error * sigmoid(l2, True)
    
    
    	# How much did each l1 contirbue to l2? adjust accordingly
        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error * sigmoid(l1, True)
    
    
	    #update weights
        syn2 += l2.T.dot(l3_delta)
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)
        
    print "Done!\n"  
    print "Output: "
    print l3
    print
    main()
    
    
    
def main():
    
    # input data
    X = np.array([    [0,0],
				      [0,1],
				      [1,0],
				      [1,1]    ])

    # output data
    xor = np.array([[0,1,1,0]]).T
    orr = np.array([[0,1,1,1]]).T
    andd = np.array([[0,0,0,1]]).T
    nand = np.array([[1,1,1,0]]).T
    nor = np.array([[1,0,0,0]]).T
    xnor = np.array([[1,0,0,1]]).T
    
    
        
    while True:
        print "\nTo quit press 'q'.\nChoose a # for Logic Gate Operatons:"
        userInput = raw_input("1) AND \n2) OR \n3) XOR\n4) NAND \n5) NOR \n6) XNOR\n")
        if userInput not in ('1','2','3','4','5','6', 'q', 'Q'):
            print "Wrong input. Try again"
        else:
            break
    
    if (userInput == '1'):
        print "\nAND Training Starts " ,
        sys.stdout.flush()
        Neural_Network(X, y = andd)
    
    if (userInput == '2'):
        print "\nOR Training Starts " ,
        sys.stdout.flush()
        Neural_Network(X, y = orr)
    
    if (userInput == '3'):
        print "\nXOR Training Starts " ,
        sys.stdout.flush()
        Neural_Network(X, y = xor)
        
    if (userInput == '4'):
        print "\nNAND Training Starts " ,
        sys.stdout.flush()
        Neural_Network(X, y = nand)
        
    if (userInput == '5'):
        print "\nNOR Training Starts " ,
        sys.stdout.flush()
        Neural_Network(X, y = nor)
        
    if (userInput == '6'):
        print "\nXNOR Training Starts " ,
        sys.stdout.flush()
        Neural_Network(X, y = xnor)
        
    if (userInput == 'q' or 'Q'):
        exit()
   

if __name__ == '__main__': 
    
    main()

