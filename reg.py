# Topic    :	Regression Module for Machine Learning , L1 and L2 models
# Author   : 	Priyankar Talukdar, MS2011009.
#		International Institute of Information Technology, Bangalore 
# Date     : 	Sunday 07 September 2014 04:09:37 PM IST 	
# Ref Book :	Linear Programming: Foundations and Extensions by Robert J Vanderbei 


# Ver-1.0: The design sample is based on L1 regression 
# Example: Let (x_1, y_2) and (x_2,y_2) be the data points and a_1 and a_o be the parameters of the regression
#	   line, y = a_i x + a_o. The design formulation is given as
# 		  
#	 The L1 regression problem can be formulated as : for n data points 
#		Min 		Sum_{z_i} for i = 1..to n
#		Subject to:	z_i >=   y_i - (a_1 x_i + a_o ) for i = 1..to n
#				z_i >= - y_i + (a_1 x_i + a_o ) for i = 1..to n
#
#	 The design problem can be extended for multiple data points 
#
#
#	Related Package: sudo apt-get install python-cvxopt
#	Python: 2.7 and higher

# Imported files : Library functions and modules called here 
from cvxopt import matrix, exp, solvers
import matplotlib.pyplot as plt

# sets the type of equation mapping: Y = a_m x^m + a_(m-1) x^(x-1) ... a_1 x + a_o
nonLinearDegree = 4

# define dimension boundary param
xLocStart =     -10
xLocEnd = 	 30
yLocStart =	-10 
yLocEnd =  	-50

stepMap = 	 10

# Design points: Optional and can be read from a text file
xLoc= [1.0, 2.0, 3.0, 8.0, 4.0, 16.0, -4.0, 4.9,1.5]
yLoc = [1.0, 5.0, 4.0, 8.0, -1.0, 7.0, 8.0, 11.4,2.3]

def sampleModule():
	A = matrix([ [-1.0, 1.0, -2.0, 2.0], [-1.0, 1.0, -1.0, 1.0], [-1.0, -1.0, -0.0, 0.0],[0.0, 0.0, -1.0, -1.0]])
	b = matrix([ -1.0, 1.0, -2.0, 2.0 ])
	c = matrix([ 0.0, 0.0, 1.0, 1.0 ])
	sol=solvers.lp(c,A,b)
	print(sol['x'])
#-------------------------------------------------------------------------------------------------------------
# L1: Linear Regression Model Design Framework	
#-------------------------------------------------------------------------------------------------------------
def automateModule():
		l1 = []
		l2 = []
		for j in range(0,len(xLoc)):
			#l3 = [0] * (len(xLoc)*2)
			l1.append(-xLoc[j])
			l1.append(xLoc[j])
			
			l2.append(-1)
			l2.append(1)
	
		ma = matrix([l2])
		A = matrix([ma])
		ma = matrix([l1])		
		A = matrix([[ma],[A]])
		
		for j in range(0,len(xLoc)):
			l3 = [0] * (len(xLoc)*2)
			l3[2*j] = -1.0
			l3[2*j +1 ] = -1.0
			ma = matrix([l3])
			A = matrix([[A],[ma]])
		print A

		l1 = []
		for j in range(0,len(yLoc)):	
			l1.append(-yLoc[j])	
			l1.append(yLoc[j])
		
		b = matrix([l1])	
		print b

		l3 = [0] * 2
		for j in range(0, len(yLoc)):
			l3.append(1.0)
		c = matrix([l3])
		print c

		sol=solvers.lp(c,A,b)
		print(sol['x'])
		return (sol['x'])


def defineLocRegressLine(result):
	xRegressLoc = []
	yRegressLoc = []	
	a_1 = result[0]
	a_o = result[1]
	for i in range(xLocStart, xLocEnd):
		xRegressLoc.append(i)
		yRegressLoc.append(a_1*i + a_o)		
		i = i + 15
	return xRegressLoc, yRegressLoc

def displayMap(result):
	plt.plot(xLoc, yLoc, 'ro')
	xRegressLoc, yRegressLoc = defineLocRegressLine(result);
	#print xRegressLoc, yRegressLoc
	plt.plot(xRegressLoc, yRegressLoc)
	plt.axis([-10, 30, -10, 50])
	plt.show()	

def L1_LinearRegression():
	result = automateModule()
	displayMap(result)


#-------------------------------------------------------------------------------------------------------------
# L1: Non Linear Regression Model Design Framework
#-------------------------------------------------------------------------------------------------------------
def automateNonLinearModule():
		l2 = []
		for j in range(0,len(xLoc)):
			l2.append(-1)
			l2.append(1)
	
		ma = matrix([l2])
		A = matrix([ma])

		l1 = []
		for i in range(1,nonLinearDegree + 1):
			l1 = []
			for j in range(0,len(xLoc)):
				l1.append(-pow(xLoc[j],i))
				l1.append(pow(xLoc[j],i))	
			ma = matrix([l1])		
			A = matrix([[ma],[A]])			

		
		for j in range(0,len(xLoc)):
			l3 = [0] * (len(xLoc)*2)
			l3[2*j] = -1.0
			l3[2*j +1 ] = -1.0
			ma = matrix([l3])
			A = matrix([[A],[ma]])
		print A

		l1 = []
		for j in range(0,len(yLoc)):	
			l1.append(-yLoc[j])	
			l1.append(yLoc[j])
		
		b = matrix([l1])	
		print b

		l3 = [0] * (nonLinearDegree + 1)
		for j in range(0, len(yLoc)):
			l3.append(1.0)
		c = matrix([l3])
		print c

		sol=solvers.lp(c,A,b)
		print(sol['x'])
		return (sol['x'])


def computeYRegVal(xVal,result):
	yVal = 0
	for i in range(nonLinearDegree,-1,-1):
		yVal = yVal + (result[i] * pow(xVal,(nonLinearDegree-i)) )
	return yVal

def defineLocNonLinearRegressLine(result): # nonLinearDegree
	xRegressLoc = []
	yRegressLoc = []
	for i in range(xLocStart,xLocEnd):
		xRegressLoc.append(i)		
		yRegressLoc.append(computeYRegVal(i,result))
		i = i + stepMap
	return xRegressLoc, yRegressLoc

def displayNonLinearMap(result,sigmaVal,eqn):
	eq_lb = '$ y = ' + eqn + '$'	
	ax = plt.gca()
	ax.plot(1, 2,'g', label=eq_lb)
	ax.legend()

	plt.xlabel('x-axis')
	plt.ylabel('y-axis')
	status = 'Regression Analysis: L1 and Degree:' + str(nonLinearDegree)
	plt.title(status)
	plt.plot(xLoc, yLoc, 'ro')
	xRegressLoc, yRegressLoc = defineLocNonLinearRegressLine(result);
	plt.plot(xRegressLoc, yRegressLoc,'g')
	plt.text(-8, 45, '$\sum \sigma = $'+ str(round(sigmaVal,1)))
	plt.axis([-10, 30, -10, 50])
	plt.grid(True)
	plt.show()	

def computeSumDeviation(result):
	sigmaVal = 0
	for i in range(nonLinearDegree+1, len(result)):
		sigmaVal = sigmaVal + result[i]
	return sigmaVal

def genEquationString(result):
	eqnStrSet = []
	eqnStr = ''	
	for i in range(0,nonLinearDegree+1):
		if(nonLinearDegree == i):
			eqnStr = str(round(result[i],2))
		else:
			eqnStr = str(round(result[i],2)) + 'x^' + str(nonLinearDegree-i)
		eqnStrSet.append(eqnStr)
	eqnStr = ''	
	for j in range(len(eqnStrSet)-1,-1,-1):
			if (result[j] > 0):
				print result[j]
				if(j == 0):
					eqnStr = eqnStrSet[j] + eqnStr	
				else:
					eqnStr = '+' + eqnStrSet[j] + eqnStr
			else:
				eqnStr = eqnStrSet[j] + eqnStr	
	return eqnStr	
	

def L1_NonLinearRegression():
	result = automateNonLinearModule()
	sigmaVal = computeSumDeviation(result)
	eqnStr = genEquationString(result)
	displayNonLinearMap(result,sigmaVal,eqnStr)	

def main():
	#L1_LinearRegression()
	L1_NonLinearRegression()

if __name__ == '__main__':
	main()
