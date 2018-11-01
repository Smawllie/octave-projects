# linear-regression
A program that takes in a set of data (in this case #bedrooms, square footage, and the price of a house) and uses linear regression to obtain the optimal line that "fits" the data. This program showcases:
- Application of linear algebra to create a line from given inputs and outputs to predict new input and output values
- Feature normalization which scales down data to guarantee gradient descent doesn't take forever to find a minimum
- Gradient descent which is an iterative algorithm that obtains lines of fit that will always be better than before
- The normal equation which computes the minimum without the need of iterations from gradient descent
