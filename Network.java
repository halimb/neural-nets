package network;

import java.util.Random;

public class Network {

	public int numberOfLayers;
	public int inputLayerSize;
	public double[][][] weights;
	public double[][] biases;
	public double[][] activations;
	public double[][] zs;
	public double[][] deltas;

	/* CONSTRUCTOR
	 * The number of int parameters in the Network(int...n)
	 * is the number of layers in the network, with each 
	 * entry represents the number of neurons in a given layer.
	 * */
	public Network(int ... n){
		numberOfLayers = n.length;
		inputLayerSize = n[0];
		activations = new double[numberOfLayers][];
		biases = new double[numberOfLayers - 1][];
		weights = new double[numberOfLayers - 1][][];
		zs = new double[numberOfLayers - 1][];
		deltas = new double[numberOfLayers - 1][];
		Random rand = new Random();

		//Populating the network weights and biases
		for(int i = 0; i < biases.length; i++){
			biases[i] = new double[n[i + 1]];
			for(int j = 0; j < biases[i].length; j++){
				biases[i][j] = rand.nextGaussian();
			}
		}
		
		for(int i = 0; i < n.length - 1; i++){
			weights[i] = new double[n[i+1]][];
			for(int j = 0; j < n[i+1]; j++){
				weights[i][j] = new double[n[i]];
				for(int k = 0; k < weights[i][j].length; k++){
					weights[i][j][k] = rand.nextGaussian();
				}
			}
		}
	}
	
	/* takes in an input vector, of the same dimension as 
	 * the first layer of the network (the input layer), 
	 * then feeds that input to the next layer, through 
	 * the weights, adds the biases then applies the Sigmoid 
	 * function to the weighted input, obtaining an output
	 * vector which is in turn going to be fed to the next 
	 * layer and so on until the last layer (output layer)
	 * is reached. the method  returns the output vector 
	 * of the last layer, after populating the network's 
	 * activation and weighted input vectors arrays.*/
	public double[] feedForward(double[] input) throws MismatchException{
		if(input.length != inputLayerSize){
			throw new MismatchException("The input vector and the network"
					           + " input layer dimensions are different!");
		}
		double[] a = null, z;
		activations[0]= input;
		for(int i = 0; i < numberOfLayers - 1; i++){
			z = F.sum( F.mDotv(weights[i], activations[i]),
					   biases[i]);
			a = F.sigmoid(z);
			zs[i] = z;
			activations[i + 1] = a;
		}
		return a;
	}
	
	/*calculates the errors of the network, with  a recursive 
	 * call, from the given layer to the output layer */ 
	// <<TO REVISE>>
	private double[] calculateDeltas(int layer, double[] input, double[] idealOutput){
		double[] result = new double[getBiases(layer).length];
		if(layer == numberOfLayers){
			double[] actualOutput = feedForward(input);
			result = F.hadamard(F.substract(actualOutput, idealOutput),
					    F.sigmoidPrime(getZs(layer)));
			
			setDeltas(layer, result);
		}
		else{
			result = F.hadamard(F.mDotv( F.transpose(getWeights(layer + 1)), 
						     calculateDeltas(layer + 1, input, idealOutput) ),
					    F.sigmoidPrime(getZs(layer)) );
			setDeltas(layer, result);
		}
		return result;
	}
	
	/* makes a call to the private method above to calculate
	 * all the errors in the network based on the provided actual 
	 * and ideal outputs, starting from the first hidden layer (l:2).
	 * Ignores the return value.  */
	public void  calculateDeltas(double[] input, double[] idealOutput){
		calculateDeltas(2, input, idealOutput);
	}
	

	/* Stochastic Gradient Descent (SGD) implementation. given a learning rate
	 * and a set of training data (input/ideal output pairs), this method calculates
	 * the cost function derivative (gradient) for each input, then uses the prior 
	 * to update the network's weights and biases with the goal of minimizing the cost.*/
	public void train(TrainingData[] training, double learningRate, int chunkSize, int epochs){
		
		int numOfchunks = training.length / chunkSize;
		int remainder = training.length % chunkSize;
		
		/* initializes the matrices that will contain the 
		 * sum cost function partial derivative with respect
		 * to individual weights (dC/dw) and biases (dC/db) 
		 * (i.e: components of the cost function's gradient)*/
		double[][][] dCdw = F.zeros(weights);
		double[][] dCdb = F.zeros(biases);
		
		if(epochs < 1){
			epochs = 1;
		}
		for(int e = 0; e < epochs; e++){
			// loops through the learning batch chunks
			for(int x = 0; x <= numOfchunks; x++){
				int startIndex, endIndex;
				if(x < numOfchunks - 1){
					startIndex = x * chunkSize;
					endIndex = startIndex + chunkSize;
				}
				else if(remainder > 0){
					startIndex = numOfchunks * chunkSize;
					endIndex = training.length;
				}
				else{
					break;
				}
				Random random = new Random(System.currentTimeMillis());
				// loops through the chunk of training data
				for(int i = x * chunkSize; i < endIndex; i++){
					int randIndex = random.nextInt(training.length);
					dCdw = F.zeros(weights);
					dCdb = F.zeros(biases);
					double[] input = training[randIndex].getInput();
					double[] ideal = training[randIndex].getDesired();
					calculateDeltas(input, ideal);
					
					 /* increments the sum of the cost function's partial
					 * derivative with respect to the bias and  weight 
					 * with the value of the partial derivatives for the 
					 * current training data couple*/
					
					//biases errors
					for(int j = 0; j < numberOfLayers - 1; j++){
						for(int k = 0; k < biases[j].length; k++){
							dCdb[j][k] += deltas[j][k];
						}
					}
					
					//weights errors
					for(int j = 0; j < numberOfLayers - 1; j++){
						for(int k = 0; k < weights[j].length; k++){
							for(int n = 0; n < weights[j][k].length; n++){
								double activation = activations[j][n];
								double delta = deltas[j][k];
								dCdw[j][k][n] += activation * delta;
							}
						}
					}
					
				}
				/* applies stochastic gradient descent to the networks  *
				 * weights and biases using the values calculated above */
				//biases
				for(int i = 0; i < numberOfLayers - 1; i++){
					for(int j = 0; j < biases[i].length; j++){
						biases[i][j] -= ((learningRate/chunkSize) * dCdb[i][j]);
					}
				}
					
				//weights
				for(int i = 0; i < numberOfLayers - 1; i++){
					for(int j = 0; j < weights[i].length; j++){
						for(int k = 0; k < weights[i][j].length; k++){
							weights[i][j][k] -= ((learningRate/chunkSize) * dCdw[i][j][k]);
						}
					}
				}
				if(x % 1000 == 0){
				System.out.println("Completed chunk : " + x);
				}
			}
			
			System.out.println("Completed epoch:" + e + ". Testing the network...");
			MNISTTest.testNetwork(this, MNISTTest.testData, 5000);
		}
		System.out.println("SUCCESSFULLY FINISHED LEARNING");
	}
	
	public double[] getBiases(int layer){
		return biases[layer - 2];
	}
	public double[][] getWeights(int layer){
		return weights[layer - 2];
	}
	public double[] getZs(int layer){
		return zs[layer - 2];
	}
	public double[] getActivations(int layer){
		return activations[layer - 1];
	}
	public double[] getDeltas(int layer){
		return deltas[layer - 2];
	}
	public void setDeltas(int layer, double[] values){
			deltas[layer - 2] = values;
	}
	public void setBiases(int layer, double[] values){
		biases[layer - 2] =  values;
	}
	public void setWeights(int layer, double[][] values){
		weights[layer - 2] =  values;
	}
	public void setZs(int layer, double[] values){
		zs[layer - 2] =  values;
	}
	public void setActivations(int layer, double[] values){
		activations[layer - 1] =  values;
	}

	@Override
	public String toString(){
		return "biases: " + biases.length
		+"\nweights: " + weights.length
		+"\nlayers: " + numberOfLayers
		+"\nactivations: " + activations.length
		+"\nZs: " + zs.length
		+"\nDeltas: " + deltas.length;
	}
	
	public void printWeights(){
		System.out.println("WEIGHTS:");
		for(int i = 0; i < weights.length; i++){
			System.out.println("Layer: " + (i + 2) );
			for(int j = 0; j < weights[i].length; j++){
				System.out.print(j + ":");
				for(int k = 0; k < weights[i][j].length; k++){
					System.out.print("\t" +  weights[i][j][k]);
				}
				System.out.println("");
			}
		}
		System.out.println("\n");
	}
	
	public void printDeltas(){
		System.out.println("DELTAS:");
		for(int i = 0; i < deltas.length; i++){
			System.out.println("Layer: " + (i + 2) );
			for(int j = 0; j < deltas[i].length; j++){
				System.out.print("\t" + deltas[i][j]);
			}
			System.out.println("");
		}
		System.out.println("\n");
	}
	
	public void printActivations(){
		System.out.println("ACTIVATIONS:");
		for(int i = 0; i < activations.length; i++){
			System.out.println("Layer: " + (i + 1) );
			for(int j = 0; j < activations[i].length; j++){
				System.out.print("\t" + activations[i][j]);
			}
			System.out.println("");
		}
		System.out.println("\n");
	}
	
	public void printBiases(){
		System.out.println("BIASES:");
		for(int i = 0; i < biases.length; i++){
			System.out.println("Layer: " + (i + 2) );
			for(int j = 0; j < biases[i].length; j++){
				System.out.print("\t" + biases[i][j]);
			}
			System.out.println("");
		}
		System.out.println("\n");
	}
}
