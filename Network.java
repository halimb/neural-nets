package network;

import java.util.Random;

public class Network {

	public int numberOfLayers;
	public int inputLayerSize;
	public double[] ideal;
	public double[] actual;
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
		for(int i = 1; i < n.length; i++){
			double[][] layerWeights = new double[n[i - 1]][n[i]];
			double[] layerBiases = new double[n[i]];
			for(int j = 0; j < n[i]; j++){
				layerBiases[j] = rand.nextGaussian();
				
				for(int k = 0; k < n[i - 1]; k++){
					layerWeights[k][j] = rand.nextGaussian();
				}
			}
			weights[i - 1] = layerWeights;
			biases[i - 1] = layerBiases;
		}
		System.out.println("Weights Size = " + weights.length + ", Biases size = " + biases.length);
		System.out.println("Total number of layers = " + numberOfLayers);
	}
	
	/* METHOD
	 * takes in an input vector, of the same dimension as 
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
	private double[] calculateDeltas(int layer){
		
		double[] result = new double[getBiases(layer).length];
		if(layer == numberOfLayers){
			result = F.hadamard(F.substract(actual, ideal),
							    F.sigmoidPrime(getZs(layer)));
			
			setDeltas(layer, result);
			
		}
		else{
			result = F.hadamard( 
					   F.mDotv( F.transpose(getWeights(layer + 1)), calculateDeltas(layer + 1) ),
					   F.sigmoidPrime(getZs(layer))
					   );
			setDeltas(layer, result);
		}
		return result;
		
	}
	
	/* makes a call to the private method above to calculate
	 * all the errors in the network starting from the first
	 * hidden layer (layer 2). ignores the return value.  */
	// <<TO REVISE>>
	public void  calculateDeltas(){
		calculateDeltas(2);
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
}
