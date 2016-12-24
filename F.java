package network;

public class F {
	
	//returns the sum of two or more vectors 
	public static double[] sum(double[] a, double[] b){
		double[] res = new double[a.length];
		for(int i = 0; i < res.length; i++){
			res[i] = a[i] + b[i];
		}
		return res;
	}
	
	//subsracts vy from vx and returns the resulting vector 
	public static double[] substract(double[] vx, double[] vy ){
		if(vx.length != vy.length){
			throw new MismatchException("trying to add vectors of different dimensions."
					+ "vx.dimension = " + vx.length  
					+ "\nvy.dimension = " + vy.length);
		}
		double[] res = new double[vx.length];
		for(int i = 0; i < vx.length; i++){
				res[i] = vx[i] - vy[i];
		}
		return res;
	}
	
	//Sigmoid (logistic) function
	public static double sigmoid(double x){
		return (1.0/(1.0+Math.exp(-x)) );
	}
	
	//Applies the Sigmoid function to each entry in a vector and returns it
	public static double[] sigmoid(double[] x){
		for(int i = 0; i < x.length; i++){
			x[i] = sigmoid(x[i]);
		}
		return x;
	}
	
	//Self-descriptive helper method
	public static double sigmoidPrime(double x){
		return sigmoid(x)*(1 - sigmoid(x));
	}
	
	//Sigmoid prime for elementwise application to a vector 
	public static double[] sigmoidPrime(double[] x){
		for(int i = 0; i < x.length; i++){
			x[i] = sigmoidPrime(x[i]);
		}
		return x;
	}
	
	/*Helper method to apply Hadamard product 
	 * of two vectors, return the result vector*/
	public static double[] hadamard(double[] vx, double[] vy){
		if(vx.length != vy.length){
			throw new MismatchException("can't apply Hadamard product "
					+ "of vecotrs of different dimensions:\nvx.dimension = " 
					+ vx.length + ", vy.dimension = " + vy.length);
		}
		double[] res = new double[vx.length];
		for(int i = 0; i < vx.length; i++){
			res[i] = vx[i] * vy[i];
		}
		return res;
	}
	
	//takes a matrix and a vector parameters, returns the matrix product vector
	public static double[] mDotv(double[][] matrix, double[] vector) throws MismatchException{

		if(vector.length == matrix[0].length){
			double[] result = new double[matrix.length];
			if(vector.length > 1){
				for(int i = 0; i < matrix.length; i++){
					result[i] = 0.0;
					for(int j = 0; j < vector.length; j++){
						result[i] += matrix[i][j] * vector[j];
					}
				}
			}
			else{
				for(int i = 0; i < result.length; i++){
					result[i] = matrix[i][0] * vector[0];
				}
			}
			return result;
		}
		else{
			throw new MismatchException("vector and matrix dimensions don't  match"
					+ " for multiplication \n matrix dimensions : "
					+ matrix.length + " x " + matrix[0].length
					+ "; vector dimension: " + vector.length);
		}
		
	}
	
	public static double[][] transpose(double[][] matrix){
		double[][] result = new double[matrix[0].length][matrix.length];
		for(int i = 0; i < matrix.length; i++){
			for(int j = 0; j < matrix[0].length; j++){
				result[j][i] = matrix[i][j];
			}
		}
		return result;
	}
	
	/* helper method that returns an array of matrices 
	 * of the same dimensions as its parameter's. 
	 * all elements are initialized to 0.0*/
	public static double[][][] zeros(double[][][] matrix){
		double[][][] zeros =new double[matrix.length][][];
		
		for(int i = 0; i < zeros.length; i++){
			zeros[i] = new double[matrix[i].length][];
			for(int j = 0; j < zeros[i].length; j++){
				zeros[i][j] = new double[matrix[i][j].length];
				for(int k = 0; k < zeros[i][j].length; k++){
					zeros[i][j][k] = 0.0;
				}
			}
		}
		return zeros;
	}
	
	/* helper method that returns an array of vectors 
	 * of the same dimension as its parameter's. 
	 * all elements are initialized to 0.0*/
	public static double[][] zeros(double[][] matrice){
		double[][] zeros = new double[matrice.length][];
		for(int j = 0; j < zeros.length; j++){
			zeros[j] = new double[matrice[j].length];
			for(int k = 0; k < zeros[j].length; k++){
				zeros[j][k] = 0.0;
			}
		}
		return zeros;
	}
	
}
