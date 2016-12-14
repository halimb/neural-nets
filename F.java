package network;



public class F {
	
	//returns the sum of two or more vectors 
	public static double[] sum(double[] ...ds ){
		int size = ds[0].length;
		for(int i = 0; i < ds.length; i++){
			if(ds[i].length != size){
				throw new MismatchException("trying to add vectors of different dimensions."
											+ "dimension should be: " + size 
											+ "\nactual dimension" + ds[i].length);
			}
		}
		
		double[] res = new double[size];
		for(int i = 0; i < ds[0].length; i++){
			for(int j = 0; j < ds.length; j++){
				res[i] += ds[j][i];
			}
		}
		return res;
	}
	
	//subsracts vy from vs and returns the resulting vector 
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
		return (1.0/(1.0+Math.exp(-x)));
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
			x[i] = sigmoid(x[i])*(1 - sigmoid(x[i]));
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

		if(vector.length == matrix.length){
			for(int i = 0; i < vector.length; i++){
				for(int j = 0; j < matrix[i].length; j++){
					matrix[i][j] *= vector[i];
				}
			}
			double[] result = new double[matrix[0].length];
			for(int i = 0; i < result.length; i++){
				double res = 0.0;
				for(int j = 0; j < matrix.length; j++){
					//SUM THE COLUMNS
					res += matrix[j][i];
				}
				result[i] = res;
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
	
}