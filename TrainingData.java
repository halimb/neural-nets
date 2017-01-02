package network;

public class TrainingData {
	private double[] input;
	private double[] desired;
	
	public TrainingData(double[] input, double[] desired){
		this.input = input;
		this.desired = desired;
	}
	
	public double[] getInput() {
		return input;
	}
	
	public double[] getDesired() {
		return desired;
	}

}
