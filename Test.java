package network;

public class Test {

	public static void main(String[] args){
		
		Network net = new Network( 4, 2, 5, 1, 3, 6);
		double[] input = new double[4];
		input[0] = 0.7;
		input[1] = 0.2;
		input[2] = 0.9;
		input[3] = 0.2;
		net.ideal = new double[6];
		net.actual = new double[6];
		net.ideal[0] = 0.6;
		net.ideal[1] = 0.6;
		net.ideal[2] = 0.6;
		net.ideal[3] = 0.6;
		net.ideal[4] = 0.3;
		net.ideal[5] = 0.2;
		
		net.actual[0] = 0.5;
		net.actual[1] = 0.5;
		net.actual[2] = 0.5;
		net.actual[3] = 0.9;
		net.actual[4] = 0.1;
		net.actual[5] = 0.5;
		
		System.out.println(net);
		
		net.feedForward(input);
		net.calculateDeltas();
		System.out.println("\n\nAfter calculating deltas:\n");
		System.out.println(net + "\n");
		for(int layer = 2; layer <= net.numberOfLayers; layer++){
			System.out.print("Layer : " + layer + "\t");
			for(int j = 0; j < net.getDeltas(layer).length; j++){
				System.out.print(String.format("%.6f\t", net.getDeltas(layer)[j]));
			}
			System.out.println("\n");
		}
	}
}
