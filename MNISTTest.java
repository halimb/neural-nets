package network;

import java.util.Random;


public class MNISTTest {
	/* MNIST_train: 60000 MNIST (label, features) pairs */
	/* MNIST_test: 10000 MNIST (label, features) pairs */
	public static final String TRAINING_FILEPATH = "MNIST_train.txt";
	public static final String TESTING_FILEPATH = "MNIST_test.txt";
	public static TrainingData[] trainingData;
	public static TrainingData[] testData;
	
	public static void main(String[] args) throws Exception{

		trainingData = new MNISTParser(TRAINING_FILEPATH).getData();
		testData = new MNISTParser(TESTING_FILEPATH).getData();
		Network net = new Network(784, 30, 10);
		net.train(trainingData, 2.9, 10, 10, 30);
	}
	
	public static void testNetwork(Network network, TrainingData[] testData){
		int idealIndex = 0;
		int actualIndex = 0;
		int setSize = testData.length;
		Random random = new Random();
		int count = 0;
		double[] actual = null;
		double[] ideal = null;
		for(int i = 0; i < setSize; i++){
			int index = random.nextInt(setSize);
			TrainingData testDatum = testData[index];
			actual = network.feedForward(testDatum.getInput());
			ideal = testDatum.getDesired();
			double max = 0;
			
			for(int k = 0; k < actual.length; k++){
				if(actual[k] > max){
					actualIndex = k;
					max = actual[k];
				}
			}
			
			for(int j = 0; j < ideal.length; j++){
				if(ideal[j] == 1){
					idealIndex = j;
					break;
				}
			}
			
			if(actualIndex == idealIndex){
				count++;
			}
			
		}
		System.out.println(" <  <  <  <");
		System.out.println("Ideal\tActual");
		for(int o = 0; o < ideal.length; o++){
			System.out.println(ideal[o] + "\t" + actual[o]);
		}
		System.out.println(" >  >  >  >");
		double accuracy = 100.0 * count/setSize;
		System.out.println(
				String.format("\nTested %d inputs, "
						+ "got %d correct predictions."
						+ "\n> > > > Accuracy : %.2f%%< < < <",
						setSize, count, accuracy));
	}
}
