package network;

import java.util.Random;


public class MNISTTest {
	public static final String TRAINING_FILEPATH = "/MNIST_train.txt";
	public static final String TESTING_FILEPATH = "/MNIST_test.txt";
	public static TrainingData[] trainingData;
	public static TrainingData[] testData;
	
	public static void main(String[] args) throws Exception{

		trainingData = new MNISTParser(TRAINING_FILEPATH).getData();
		testData = new MNISTParser(TESTING_FILEPATH).getData();
		Network net = new Network(784, 30, 10);
		net.train(trainingData, 2.9, 10, 10);
	}
		
	
	public static void testNetwork(Network network, TrainingData[] testData, int testingSetSize){
		int setSize;
		if(testingSetSize == 0){
			setSize = testData.length;
		}
		else{
			setSize = testingSetSize;
		}
		
		Random random = new Random();
		int correctPredictionsCount = 0;
		double[] actual;
		double[] ideal;
		for(int i = 0; i < testingSetSize; i++){
			int index = random.nextInt(testData.length);
			TrainingData testDatum = testData[index];
			actual = network.feedForward(testDatum.getInput());
			ideal = testDatum.getDesired();
			int idealIndex = 42;
			int actualIndex = 13;
			double max = 0;
			for(int k = 0; k < actual.length; k++){
				if(actual[k] > max){
					actualIndex = k;
					max = actual[k];
				}
			}
			for(int j = 0; j < ideal.length; j++){
				if(ideal[j] > 0){
					idealIndex = j;
					break;
				}
			}
			
			if(actualIndex == idealIndex){
				correctPredictionsCount++;
			}
			if(i == testingSetSize - 1){
				System.out.println(" <  <  <  <");
				System.out.println("Ideal");
				for(int o = 0; o < ideal.length; o++){
					System.out.println(ideal[o]);
				}
				System.out.println("Actual");
				for(int o = 0; o < actual.length; o++){
					System.out.println(actual[o]);
				}
				System.out.println(" >  >  >  >");
			}
		}
		double accuracy = 100.0 * correctPredictionsCount/setSize;
		System.out.println(
				String.format("\nTested %d inputs, "
						+ "got %d correct predictions.\n> > > > Accuracy : %.2f%%< < < <",
						setSize, correctPredictionsCount, accuracy));
	}
}
