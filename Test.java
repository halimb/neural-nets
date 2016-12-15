package network;

import java.util.Random;


public class Test {
	public static int GLOBALCOUNTER = 0;
	public static final String TRAINING_FILEPATH = "src/network/poker-hand-training-true.data";
	public static final String TESTING_FILEPATH = "src/network/poker-hand-testing.data";
	public static void main(String[] args){
		
		Network net = new Network(10, 30,10);
		TrainingData[] trainingData = 
				new PokerDataParser(TRAINING_FILEPATH).getFormattedTrainingData();
		TrainingData[] testData = 
				new PokerDataParser(TESTING_FILEPATH).getFormattedTrainingData();
		
		net.train(trainingData, 1.9, 90);
		testNetwork(net, testData, 20000);
	}
	
	public static void testNetwork(Network network, TrainingData[] testData, int testingSetSize){
		Random random = new Random();
		int correctPredictionsCount = 0;
		for(int i = 0; i < testingSetSize; i++){
			int index = random.nextInt(testData.length);
			TrainingData testDatum = testData[index];
			double[] actual = network.feedForward(testDatum.getInput());
			double[] ideal = testDatum.getDesired();
			int idealIndex = 42;
			int actualIndex = 13;
			double max = 0;
			for(int j = 0; j < actual.length; j++){
				if(actual[j] > max){
					actualIndex = j;
					max = actual[j];
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
		}
		double batchSize = testingSetSize;
		double accuracy = 100.0 * correctPredictionsCount/batchSize;
		System.out.println(
				String.format("\nTested %d inputs, "
						+ "got %d correct predictions.\n Accuracy : %.2f%%",
						testingSetSize, correctPredictionsCount, accuracy));
	}
}
