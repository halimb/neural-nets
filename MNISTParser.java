package network;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;

public class MNISTParser {
	private static final int NUM_PIXELS = 784; 
	private File mnistFile;
	
	public MNISTParser(String filePath){
		this.mnistFile = new File(filePath);
	}
	
	public TrainingData[] getData() throws IOException{
		BufferedReader reader = new BufferedReader( new FileReader(mnistFile));
		LineNumberReader lnr = new LineNumberReader( new FileReader(mnistFile));
		//counting the number of lines in the poker data file 
		int lines = 0;
		while(lnr.readLine() != null){
			lines++;
		}
		lnr.close();
		
		String[] values;
		TrainingData[] numbers = new TrainingData[lines];
		int i = 0;
		while(reader.ready()){
			values = reader.readLine().split(",");
			double[] input = new double[NUM_PIXELS];
			double[] desired = new double[10];
			int number = Integer.parseInt(values[0]);
			for(int j = 0; j < 10; j++){
				if( number == j){
					desired[j] = 1.0;
				}
				else{
					desired[j] = 0.0;
				}
			}
			
			for(int px = 1; px < NUM_PIXELS + 1; px++){
				input[px - 1] = Double.parseDouble(values[px]) / 255;
			}
			numbers[i] = new TrainingData(input, desired);
			i++;
		}
		System.out.println("Successfully read and parsed mnist data:");
		System.out.println(String.format("parsed %d training examples.", numbers.length));
		reader.close();
		return numbers;
	}
}
