package network;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;

public class PokerDataParser {
	private TrainingData[] formatted;
	
	//CONSTRUCTOR
	public PokerDataParser(String filePath){
		try(BufferedReader reader =
				new BufferedReader(
						new FileReader(filePath));
			LineNumberReader lnr = new LineNumberReader(new FileReader(filePath)))
		{
			//counting the number of lines in the poker data file 
			int linesNumber = 0;
			while(lnr.readLine() != null){
				linesNumber++;
			}
			System.out.println(linesNumber);
			
			formatted = new TrainingData[linesNumber];
			String currentLine;
			int lineNumber = 0;
			while((currentLine = reader.readLine()) != null){
				
				String[] datums = currentLine.split(",");
				double[] input = new double[10];
				double[] ideal = new double[10];
				
				/* populating the input vector which consists of pairs of card 
				 * number (from 1 to 13) and card colors (from 1 to 4, standing
				 * for hearts, clubs...etc)*/
				for(int i = 0; i < datums.length - 1; i++){
					input[i] = Double.parseDouble(datums[i])/10.0;
				}
				
				/* setting the ideal ouput vector, a each neuron represents
				 * a poker hand in the order : { Nothing; One pair; Two pairs;
				 * Three of a kind; Straight; Flush; Full house; Four of a kind; 
				 * Straight flush; Royal flush } if hit, the value of that output
				 * is set to 1.0, else :0.0*/
				int pokerHand = Integer.parseInt(datums[10]);
				for(int i = 0; i < 10; i++){
					if(pokerHand == i){
						ideal[i] = 1.0;
					}
					else{
						ideal[i] = 0.0;
					}
				}
				formatted[lineNumber] = new TrainingData(input, ideal);
				lineNumber++;
			}
		}
		catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public TrainingData[] getFormattedTrainingData(){
		return formatted;
	}
}
