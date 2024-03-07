package com.yerboi.simpleperceptron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;

public class SimplePerceptronTest {
    
    public static void main(String[] args) {
	
	Perceptron perc = new Perceptron(new int[] {10,5,1}, 0.7, 0.75);
	//test code
	//System.out.println("Before backpropagation");
	//printNetwork(perc);
	//
	
	//double[] inputs = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
	double[] inputs = ThreadLocalRandom.current().doubles(1000).toArray();
	int[] inputsFinal = new int[1000];
	for (int i = 0; i < inputs.length; i++) {
	    inputs[i] = inputs[i]*10;
	    inputsFinal[i] = (int) inputs[i];
	}
	for (int i: inputsFinal) {
	    try {
		double[] inputVector = new double[10];
		for (int j = 0; j <= i; j++) {
		    inputVector[j] = 1.0; 
		}
		perc.setInputData(inputVector);
		double[] output = perc.getPerceptronOutput();
		perc.backpropagateError((i <= 5) ? (new double[] {0}) : (new double[] {1.0}));
	    } catch (Exception e) {
		e.printStackTrace();
	    }
	}
	
	//test code
	//System.out.println("After backpropagation");
	//printNetwork(perc);
	//end of test code
	
	Scanner sc = new Scanner(System.in);
	while (true) {
	    System.out.println("Enter a number between 0 and 9 (inclusive):");
	    int d = sc.nextInt();
	    double[] input = new double[10];
	    for (int i = 0; i <= d; i++) {
		input[i] = 1.0;
	    }
	    double halfwayMark = 0.0;
	    try {
		double[] minInput = {1,0,0,0,0,0,0,0,0,0};
		perc.setInputData(minInput);
		double min = perc.getPerceptronOutput()[0];
		double[] maxInput = {1,1,1,1,1,1,1,1,1,1};
		perc.setInputData(maxInput);
		double max = perc.getPerceptronOutput()[0];
		halfwayMark = max - (max-min)/2;
	    } catch (Exception e) {
		
	    }
	    try {
		perc.setInputData(input);
		System.out.println("Input: "+Arrays.toString(input));
		double[] output = perc.getPerceptronOutput();
		System.out.println("Output: "+Arrays.toString(output));
		if (output[0] < halfwayMark) {
		    System.out.println("Number is lower than 5!\n");
		} else {
		    System.out.println("Number is higher than 5!\n");
		}
	    } catch (Exception e) {
		e.printStackTrace();
	    }
	}
    }
    
    public static void printNetwork(Perceptron perc) {
	System.out.println("Now printing network:");
	ArrayList<Weights> weights = perc.getWeights();
	for (int i = 0; i < weights.size(); i++) {
	    System.out.println("Layer "+i+" weights");
	    double[][] weightSet = weights.get(i).getWeights();
	    for (int j = 0; j < weightSet.length; j++) {
		for (int k = 0; k < weightSet[j].length; k++) {
		    System.out.println(j+"-"+k+": "+weightSet[j][k]);
		}
	    }
	    System.out.println("Layer "+i+" bias");
	    double [] biasSet = weights.get(i).getBias();
	    for (int j = 0; j < biasSet.length; j++) {
		System.out.println(j+": "+biasSet[j]);
	    }
	    System.out.println();
	}
    }

}
