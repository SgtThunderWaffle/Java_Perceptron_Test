package com.yerboi.simpleperceptron;

import java.lang.Math;

public enum ActivationFunctions {
    
    RELU ('R'),
    SIGMOID('S');
    
    private final char id;
    
    ActivationFunctions(char id) {
	this.id = id;
    }
    
    public double getOutput(double input) {
	switch (this.id) {
	case 'R':
	    return ((input > 0) ? (input + Math.abs(input))/2 : 0);
	case 'S':
	    double sigmoid = 1/(1+Math.pow(Math.E, -input));
	    return sigmoid;
	default:
	    return 0;
	}
    }
    
    public double getDerivative(double input) {
	switch (id) {
	case 'R':
	    return ((input > 0) ? 1 : 0);
	case 'S':
	    return Math.pow(Math.E, -input)/Math.pow(Math.pow(Math.E, -input)+1, 2);
	default:
	    return 0;
	}
    }

}
