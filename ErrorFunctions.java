package com.yerboi.simpleperceptron;

import java.lang.Math;

public enum ErrorFunctions {
    
    SQUARE_ERROR ('S');
    
    private final char id;
    
    ErrorFunctions(char id) {
	this.id = id;
    }
    
    public double getOutput(double input, double expected) {
	switch (id) {
	case 'S':
	    return Math.pow(expected-input, 2);
	default:
	    return 0;
	}
    }
    
    public double getDerivative(double input, double expected) {
	switch (id) {
	case 'S':
	    return -2*(expected-input);
	default:
	    return 0;
	}
    }

}
