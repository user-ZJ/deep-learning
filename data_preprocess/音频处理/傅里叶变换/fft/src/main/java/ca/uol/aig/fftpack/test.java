package com.java.fftpack;

import java.util.Arrays;

public class test {

	
	public static void main(String [] args) {
		double [] x = {1,2,3,4,5,6,7,8,9};
		RealDoubleFFT rdf = new RealDoubleFFT(x.length);
		rdf.ft(x);
		System.out.println(Arrays.toString(x));
		
		double [] complex = new double[]{1.0, 0.0,2.0, 0.0,1.0, 0.0,-1.0,0.0, 1.5,0.0,1.0, 0.0,2.0, 0.0,1.0, 0.0,-1.0,0.0, 1.5,0.0,1.5,0.0,1.5,0.0};
		ComplexDoubleFFT cdf = new ComplexDoubleFFT(complex.length/2);
		cdf.ft(complex);
		System.out.println(Arrays.toString(complex));
	}
	
	

}
