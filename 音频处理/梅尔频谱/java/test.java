package fft;

import java.util.Arrays;

public class test {

	
	static Complex omega(int N,int k) {
		return new Complex(Math.cos(-2*Math.PI/N*k),Math.sin(-2*Math.PI/N*k));
	}

	
	public static void main(String [] args) {
		
		Complex[] x = new Complex[8];
		for(int i=0;i<8;i++) {
			x[i] = new Complex(i,0);
		}
		
//		Complex [] ttt = new Complex[4];
//		ttt[0] = new Complex(0,0);
//		ttt[1] = new Complex(4,0);
//		ttt[2] = new Complex(8,0);
//		ttt[3] = new Complex(12,0);
//		
//		FFT r= new FFT();
//		System.out.println(Arrays.toString(r.DFT_slow(ttt)));

//		Complex [] result = r.fft(x);
//		for(int i=0;i<result.length;i++) {
//			System.out.println(result[i].toString());
//		}
		
		
		MEL mel = new MEL();
		double [][] mm = mel.mel(8000, 512, 40, 0.0, -1.0, false, 1);
		for(int i=0;i<mm.length;i++) {
			System.out.println(Arrays.toString(mm[i]));
		}
	}
	
	
}
