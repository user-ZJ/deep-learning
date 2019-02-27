package fft;

import java.util.Arrays;

public class MEL {
	
	util ut = new util();
	
	//²úÉúÃ·¶ûÂË²¨Æ÷
	public double [][] mel(int sr,int n_fft,int n_mels,double fmin,double fmax,boolean htk,int norm) {
	    if(fmax<0)
	        fmax = sr / 2;
	    double [][] weights = new double[n_mels][1+n_fft/2];
	    double [] fftfreqs = ut.linspace(0.0, sr/2, 1+n_fft/2);
	    double [] mel_f = ut.mel_frequencies(n_mels + 2, fmin, fmax, htk);
	    double [] fdiff = ut.diff(mel_f);
	    double [][] ramps = ut.outer(mel_f,fftfreqs);
	    
	    for(int i=0;i<n_mels;i++) {
	    	// lower and upper slopes for all bins
	    	for(int j=0;j<fftfreqs.length;j++) {	
	    		double lower = -ramps[i][j]/fdiff[i];
	    		double upper = ramps[i+2][j]/fdiff[i+1];
	    		weights[i][j] = Math.max(0, Math.min(lower, upper));
	    	}
	    }
	 
	    if(norm == 1) {
	    	for(int i=0;i<n_mels;i++) {
	    		double enorm = 2.0/(mel_f[i+2]-mel_f[i]);
	    		for(int j=0;j<1+n_fft/2;j++) {
	    			weights[i][j] *= enorm;
	    		}
	    	}
	    }
	    
		return weights;
	}

}
