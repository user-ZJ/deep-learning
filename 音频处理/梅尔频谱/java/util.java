package fft;

public class util {
	
	public double [] linspace(double min,double max,int n) {
		double [] y = new double[n];
		double grade = (max-min)/(n-1);
		y[0] = min;
		for(int i=0;i<n-1;i++) {
			y[i+1] = y[i] + grade;
		}
		return y;
	}
	
	public double [] mel_frequencies(int n_mels,double fmin,double fmax,boolean htk) {
		double min_mel = hz_to_mel(fmin, htk);
		double max_mel = hz_to_mel(fmax, htk=htk);
		double [] mels = linspace(min_mel, max_mel, n_mels);
		return mel_to_hz(mels, htk=htk);
	}
	
	
	public double [] hz_to_mel(double [] frequencies,boolean htk) {
		//usehtk:use HTK formula instead of Slaney
		int length = frequencies.length;
		double [] mels = new double[length];
		if(htk) {
			for(int i=0;i<length;i++) {
				mels[i] = 2595.0 * Math.log10(1.0 + frequencies[i] / 700.0);
			}
			return mels;
		}
		// Fill in the linear part
	    double f_min = 0;
	    double f_sp = 200f / 3;
	    // Fill in the log-scale part
	    double min_log_hz = 1000;                         //beginning of log region (Hz)
	    double min_log_mel = (min_log_hz - f_min) / f_sp; // same (Mels)
	    double logstep = Math.log(6.4) / 27.0;              // step size for log region
	    for(int i=0;i<length;i++) {
	    	if(frequencies[i]>=min_log_hz) {
	    		mels[i] = min_log_mel + Math.log(frequencies[i]/min_log_hz)/logstep;
	    	}else {
	    		mels[i] = (frequencies[i]-f_min)/f_sp;
	    	}
	    }
		return mels;
	}
	
	public double hz_to_mel(double frequencies,boolean htk) {
		double mel;
		if(htk) {
			return 2595.0 * Math.log10(1.0 + frequencies / 700.0);
		}
		// Fill in the linear part
	    double f_min = 0;
	    double f_sp = 200f / 3;
	    // Fill in the log-scale part
	    double min_log_hz = 1000;                         //beginning of log region (Hz)
	    double min_log_mel = (min_log_hz - f_min) / f_sp; // same (Mels)
	    double logstep = Math.log(6.4) / 27.0;              // step size for log region
	    if(frequencies>=min_log_hz) {
    		mel = min_log_mel + Math.log(frequencies/min_log_hz)/logstep;
    	}else {
    		mel = (frequencies-f_min)/f_sp;
    	}
	    return mel;
	}
	
	public double [] mel_to_hz(double [] mels,boolean htk) {
		int length = mels.length;
		double [] freqs = new double[length];
		if(htk) {
			for(int i=0;i<length;i++) {
				freqs[i] = 700.0 * (Math.pow(10.0, mels[i]/2595.0)-1.0); 
			}
			return freqs;
		}
		 //Fill in the linear scale
	    double f_min = 0.0;
		double f_sp = 200.0 / 3;

		//And now the nonlinear scale
		double min_log_hz = 1000.0;                         // beginning of log region (Hz)
		double min_log_mel = (min_log_hz - f_min) / f_sp;   // same (Mels)
		double logstep = Math.log(6.4) / 27.0;                // step size for log region
		for(int i=0;i<length;i++) {
			if(mels[i] >= min_log_mel) {
				freqs[i] = min_log_hz * Math.exp(logstep * (mels[i] - min_log_mel));
			}else {
				freqs[i] = f_min + f_sp * mels[i];
			}
		}
		return freqs;
	}
	
	public double mel_to_hz(double mels,boolean htk) {
		double freqs=0.0;
		if(htk) {
			freqs = 700.0 * (Math.pow(10.0, mels/2595.0)-1.0); 
			return freqs;
		}
		 //Fill in the linear scale
	    double f_min = 0.0;
		double f_sp = 200.0 / 3;

		//And now the nonlinear scale
		double min_log_hz = 1000.0;                         // beginning of log region (Hz)
		double min_log_mel = (min_log_hz - f_min) / f_sp;   // same (Mels)
		double logstep = Math.log(6.4) / 27.0;                // step size for log region
		if(mels >= min_log_mel) {
			freqs = min_log_hz * Math.exp(logstep * (mels - min_log_mel));
		}else {
			freqs = f_min + f_sp * mels;
		}
		return freqs;
	}

	public double[] diff(double[] mel_f) {
		// TODO Auto-generated method stub
		double [] diff = new double[mel_f.length-1];
		for(int i=0;i<mel_f.length-1;i++) {
			diff[i] = mel_f[i+1]-mel_f[i];
		}
		return diff;
	}

	public double[][] outer(double[] mel_f, double[] fftfreqs) {
		// TODO Auto-generated method stub
		double [][] ramps = new double[mel_f.length][fftfreqs.length];
		for(int i=0;i<mel_f.length;i++) {
			for(int j=0;j<fftfreqs.length;j++) {
				ramps[i][j] = mel_f[i] - fftfreqs[j];
			}
		}
		return ramps;
	}

}
